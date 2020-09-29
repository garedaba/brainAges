import numpy as np
import pandas as pd

import os, glob
import yaml
from tqdm import tqdm

from functions.surfaces import load_surf_data, parcellateSurface
from functions.models import get_ensemble_model, get_linear_model, get_nonlinear_model
from functions.models import get_model_explanations, get_age_corrected_model_explanations, correct_age_predictions
from functions.misc import pre_process_metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error

def main():

    #####################################################################################################
    # CONFIG
    # load configuration file and set parameters accordingly
    #####################################################################################################
    with open("config.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    print('')
    print('---------------------------------------------------------')
    print('Configuration:')
    print(yaml.dump(cfg, default_flow_style=False, default_style=''))
    print('---------------------------------------------------------')
    print('')

    # set paths
    datapath = cfg['paths']['datapath']
    metricpath = datapath + 'surfaces/'
    outpath = cfg['paths']['results']
    genpath = cfg['paths']['genpath']

    # other params - whether to regress out global metrics and run PCA
    preprocessing_params = cfg['preproc']
    regress = 'Corrected' if preprocessing_params['regress'] else 'Raw'
    run_pca = 'PCA' if preprocessing_params['pca'] else 'noPCA'
    run_combat = 'Combat' if preprocessing_params['combat'] else 'noCombat'

    # cortical parcellation
    parc = cfg['data']['parcellation']

    # k-fold CV params
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #####################################################################################################
    # LOADING
    # load surface data and parcellate into regions
    # load existing parcellated data if it's already been calculated
    #####################################################################################################

    print('---------------------------------------------------------')
    print('loading data')
    print('---------------------------------------------------------')

    # load age, sex, site data
    subject_data = pd.read_csv(datapath + 'Participant_MetaData.csv')

    # load image data
    metric_names = ['thickness', 'area']
    metric_data = []
    for metric in metric_names:
        # check if parcellated data already exists
        chkfile = genpath + metric + '-parcellated-data-' + parc + '.csv'
        if os.path.exists(chkfile):
            print('parcellated ' + metric + ' data already created, loading existing files')
            metric_data.append(np.loadtxt(chkfile, delimiter=','))
        # otherwise load surfaces and parcellate
        else:
            lh_files = sorted(glob.glob(metricpath + '/*.lh.' + metric + '_fs5.s10.mgh'))
            rh_files = sorted(glob.glob(metricpath + '/*.rh.' + metric + '_fs5.s10.mgh'))

            image_data = load_surf_data(lh_files, rh_files)
            zero_vector = (np.sum(image_data, axis=0)==0).astype(int)

            parc_data = parcellateSurface(image_data[:,zero_vector==0], zero_vector, parc=parc)
            # save for later runs
            np.savetxt(chkfile, parc_data, delimiter=',')

            metric_data.append(parc_data)

    #####################################################################################################
    # K-FOLD
    #####################################################################################################
    # some variables for later
    n_subs = np.shape(metric_data[0])[0]
    n_features = np.shape(metric_data[0])[1]

    # space for predictions and explanations
    preds = np.zeros((n_subs, 3))
    uncorr_preds = np.zeros((n_subs, 3))
    fold = np.zeros((n_subs, 1))
    feature_explanations = np.zeros((3, n_subs, n_features*2))
    sample_mean_feature_explanations = np.zeros((3, n_subs, n_features*2))

    fold_predictions = np.zeros((3, n_subs, 5)) #num models x num subs x num folds
    fold_feature_explanations = np.zeros((3, n_subs, n_features*2, 5)) # num models x num subs x num features x num folds

    linear_model_coefficients = np.zeros((5, n_features*2)) # num_folds x features

    # cross-validation - stratified by site
    for n, (train_idx, test_idx) in enumerate(skf.split(np.arange(n_subs), subject_data.Site)):
        print('')
        print('FOLD {:}:------------------------------------------------'.format(n+1))

        # age data for train and test sets
        train_y, test_y = subject_data.Age[train_idx], subject_data.Age[test_idx]

        # run any required processing separetely on metric data (thickness and area)
        train_x, test_x = pre_process_metrics(metric_names, metric_data, subject_data, train_idx, test_idx, preprocessing_params)

        # run models in each fold
        fold[test_idx] = n+1
        print('')
        # get pre-specified models as pipelines with randomised nested CV parameter search, with or without PCA
        models = [get_linear_model(preprocessing_params),
                   get_nonlinear_model(preprocessing_params),
                   get_ensemble_model(preprocessing_params)]

        # for each model
        for m, (model_name, model) in enumerate(zip(['linear', 'nonlinear','ensemble'], models)):

            # FIT
            print('fitting {:} model'.format(model_name))
            model.fit(train_x, train_y)

            # PREDICT
            train_predictions = model.predict(train_x)
            test_predictions = model.predict(test_x)

            # CORRECT FOR AGE EFFECT
            uncorr_preds[test_idx, m] = test_predictions
            preds[test_idx, m] = correct_age_predictions(train_predictions, train_y, test_predictions, test_y)

            # collate brain age delta for later models
            fold_predictions[m, train_idx, n] =  train_predictions - train_y
            fold_predictions[m, test_idx, n] =  test_predictions - test_y

            print('{:} model: r2 score = {:.2f}'.format(model_name, r2_score(test_y, uncorr_preds[test_idx, m])))
            print('{:} correlation between age and delta = {:.2f}'.format(model_name, np.corrcoef(test_y, uncorr_preds[test_idx, m]-test_y)[0,1]))
            print('{:} correlation between age and corrected delta = {:.2f}'.format(model_name, np.corrcoef(test_y, preds[test_idx, m]-test_y)[0,1]))
            print('{:} model: corrected r2 score = {:.2f}'.format(model_name, r2_score(test_y, preds[test_idx, m])))

            # EXPLAIN
            print('calculating {:} model explanations for test data'.format(model_name))
            exp_features = round(np.shape(train_x)[1]/2) # at most 50% of regions used in explanation
            test_model_explanations = np.zeros((np.shape(test_x)[0], np.shape(test_x)[1]))
            train_model_explanations = np.zeros((np.shape(train_x)[0], np.shape(train_x)[1]))
            sample_mean_model_explanations = np.zeros((np.shape(test_x)[0], np.shape(test_x)[1]))

            # test explanations
            for s in tqdm(np.arange(len(test_x))):
                test_model_explanations[s,:] = get_age_corrected_model_explanations(model, train_x, train_y, test_x[s,:].reshape(1,-1),
                                                                                    age=test_y.iloc[s], num_features=exp_features)
            sample_mean_model_explanations = get_model_explanations(model, train_x, test_x, num_features=exp_features)

            # train explanations - for training set examples
            for s in tqdm(np.arange(len(train_x))):
                train_model_explanations[s,:] = get_age_corrected_model_explanations(model, train_x, train_y, train_x[s,:].reshape(1,-1),
                                                                                    age=train_y.iloc[s], num_features=exp_features)

            # collate
            feature_explanations[m, test_idx, :] = test_model_explanations
            sample_mean_feature_explanations[m, test_idx, :] = sample_mean_model_explanations
            fold_feature_explanations[m, test_idx, :, n] = test_model_explanations
            fold_feature_explanations[m, train_idx, :, n] = train_model_explanations

            # also keep model coefficients for linear models
            if model_name == 'linear':
                if run_pca == 'PCA':
                    # not sure about this bit...
                    model_coefficients = model.best_estimator_['pca'].components_.T.dot( model.best_estimator_['model'].coef_)
                else:
                    model_coefficients = model.best_estimator_['model'].coef_
                linear_model_coefficients[n, :] = model_coefficients


    #####################################################################################################
    # RESULTS
    #####################################################################################################
    print('---------------------------------------------------------')
    print('compiling results')
    print('---------------------------------------------------------')
    # collate data
    preds = pd.DataFrame(np.hstack((preds, uncorr_preds)), columns = ['linear_preds', 'nonlinear_preds', 'ensemble_preds', 'linear_uncorr_preds', 'nonlinear_uncorr_preds', 'ensemble_uncorr_preds'])
    fold = pd.DataFrame(fold.astype(int), columns=['fold'])
    predictions = pd.concat((subject_data, fold, preds), axis=1)

    # saving
    print('model predictions: {:}model_predictions-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc))
    print('')
    predictions.to_csv('{:}model_predictions-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc), index=False)

    # accuracies and AUC
    n_fold = len(np.unique(predictions.fold))
    models = ['linear', 'nonlinear', 'ensemble']

    fold_mae = np.zeros((n_fold, len(models)*2))
    fold_r2 = np.zeros((n_fold, len(models)*2))

    for n, f in enumerate(np.unique(predictions.fold)):
        for m, model in enumerate(models):
            fold_mae[n, m] = mean_absolute_error(predictions.Age[predictions.fold==f], predictions[model+'_preds'][predictions.fold==f])
            fold_mae[n, m+3] = mean_absolute_error(predictions.Age[predictions.fold==f], predictions[model+'_uncorr_preds'][predictions.fold==f])

            fold_r2[n, m] = r2_score(predictions.Age[predictions.fold==f], predictions[model+'_preds'][predictions.fold==f])
            fold_r2[n, m+3] = r2_score(predictions.Age[predictions.fold==f], predictions[model+'_uncorr_preds'][predictions.fold==f])

    fold_mae = pd.DataFrame(fold_mae, columns=['linear', 'nonlinear', 'ensemble','linear_uncorr', 'nonlinear_uncorr', 'ensemble_uncorr'])
    fold_mae.insert(0, 'fold', np.unique(predictions.fold))

    fold_r2 = pd.DataFrame(fold_r2, columns=['linear', 'nonlinear', 'ensemble','linear_uncorr', 'nonlinear_uncorr', 'ensemble_uncorr'])
    fold_r2.insert(0, 'fold', np.unique(predictions.fold))

    # saving
    print('model accuracy (MAE): {:}MAE-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc))
    fold_mae.to_csv('{:}MAE-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc), index=False)
    print('model accuracy (R2): {:}R2-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc))
    fold_r2.to_csv('{:}R2-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc), index=False)
    print('')

    # explanations
    for m, model_name in enumerate(['linear', 'nonlinear', 'ensemble']):
        exp = pd.DataFrame(feature_explanations[m])
        mean_exp = pd.DataFrame(sample_mean_feature_explanations[m])
        fold = pd.DataFrame(fold.astype(int), columns=['fold'])
        feat_exp = pd.concat((subject_data, fold, exp), axis=1)
        mean_feat_exp = pd.concat((subject_data, fold, mean_exp), axis=1)
        print('model explanations: {:}{:}-model-feature-explanations-{:}-{:}-{:}-{:}.csv'.format(genpath, model_name, run_combat, regress, run_pca, parc))
        print('')
        feat_exp.to_csv('{:}{:}-model-feature-explanations-{:}-{:}-{:}-{:}.csv'.format(genpath, model_name, run_combat, regress, run_pca, parc), index=False)
        mean_feat_exp.to_csv('{:}{:}-model-group-mean-feature-explanations-{:}-{:}-{:}-{:}.csv'.format(genpath, model_name, run_combat, regress, run_pca, parc), index=False)

        # save for later CV models
        print('model explanations for cross-validation: {:}{:}-model-all-fold-feature-explanations-{:}-{:}-{:}-{:}.npy'.format(genpath, model_name, run_combat, regress, run_pca, parc))
        np.save('{:}{:}-model-all-fold-feature-explanations-{:}-{:}-{:}-{:}.npy'.format(genpath, model_name, run_combat, regress, run_pca, parc), fold_feature_explanations[m,:,:,:])
        print('model predictions for cross-validation: {:}{:}-model-all-fold-delta-{:}-{:}-{:}-{:}.npy'.format(genpath, model_name, run_combat, regress, run_pca, parc))
        np.save('{:}{:}-model-all-fold-delta-{:}-{:}-{:}-{:}.npy'.format(genpath, model_name, run_combat, regress, run_pca, parc), fold_predictions[m,:,:])

        if model_name == 'linear':
            fold_col = pd.DataFrame(np.arange(5).reshape(-1,1), columns=['fold'])
            coef = pd.DataFrame(linear_model_coefficients)
            feat_coef = pd.concat((fold_col, coef), axis=1)
            print('model coefficients: {:}{:}-model-feature-coefficients-{:}-{:}-{:}-{:}.csv'.format(genpath, model_name, run_combat, regress, run_pca, parc))
            print('')
            feat_coef.to_csv('{:}{:}-model-feature-coefficients-{:}-{:}-{:}-{:}.csv'.format(genpath, model_name, run_combat, regress, run_pca, parc), index=False)


if __name__ == '__main__':
    main()

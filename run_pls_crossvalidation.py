import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import yaml, os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from functions.misc import deconfound
from functions.pls_models import plsr_training_curve, run_plsr

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
    outpath = cfg['paths']['results']
    genpath = cfg['paths']['genpath']

    # other params - whether to regress out global metrics and run PCA
    preprocessing_params = cfg['preproc']
    regress = 'Corrected' if preprocessing_params['regress'] else 'Raw'
    run_pca = 'PCA' if preprocessing_params['pca'] else 'noPCA'
    run_combat = 'Combat' if preprocessing_params['combat'] else 'noCombat'

    # cortical parcellation
    parc = cfg['data']['parcellation']

    # for pls
    ss = StandardScaler()

    # for each model
    for model in ['linear', 'nonlinear', 'ensemble']:
        print('')
        print('###### MODEL: {:} ##############'.format(model))

        #####################################################################################################
        # LOADING
        # load previously calculated model predictions and explanations
        #####################################################################################################
        # load data
        # n_sub x n_feature x n_fold array: for each fold, explanations calculated using model trained on training folds
        model_explanations = np.load('{:}{:}-model-all-fold-feature-explanations-{:}-{:}-{:}-{:}.npy'.format(genpath, model, run_combat, regress, run_pca, parc))
        # n_sub x n_folds: for each fold, deltas (uncorrected) predictions using model trained on training folds
        model_predictions = np.load('{:}{:}-model-all-fold-delta-{:}-{:}-{:}-{:}.npy'.format(genpath, model, run_combat, regress, run_pca, parc))

        # metadata - get this elsewhere
        subject_info = subject_data = pd.read_csv(datapath + 'Participant_MetaData.csv')

        # folds
        cv_preds = pd.read_csv('{:}model_predictions-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc))
        cv_folds = cv_preds.fold.values

        # metric data
        metric_names = ['thickness', 'area']
        metric_data = []
        for metric in metric_names:
            chkfile = genpath + metric + '-parcellated-data-' + parc + '.csv'
            metric_data.append(np.loadtxt(chkfile, delimiter=','))

        # confounds
        confounds = pd.DataFrame()
        confounds['age'] = subject_info.Age
        confounds['sex'] = subject_info.Male
        confounds['meanthick'] = np.mean(metric_data[0],1)
        confounds['meanarea'] = np.mean(metric_data[1],1)

        #########################################################################################################
        # DECONFOUND
        # calculate deconfounded explanations and deltas for PLS within CV folds
        #########################################################################################################
        # partial out confounds
        explanations_deconf, cv_explanations_deconf, delta_deconf = deconfound_train_test_data(model_explanations, model_predictions, confounds, cv_folds)

        #########################################################################################################
        # PLS
        ##########################################################################################################################
        #####################################################################################################
        # MODEL TESTING I. - Predicting deconfounded brain age delta from deconfounded model explanations...
        # calculate training curve (within 5-fold CV) for different numbers of PLS components
        #####################################################################################################
        component_choice = np.arange(10)+1

        print('')
        print('performing cross-validation for model: {:}'.format(model))
        print('number of PLS components from: {:}'.format(component_choice))

        fold_train_accuracy, fold_test_accuracy = pls_train_test_training_curve(explanations_deconf, delta_deconf, cv_folds, component_choice)

        #####################################################################################################
        # MODEL TESTING II.- Predicting brain age delta from model explanations...
        # within 5-fold CV, calculate spatial maps for each component
        #####################################################################################################
        plsr_comps = 1

        # outputs
        target_delta = np.zeros((len(delta_deconf)))
        predicted_delta = np.zeros((len(delta_deconf)))
        feature_loadings = np.zeros((np.shape(explanations_deconf)[1], plsr_comps, 5))  # num_features, num_comps, num_folds
        explained_delta_var = np.zeros((plsr_comps, 5))
        explained_image_var = np.zeros((plsr_comps, 5))
        subject_scores = np.zeros((np.shape(explanations_deconf)[0], plsr_comps))

        print('')
        print('for n components = {:}'.format(plsr_comps))
        print('cross-validating PLS model')
        for f in np.arange(5):
            train_data = explanations_deconf[cv_folds!=f+1,:,f]
            test_data = explanations_deconf[cv_folds==f+1,:,f]

            # preprocess with scaling
            train_X, test_X = ss.fit_transform(train_data), ss.transform(test_data)

            train_Y, test_Y = delta_deconf[cv_folds!=f+1, f], delta_deconf[cv_folds==f+1, f]

            regional_loadings, component_scores, component_loadings, coefs, weights = run_plsr(train_X, train_Y,  n_comps=plsr_comps)

            # collect
            target_delta[cv_folds==f+1] = test_Y
            predicted_delta[cv_folds==f+1] = test_X.dot(coefs.reshape(-1))

            # add norm back in
            rescaled_regional_loadings = np.multiply(regional_loadings, np.linalg.norm(train_X, axis=0).reshape(np.shape(train_X)[1],1))
            feature_loadings[:,:,f] = rescaled_regional_loadings

            # explained variance
            explained_delta_var[:,f] = component_loadings**2
            # predicted subject scores
            subject_scores[cv_folds==f+1,:] = test_X.dot(weights)

            # add norm back in to calculate correctly
            for c in np.arange(plsr_comps):
                explained_image_var[c,f] = r2_score(train_X, component_scores[:,[c]].dot(rescaled_regional_loadings[:,[c]].T))

        #########################################################################################################
        # save out - PLSR results
        #########################################################################################################
        # deconfounded data arrays
        print('deconfounded model explanations for surrogate analysis: {:}{:}-model-all-fold-deconfounded-feature-explanations-{:}-{:}-{:}-{:}.npy'.format(genpath, model, run_combat, regress, run_pca, parc))
        np.save('{:}{:}-model-all-fold-deconfounded-feature-explanations-{:}-{:}-{:}-{:}.npy'.format(genpath, model, run_combat, regress, run_pca, parc), explanations_deconf)
        print('deconfounded model deltas for surrogate analysis: {:}{:}-model-all-fold-deconfounded-delta-{:}-{:}-{:}-{:}.npy'.format(genpath, model, run_combat, regress, run_pca, parc))
        np.save('{:}{:}-model-all-fold-deconfounded-delta-{:}-{:}-{:}-{:}.npy'.format(genpath, model, run_combat, regress, run_pca, parc), delta_deconf)
        print('')

        # TRAINING CURVE
        train_results = pd.DataFrame(np.hstack((component_choice[:,np.newaxis],fold_train_accuracy)), columns=['num_comp','1','2','3','4','5'])
        train_results.insert(0, 'group','train')
        test_results = pd.DataFrame(np.hstack((component_choice[:,np.newaxis],fold_test_accuracy)), columns=['num_comp','1','2','3','4','5'])
        test_results.insert(0, 'group','test')
        out_results = pd.concat((train_results, test_results)).melt(id_vars=['group', 'num_comp'], var_name=['fold'])
        print('see: {:}{:}-delta-PLSR-component-results-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc))
        out_results.to_csv('{:}{:}-delta-PLSR-component-results-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=False)

        # PLS CV
        # feature importance
        for p in np.arange(plsr_comps):
            pd.DataFrame(feature_loadings[:,p,:]).T.to_csv('{:}{:}-feature-loadings-PLS-CV-component-{:}-{:}-{:}-{:}-{:}.csv'.format(outpath, model, p+1, run_combat, regress, run_pca, parc), index=False)
        # averaged over fold
        print('see: {:}{:}-mean-feature-loadings-PLS-CV-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc))
        pd.DataFrame(np.mean(feature_loadings, axis=2)).T.to_csv('{:}{:}-mean-feature-loadings-PLS-CV-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=False)
        # delta predictions
        deltas = cv_preds[model + '_uncorr_preds']-cv_preds.Age
        delta_predictions = pd.DataFrame((cv_folds, deltas, target_delta, predicted_delta)).T
        delta_predictions.columns = ['fold','delta', 'deconfounded_delta','predicted_delta']
        print('see: {:}{:}-delta-predictions-PLS-CV-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc))
        delta_predictions.to_csv('{:}{:}-delta-predictions-PLS-CV-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=False)
        # subject scores
        subject_scores = pd.DataFrame(np.hstack((subject_info.ID[:,np.newaxis], cv_folds[:,np.newaxis], deltas[:,np.newaxis], target_delta[:,np.newaxis], subject_scores)))
        subject_scores.columns = ['id','fold','deltas', 'deconfounded_delta'] + ['PLS{:}'.format(i+1) for i in np.arange(plsr_comps)]
        print('see: {:}{:}-subject_scores-PLS-CV-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc))
        subject_scores.to_csv('{:}{:}-subject_scores-PLS-CV-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=False)

        # explained variances (in training data)
        colnames = ['fold']
        for i in np.arange(plsr_comps):
            colnames.append('comp{:}'.format(i+1))
        tmpa = pd.DataFrame(np.hstack(((np.arange(5)+1).reshape(-1,1),explained_delta_var.T)), columns=colnames)
        tmpa.insert(0, 'type', 'delta')
        tmpb = pd.DataFrame(np.hstack(((np.arange(5)+1).reshape(-1,1),explained_image_var.T)), columns=colnames)
        tmpb.insert(0, 'type', 'image')
        print('see: {:}{:}-explained_variance-PLS-CV-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc))
        pd.concat((tmpa, tmpb)).to_csv('{:}{:}-explained_variance-PLS-CV-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=False)

def deconfound_train_test_data(explanations, deltas, confounds, fold_ids):
    """
    split data according to prespecific folds and deconfound explaination data and deltas within folds

    parameters
    ----------
    explainations: n_sub x n_feature x n_fold array, previously calculated model explaination for each fold
    deltas: n_sub x n_fold array, previously calculated age deltas for each fold
    confounds: n_sub x n_var, dataframe of confound variables for each subject
    fold_ids: n_sub array, fold label for each subject

    returns
    ------
    explanations_deconf: n_sub x n_feature x n_fold array, deconfounded explanation data (model calculated within training data and applied to test data in each fold)
    cv_explanations_deconf: n_sub x n_feature, deconfounded data for each subject in test set of each fold
    delta_deconf: n_sub x n_fold, deconfounded delta (from model calculated with training data in each fold)
    """
    print('')
    print('deconfounding')

    explanations_deconf = np.zeros_like(explanations)
    cv_explanations_deconf = np.zeros_like(explanations[:,:,0])
    delta_deconf = np.zeros_like(deltas)

    for f in np.arange(np.max(fold_ids)):
        fold = f+1

        # explanations = explained using models fit on train data
        f_explanations = explanations[:,:,f]
        train_data, test_data = f_explanations[fold_ids!=fold], f_explanations[fold_ids==fold]

        # split confounds
        train_confounds, test_confounds = confounds[fold_ids!=fold], confounds[fold_ids==fold]

        # deltas, based on training folds
        f_deltas = deltas[:,f]
        train_deltas, test_deltas = f_deltas[fold_ids!=fold], f_deltas[fold_ids==fold]

        # remove variance due to confounds from explanations (PCA to improve conditioning)
        train_data_deconf, test_data_deconf = deconfound(train_data, train_confounds, test_data=test_data, test_confounds=test_confounds)

        # collate
        explanations_deconf[fold_ids!=fold,:, f] = train_data_deconf
        explanations_deconf[fold_ids==fold,:, f] = cv_explanations_deconf[fold_ids==fold,:] = test_data_deconf

        # remove variance in delta due to confounds
        train_delta_deconf, test_delta_deconf = deconfound(train_deltas, train_confounds, test_data = test_deltas, test_confounds=test_confounds)
        delta_deconf[fold_ids!=fold, f] = train_delta_deconf
        delta_deconf[fold_ids==fold, f] = test_delta_deconf

    return explanations_deconf, cv_explanations_deconf, delta_deconf


def pls_train_test_training_curve(explanations_deconfounded, delta_deconfounded, fold_ids, component_choice):
    """
    run PLS training curve within train-test folds

    parameters
    ---------
    explanations_deconf: n_sub x n_feature x n_fold array, deconfounded explanation data (model calculated within training data and applied to test data in each fold)
    delta_deconf: n_sub x n_fold, deconfounded delta (from model calculated with training data in each fold)
    fold_ids: n_sub array, fold label for each subject
    component_choice: list of PLS component number to run

    returns
    -------
    fold_train_accuracy, fold_test_accuracy: model accuracies for train and test data for each number of components

    """
    ss = StandardScaler()

    fold_train_accuracy = np.zeros((10,5))
    fold_test_accuracy = np.zeros((10,5))

    for f in np.arange(np.max(fold_ids)):
        fold = f+1
        # get model explanations for each fold (estimated in train/test split)
        train_data = explanations_deconfounded[fold_ids!=fold, :, f]
        test_data = explanations_deconfounded[fold_ids==fold, :, f]

        # scale for preprocessing
        train_X, test_X = ss.fit_transform(train_data), ss.transform(test_data)

        # use deconfounded brain age delta as target
        train_Y, test_Y = delta_deconfounded[fold_ids!=fold, f], delta_deconfounded[fold_ids==fold, f]

        # fit training curve
        train_acc, test_acc = plsr_training_curve(train_X, train_Y, test_X, test_Y, components=component_choice)
        fold_train_accuracy[:,f] = train_acc
        fold_test_accuracy[:,f] = test_acc

    return fold_train_accuracy, fold_test_accuracy

if __name__ == '__main__':
    main()

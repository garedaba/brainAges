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
    outpath = cfg['paths']['results']
    genpath = cfg['paths']['genpath']

    # other params - whether to regress out global metrics and run PCA
    preprocessing_params = cfg['preproc']
    regress = 'Corrected' if preprocessing_params['regress'] else 'Raw'
    run_pca = 'PCA' if preprocessing_params['pca'] else 'noPCA'
    run_combat = 'Combat' if preprocessing_params['combat'] else 'noCombat'

    # cortical parcellation
    parc = cfg['data']['parcellation']

    # PCA to reduce explanation matrix down a bit for model estimation
    n_pca_comps = 100
    pca = PCA(n_components=n_pca_comps)
    ss = StandardScaler(with_std=True)

    for model in ['linear', 'nonlinear', 'ensemble']:

        #####################################################################################################
        # LOADING
        # load previously calculated model predictions and explanations
        #####################################################################################################
        # load data
        # load npy file with fold-wise features to respect CV structure
        model_explanations = pd.read_csv('{:}{:}-model-feature-explanations-{:}-{:}-{:}-{:}.csv'.format(genpath, model, run_combat, regress, run_pca, parc))
        model_predictions = pd.read_csv('{:}model_predictions-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc))

        # metadata - get this elsewhere
        subject_info = model_explanations.loc[:, model_explanations.columns.isin(['ID', 'Age', 'Male', 'Site', 'fold'])]
        # feature importances - put this within fold
        explanations =  model_explanations.loc[:, ~model_explanations.columns.isin(['ID', 'Age', 'Male', 'Site', 'fold'])]
        # brain age deltas - these should be fold-wise train predictions and test predictions
        deltas = model_predictions[model+'_uncorr_preds'] - model_predictions.Age
        # folds
        cv_folds = subject_info.fold.values

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
        # K-FOLD CV
        # calculate deconfounded explanations and deltas for PLS within CV folds
        #########################################################################################################
        # partial out confounds
        explanations_deconf = np.zeros_like(explanations)
        delta_deconf = np.zeros_like(deltas)

        for f in np.arange(5):
            fold = f+1
            # split data
            # explanations = explained using model fit on train data
            train_data, test_data = explanations[cv_folds!=fold], explanations[cv_folds==fold]

            # split confounds
            train_confounds, test_confounds = confounds[cv_folds!=fold], confounds[cv_folds==fold]

            # split deltas - based on predictions using model trained on train fold - not CV'd estimates
            train_deltas, test_deltas = deltas[cv_folds!=fold], deltas[cv_folds==fold]

            # remove variance due to confounds from explanations (PCA to improve conditioning)
            train_data_deconf, test_data_deconf = deconfound(pca.fit_transform(train_data), train_confounds, test_data=pca.transform(test_data), test_confounds=test_confounds)
            explanations_deconf[cv_folds==fold,:] = pca.inverse_transform(test_data_deconf)

            # remove variance in delta due to confounds
            _, test_delta_deconf = deconfound(train_deltas, train_confounds, test_data = test_deltas, test_confounds=test_confounds)
            delta_deconf[cv_folds==fold] = test_delta_deconf

        # save out - deconfounded explanations
        pd.concat((subject_info, pd.DataFrame(explanations_deconf)), axis=1).to_csv('{:}{:}-model-deconfounded-feature-explanations-{:}-{:}-{:}-{:}.csv'.format(genpath, model, run_combat, regress, run_pca, parc), index=None)

        ##########################################################################################################################
        # PLS
        ##########################################################################################################################
        #####################################################################################################
        # MODEL TESTING I. - Predicting deconfounded brain age delta from deconfounded model explanations...
        # calculate training curve (within 5-fold CV) for different numbers of PLS components
        #####################################################################################################
        fold_train_accuracy = np.zeros((10,5))
        fold_test_accuracy = np.zeros((10,5))
        component_choice = np.arange(10)+1

        print('')
        print('performing cross-validation for model: {:}'.format(model))
        print('number of PLS components from: {:}'.format(component_choice))

        for f in np.arange(5):
            fold = f+1
            # get model explanations for each fold  -  as above make sure these are based on folds properly
            train_data = explanations_deconf[cv_folds!=fold]
            test_data = explanations_deconf[cv_folds==fold]

            # scale
            train_X, test_X = ss.fit_transform(train_data), ss.transform(test_data)

            # use deconfounded brain age delta as target
            train_Y, test_Y = delta_deconf[cv_folds!=fold], delta_deconf[cv_folds==fold]

            # fit training curve
            train_acc, test_acc = plsr_training_curve(train_X, train_Y, test_X, test_Y, components=component_choice)
            fold_train_accuracy[:,f] = train_acc
            fold_test_accuracy[:,f] = test_acc


        #####################################################################################################
        # MODEL TESTING II.- Predicting brain age delta from model explanations...
        # within 5-fold CV, calculate spatial maps for each component
        #####################################################################################################
        plsr_comps = 1

        # outputs
        predicted_delta = np.zeros((len(delta_deconf)))
        feature_loadings = np.zeros((np.shape(explanations_deconf)[1], plsr_comps, 5))  # num_features, num_comps, num_folds
        explained_delta_var = np.zeros((plsr_comps, 5))
        explained_image_var = np.zeros((plsr_comps, 5))
        subject_scores = np.zeros((np.shape(explanations_deconf)[0], plsr_comps))

        print('')
        print('for n components = {:}'.format(plsr_comps))
        print('cross-validating PLS model')
        for f in np.arange(5):
            train_data = explanations_deconf[cv_folds!=f+1]
            test_data = explanations_deconf[cv_folds==f+1]

            train_X, test_X = ss.fit_transform(train_data), ss.transform(test_data)
            train_Y, test_Y = delta_deconf[cv_folds!=f+1], delta_deconf[cv_folds==f+1]

            regional_loadings, component_scores, component_loadings, coefs, weights = run_plsr(train_X, train_Y,  n_comps=plsr_comps)

            # collect
            predicted_delta[cv_folds==f+1] = test_X.dot(coefs.reshape(-1))
            feature_loadings[:,:,f] = regional_loadings
            explained_delta_var[:,f] = component_loadings**2
            subject_scores[cv_folds==f+1,:] = test_X.dot(weights)

            # add norm back in to calculate correctly
            regional_loadings = np.multiply(regional_loadings, np.linalg.norm(train_X, axis=0).reshape(np.shape(train_X)[1],1))
            for c in np.arange(plsr_comps):
                explained_image_var[c,f] = r2_score(train_X, component_scores[:,[c]].dot(regional_loadings[:,[c]].T))

        ## save out - PLSR results by components
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
        delta_predictions = pd.DataFrame((cv_folds, deltas, delta_deconf, predicted_delta)).T
        delta_predictions.columns = ['fold','delta', 'deconfounded_delta','predicted_delta']
        print('see: {:}{:}-delta-predictions-PLS-CV-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc))
        delta_predictions.to_csv('{:}{:}-delta-predictions-PLS-CV-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=False)
        # subject scores
        subject_scores = pd.DataFrame(np.hstack((subject_info.ID[:,np.newaxis], cv_folds[:,np.newaxis], deltas[:,np.newaxis], delta_deconf[:,np.newaxis], subject_scores)))
        subject_scores.columns = ['id','fold','deltas', 'deconfounded_delta','PLS1']
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

if __name__ == '__main__':
    main()

## import numpy as np
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from functions.pls_models import run_plsr, plsr_training_curve

import yaml

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

    # set up models
    ss = StandardScaler()

    for model in ['linear', 'nonlinear', 'ensemble']:
        #####################################################################################################
        # LOADING
        # load previously calculated model predictions and explanations
        #####################################################################################################

        # load data
        model_explanations = pd.read_csv('{:}{:}-model-feature-explanations-{:}-{:}-{:}-{:}.csv'.format(genpath, model, run_combat, regress, run_pca, parc))
        model_predictions = pd.read_csv('{:}model_predictions-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc))

        # metadata
        subject_info = model_explanations.loc[:, model_explanations.columns.isin(['ID', 'Age', 'Male', 'Site', 'fold'])]
        # feature importances
        explanations =  model_explanations.loc[:, ~model_explanations.columns.isin(['ID', 'Age', 'Male', 'Site', 'fold'])]
        # brain age deltas - corrected or uncorrected
        #deltas = model_predictions[model+'_uncorr_preds'] - model_predictions.Age
        deltas = model_predictions[model+'_preds'] - model_predictions.Age

        # folds
        cv_folds = subject_info.fold.values

        #####################################################################################################
        # MODEL TESTING I. - Predicting brain age delta from model explanations...
        # calculate training curve (within 5-fold CV) for different numbers of PLS components
        # use same folds as in brain age models
        #####################################################################################################
        fold_train_accuracy = np.zeros((10,5))
        fold_test_accuracy = np.zeros((10,5))
        component_choice = np.arange(10)+1

        print('')
        print('performing cross-validation for model: {:}'.format(model))
        print('number of PLS components from: {:}'.format(component_choice))
        for f in np.arange(5):
            # get model explanations for each fold
            train_data = explanations[cv_folds!=f+1]
            test_data = explanations[cv_folds==f+1]

            # scale
            train_X, test_X = ss.fit_transform(train_data), ss.transform(test_data)

            # use brain age delta as target
            train_Y, test_Y = deltas[cv_folds!=f+1].values, deltas[cv_folds==f+1].values

            # fit training curve
            train_acc, test_acc = plsr_training_curve(train_X, train_Y, test_X, test_Y, components=component_choice)
            fold_train_accuracy[:,f] = train_acc
            fold_test_accuracy[:,f] = test_acc

        # choose number of components
        # plsr_comps = np.where(np.diff(np.insert(np.mean(fold_test_accuracy, axis=1), 0, 0))<.05)[0][0] #adding another component increses R2 by less than 5%
        plsr_comps = 3

        #####################################################################################################
        # MODEL TESTING II.- Predicting brain age delta from model explanations...
        # within 5-fold CV, calculate spatial maps for each component
        #####################################################################################################

        # outputs
        predicted_delta = np.zeros((len(deltas)))
        feature_loadings = np.zeros((np.shape(explanations)[1], plsr_comps, 5))  # num_features, num_comps, num_folds
        explained_delta_var = np.zeros((plsr_comps, 5))
        explained_image_var = np.zeros((plsr_comps, 5))
        subject_scores = np.zeros((np.shape(explanations)[0], plsr_comps))

        print('')
        print('for n components = {:}'.format(plsr_comps))
        print('cross-validating PLS model')
        for f in np.arange(5):
            train_data = explanations[cv_folds!=f+1]
            test_data = explanations[cv_folds==f+1]

            train_X, test_X = ss.fit_transform(train_data), ss.transform(test_data)
            train_Y, test_Y = deltas[cv_folds!=f+1].values, deltas[cv_folds==f+1].values

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
        delta_predictions = pd.DataFrame((cv_folds, deltas, predicted_delta)).T
        delta_predictions.columns = ['fold','delta','predicted_delta']
        print('see: {:}{:}-delta-predictions-PLS-CV-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc))
        delta_predictions.to_csv('{:}{:}-delta-predictions-PLS-CV-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=False)
        # subject scores
        subject_scores = pd.DataFrame(np.hstack((subject_info.ID[:,np.newaxis], cv_folds[:,np.newaxis], deltas[:,np.newaxis], subject_scores)))
        subject_scores.columns = ['id','fold','delta','PLS1','PLS2','PLS3']
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

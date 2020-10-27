import pandas as pd
import numpy as np

import yaml, os

from functions.misc import deconfound

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

    for model in ['linear', 'nonlinear', 'ensemble']:
        print('')
        print('###### MODEL: {:} ##############'.format(model))
        #####################################################################################################
        # LOADING
        # load model explanations and predictions for each fold
        #####################################################################################################
        # load data
        # n_sub x n_feature x n_fold array: for each fold, explanations calculated using model trained on training folds
        model_explanations = np.load('{:}{:}-model-all-fold-feature-explanations-{:}-{:}-{:}-{:}.npy'.format(genpath, model, run_combat, regress, run_pca, parc))
        # n_sub x n_folds: for each fold, deltas (uncorrected) predictions using model trained on training folds
        model_predictions = np.load('{:}{:}-model-all-fold-delta-{:}-{:}-{:}-{:}.npy'.format(genpath, model, run_combat, regress, run_pca, parc))

        # folds
        cv_preds = pd.read_csv('{:}model_predictions-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc))
        subject_info = cv_preds[['ID','Age','Male','Site','fold']]
        cv_folds = subject_info.fold.values

        # metric data
        metric_names = ['thickness', 'area']
        metric_data = []
        for metric in metric_names:
            chkfile = genpath + metric + '-parcellated-data-' + parc + '.csv'
            metric_data.append(np.loadtxt(chkfile, delimiter=','))

        # confounds
        confounds = pd.DataFrame()
        confounds['age'] = cv_preds.Age
        confounds['sex'] = cv_preds.Male
        confounds['meanthick'] = np.mean(metric_data[0],1)
        confounds['meanarea'] = np.mean(metric_data[1],1)


        #########################################################################################################
        # DECONFOUND
        # calculate deconfounded explanations and deltas for PLS within CV folds
        #########################################################################################################
        # partial out confounds
        explanations_deconf, cv_explanations_deconf, delta_deconf, cv_delta_deconf = deconfound_train_test_data(model_explanations, model_predictions, confounds, cv_folds)

        #########################################################################################################
        # SAVE OUT
        #########################################################################################################
        # deconfounded data arrays
        print('deconfounded model explanations (all folds): {:}{:}-model-all-fold-deconfounded-feature-explanations-{:}-{:}-{:}-{:}.npy'.format(genpath, model, run_combat, regress, run_pca, parc))
        np.save('{:}{:}-model-all-fold-deconfounded-feature-explanations-{:}-{:}-{:}-{:}.npy'.format(genpath, model, run_combat, regress, run_pca, parc), explanations_deconf)
        print('deconfounded model deltas (all folds): {:}{:}-model-all-fold-deconfounded-delta-{:}-{:}-{:}-{:}.npy'.format(genpath, model, run_combat, regress, run_pca, parc))
        np.save('{:}{:}-model-all-fold-deconfounded-delta-{:}-{:}-{:}-{:}.npy'.format(genpath, model, run_combat, regress, run_pca, parc), delta_deconf)

        print('')
        print('deconfounded model explanations - cross-validated: {:}{:}-model-deconfounded-feature-explanations-{:}-{:}-{:}-{:}.csv'.format(genpath, model, run_combat, regress, run_pca, parc))
        cv_deconf_out = pd.DataFrame(cv_explanations_deconf)
        cv_deconf_out = pd.concat((subject_info, cv_deconf_out), axis=1)
        cv_deconf_out.to_csv('{:}{:}-model-deconfounded-feature-explanations-{:}-{:}-{:}-{:}.csv'.format(genpath, model, run_combat, regress, run_pca, parc), index=None)

        print('deconfounded deltas - cross-validated: {:}{:}-model-deconfounded-delta-{:}-{:}-{:}-{:}.csv'.format(genpath, model, run_combat, regress, run_pca, parc))
        cv_delta_out = pd.DataFrame(cv_delta_deconf[:,np.newaxis])
        cv_delta_out = pd.concat((subject_info, cv_delta_out), axis=1)
        cv_delta_out.to_csv('{:}{:}-model-deconfounded-delta-{:}-{:}-{:}-{:}.csv'.format(genpath, model, run_combat, regress, run_pca, parc), index=None)
        print('')


def deconfound_train_test_data(explanations, deltas, confounds, fold_ids):
    """
    split data according to prespecified folds and deconfound explanation data and deltas within folds

    parameters
    ----------
    explanations: n_sub x n_feature x n_fold array, previously calculated model explanation for each fold
    deltas: n_sub x n_fold array, previously calculated age deltas for each fold
    confounds: n_sub x n_var, dataframe of confound variables for each subject
    fold_ids: n_sub array, fold label for each subject

    returns
    ------
    explanations_deconf: n_sub x n_feature x n_fold array, deconfounded explanation data (model calculated within training data and applied to test data in each fold)
    cv_explanations_deconf: n_sub x n_feature, deconfounded data for each subject in test set of each fold
    delta_deconf: n_sub x n_fold, deconfounded delta (from model calculated with training data in each fold)
    cv_delta_deconf: n_sub, deconfounded delta for each subject in test set of each fold

    """
    print('')
    print('deconfounding')

    explanations_deconf = np.zeros_like(explanations)
    cv_explanations_deconf = np.zeros_like(explanations[:,:,0])
    delta_deconf = np.zeros_like(deltas)
    cv_delta_deconf = np.zeros_like(deltas[:,0])

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
        delta_deconf[fold_ids==fold, f] = cv_delta_deconf[fold_ids==fold] = test_delta_deconf

    return explanations_deconf, cv_explanations_deconf, delta_deconf, cv_delta_deconf

if __name__ == '__main__':
    main()

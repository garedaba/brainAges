import pandas as pd
import numpy as np


from tqdm import tqdm

from sklearn.metrics import pairwise_distances

import yaml, os

def main():
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


    # load data #################################################################################3#############
    for model in ['linear', 'nonlinear', 'ensemble']:
        print('*********** {:} model *******************'.format(model))
        # load deconfounded explanations and predictions etc
        model_explanations = pd.read_csv('{:}{:}-model-deconfounded-feature-explanations-{:}-{:}-{:}-{:}.csv'.format(genpath, model, run_combat, regress, run_pca, parc))
        explanations = model_explanations.loc[:,~model_explanations.columns.isin(['ID', 'Age', 'Male', 'Site', 'fold'])]
        model_predictions = pd.read_csv('{:}model_predictions-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc))
        all_fold_explanations = np.load('{:}{:}-model-all-fold-deconfounded-feature-explanations-{:}-{:}-{:}-{:}.npy'.format(genpath, model, run_combat, regress, run_pca, parc))

        # load surrogates if calculated already
        chkfile = 'surrogates/{:}-model-fold-500surrogates-{:}-{:}-{:}-{:}.npy'.format(model, run_combat, regress, run_pca, parc)
        surrogates_exist = False
        if os.path.exists(chkfile):
            surrogates = np.load(chkfile)
            surrogates_exist = True
        else:
            print('no surrogates yet...try run_surrogates.py for this model')

        mean_within = []
        mean_between = []
        mean_surrogate = []

        mean_within_abs = []
        mean_between_abs = []
        mean_surrogate_abs = []

        for sub in tqdm(np.arange(768), desc='subjects'):
            # which fold was subject in?
            subject_test_fold = model_predictions.fold[sub]-1
            # get folds where subject was in training set for comparison
            training_folds = np.where([f for f in np.arange(5)!=subject_test_fold])[0]

            # explanations when subject was in test data
            sub_explanation = explanations.loc[sub,:]
            # explanations when subject was in train data
            sub_train_explanation = all_fold_explanations[sub,:,training_folds]

            # cosine similarity between feature importance when in test vs when in train
            sub_test_train = 1-pairwise_distances(sub_explanation[np.newaxis,:], sub_train_explanation, metric='cosine')
            mean_within.append(np.mean(sub_test_train))

            # using absolute values to account for potential anti-correlation in those with +ve and -ve brain age
            sub_test_train_abs = 1-pairwise_distances(abs(sub_explanation[np.newaxis,:]), abs(sub_train_explanation), metric='cosine')
            mean_within_abs.append(np.mean(sub_test_train))

            # explanations for all other subjects when in test folds
            notsub = np.ones(768, dtype=bool)
            notsub[sub]=False
            not_sub_explanation = explanations[notsub]

            # cosine similarity between subject feature importance and all other subjects
            sub_test_notsub = 1-pairwise_distances(sub_explanation[np.newaxis,:], not_sub_explanation, metric='cosine')
            mean_between.append(np.mean(sub_test_notsub))
            sub_test_notsub_abs = 1-pairwise_distances(abs(sub_explanation[np.newaxis,:]), abs(not_sub_explanation), metric='cosine')
            mean_between_abs.append(np.mean(sub_test_notsub_abs))

            # explanations for surrogates
            if surrogates_exist:
                surrogate_explanations = surrogates[notsub,:,:]
                tmp_mean = []
                tmp_mean_abs = []

                for sur in np.arange(surrogate_explanations.shape[2]):
                    tmp_mean.append(np.mean(1-pairwise_distances(sub_explanation[np.newaxis,:], surrogate_explanations[:,:,sur], metric='cosine')))
                    tmp_mean_abs.append(np.mean(1-pairwise_distances(abs(sub_explanation[np.newaxis,:]), abs(surrogate_explanations[:,:,sur]), metric='cosine')))

                mean_surrogate.append(np.mean(tmp_mean))
                mean_surrogate_abs.append(np.mean(tmp_mean_abs))

        # make dataframe
        all_correlations = pd.DataFrame((mean_within, mean_within_abs, mean_between, mean_between_abs, mean_surrogate, mean_surrogate_abs)).T
        indices = list(zip(['within', 'within','between','between','surrogate', 'surrogate'],['raw','abs','raw','abs','raw','abs']))
        all_correlations.columns = pd.MultiIndex.from_tuples(indices, names=['group','values'])
        print('see: {:}{:}-explanation-cosine-similarities-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc))
        all_correlations.melt(value_name='cosine_sim').to_csv('{:}{:}-explanation-cosine-similarities-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=None)

if __name__ == '__main__':
    main()

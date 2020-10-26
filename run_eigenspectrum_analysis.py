import numpy as np
import pandas as pd

import yaml, os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    surrogates = outpath + '../surrogates/'
    parcdir = outpath + '../parcellations/'

    # other params - whether to regress out global metrics and run PCA
    preprocessing_params = cfg['preproc']
    regress = 'Corrected' if preprocessing_params['regress'] else 'Raw'
    run_pca = 'PCA' if preprocessing_params['pca'] else 'noPCA'
    run_combat = 'Combat' if preprocessing_params['combat'] else 'noCombat'

    # cortical parcellation
    parc = cfg['data']['parcellation']

    # nonlinear only
    model = 'linear'

    # PCA
    n_comps = 10
    ss = StandardScaler(with_std=False)
    pca = PCA(n_components = n_comps)

    # surrogates
    n_surrogates = 100

    #####################################################################################################
    # SURROGATE ANALYSIS
    # compare variance structure of data using surrogate maps with and without
    # spatial autocorrelation
    #####################################################################################################
    # load data, decnfounded using out-of-sample training samples
    cv_deconfounded_data  = pd.read_csv('{:}{:}-model-deconfounded-feature-explanations-{:}-{:}-{:}-{:}.csv'.format(genpath, model, run_combat, regress, run_pca, parc))
    cv_folds = cv_deconfounded_data.fold.values
    cv_deconfounded_data = cv_deconfounded_data.drop(['ID', 'Age', 'Male', 'Site', 'fold'], axis='columns')

    # remove dubjectwise mean
    cv_deconfounded_data = cv_deconfounded_data - np.mean(cv_deconfounded_data, axis=1)[:,np.newaxis]

    ##############################################################################################################
    # load surrogates and run PCA
    ##############################################################################################################
    cv_surrogates = np.zeros((np.shape(cv_deconfounded_data)[0],np.shape(cv_deconfounded_data)[1], n_surrogates))
    # load surrogate maps constructed using test samples above
    for f in np.arange(5):
        fold_surrogates = np.load('{:}{:}-model-fold-{:}-{:}surrogates-{:}-{:}-{:}-{:}.npy'.format(surrogates, model, f+1, n_surrogates, run_combat, regress, run_pca, parc))
        cv_surrogates[cv_folds==f+1,:,:] = fold_surrogates[cv_folds==f+1,:,:]

    # fit PCA to explanations
    pca.fit(ss.fit_transform(cv_deconfounded_data))
    # save out
    print('')
    print('PCA on explanation data')
    print('see: {:}{:}-explanations-explainedVariance-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc))
    print('see: {:}{:}-explanations-principal-components-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc))
    print('')
    pd.DataFrame(pca.explained_variance_ratio_, columns=['explained_variance']).to_csv('{:}{:}-explanations-explainedVariance-{:}-{:}-{:}-{:}.csv'.format(outpath,  model, run_combat, regress, run_pca, parc))
    pd.DataFrame(pca.components_).to_csv('{:}{:}-explanations-principal-components-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc))

    # fit PCA to surrogates
    surrogate_explained_variance = run_pcas(cv_surrogates, n_comps=n_comps)
    # save out
    print('')
    print('PCA on surrogate data')
    print('see:{:}{:}-explanations-explainedVariance-surrogates-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc))
    pd.DataFrame(surrogate_explained_variance, columns=np.arange(n_comps)+1).to_csv('{:}{:}-explanations-explainedVariance-surrogates-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=None)

    # surrogates with randomly shuffled data
    random_maps = []
    for r in np.arange(n_surrogates):
        random_maps.append(np.array([x[np.random.rand(cv_deconfounded_data.shape[1]).argsort()] for x in cv_deconfounded_data.values]))
    random_maps = np.stack(random_maps, axis=2)
    # variance explained by principal components
    random_explained_variance = run_pcas(random_maps, n_comps=n_comps)
    # save out
    print('')
    print('PCA on random data')
    print('see:{:}{:}-explanations-explainedVariance-randoms-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc))
    pd.DataFrame(random_explained_variance, columns=np.arange(n_comps)+1).to_csv('{:}{:}-explanations-explainedVariance-randoms-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=None)


    # save some example maps for visualisation later
    print('see: {:}residuals-example-surrogates-{:}-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc))
    print('see: {:}residuals-example-randoms-{:}-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc))

    np.savetxt('{:}residuals-example-surrogates-{:}-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), cv_surrogates[0], delimiter=',')
    np.savetxt('{:}residuals-example-randoms-{:}-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), random_maps[0], delimiter=',')


def run_pcas(maps, n_comps=10):
    ss = StandardScaler(with_std=False)
    pca = PCA(n_components = n_comps)

    variance_explained = []
    for k in np.arange(np.shape(maps)[2]):
        # remove dubjectwise mean
        dm_map = maps[:,:,k] - np.mean(maps[:,:,k], axis=1)[:,np.newaxis]
        #pca.fit(ss.fit_transform(maps[:,:,k]))
        pca.fit(ss.fit_transform(dm_map))
        variance_explained.append(pca.explained_variance_ratio_)

    variance_explained = np.stack(variance_explained)

    return variance_explained

if __name__ == '__main__':
    main()

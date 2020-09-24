import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import yaml, os

from brainsmash.workbench.geo import cortex
from brainsmash.workbench.geo import parcellate

from functions.pls_models import run_plsr
from functions.misc import create_surrogates, bootstrap_surrogates

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
    model = 'nonlinear'

    # PLS components
    plsr_comps = 3
    ss = StandardScaler()

    # PCA
    n_comps = 10
    pca = PCA(n_components = n_comps)

    # surrogates
    repeats = 100

    #####################################################################################################
    # SURROGATE ANALYSIS
    # compare variance structure of data before and after PLS using surrogate maps with and without
    # spatial autocorrelation
    #####################################################################################################
    # surrogates created for separate hemispheres
    for hemi in ['lh', 'rh']:
        print('')
        print(hemi)
        print('setting up brainsmash...')
        print('')
        ##############################################################################################################
        # Create distance matrices for surrogate maps
        ##############################################################################################################
        # fsaverage5 pial surface
        surface = '{:}{:}.pial.gii'.format(surrogates, hemi)

        # first calculate distance matrices if needed
        chkfile = '{:}{:}_denseGeodesicDistMat.txt'.format(surrogates, hemi)
        if os.path.exists(chkfile):
            print('dense distance matrix already calculated')
        else:
            cortex(surface=surface, outfile='{:}{:}_denseGeodesicDistMat.txt'.format(surrogates, hemi), euclid=False)

        # convert parcelation into gifti
        if parc=='HCP':
            parcfile = '{:}{:}.HCP-MMP1-fsaverage5.annot'.format(parcdir, hemi)
        else:
            parcfile = '{:}{:}.custom500.annot'.format(parcdir, hemi)

        if os.path.exists('{:}{:}.{:}.annot.gii'.format(surrogates, hemi, parc)):
            print('annotation file already exists')
        else:
            command = '/usr/local/freesurfer/bin/mris_convert --annot {:} {:}{:}.pial.gii {:}{:}.{:}.annot.gii'.format(parcfile, surrogates, hemi, surrogates, hemi, parc)
            os.system(command)

        # calculate parcellated distance matrix
        if os.path.exists('{:}{:}_{:}_parcelGeodesicDistMat.txt'.format(surrogates, hemi, parc)):
            print('parcellated distance matrix already exists')
        else:
            infile = '{:}{:}_denseGeodesicDistMat.txt'.format(surrogates, hemi)
            outfile = '{:}{:}_{:}_parcelGeodesicDistMat.txt'.format(surrogates, hemi, parc)
            dlabel = '{:}{:}.{:}.annot.gii'.format(surrogates, hemi, parc)
            parcellate(infile, dlabel, outfile)

    ##############################################################################################################
    # Load data
    ##############################################################################################################
    # load brainage explanations
    model_explanations = pd.read_csv('{:}{:}-model-feature-explanations-{:}-{:}-{:}-{:}.csv'.format(genpath, model, run_combat, regress, run_pca, parc))
    explanations =  model_explanations.loc[:, ~model_explanations.columns.isin(['ID', 'Age', 'Male', 'Site', 'fold'])]

    # load brain age deltas
    model_predictions = pd.read_csv('{:}model_predictions-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc))
    deltas = model_predictions[model+'_preds'] - model_predictions.Age

    ##############################################################################################################
    # run PLS
    ##############################################################################################################
    # PLS
    print('')
    print('performing PLS with {:} components'.format(plsr_comps))
    print('')
    train_X = ss.fit_transform(explanations)
    train_Y = deltas.values
    regional_loadings, component_scores, component_loadings, coefs, weights = run_plsr(train_X, train_Y,  n_comps=plsr_comps)

    # calculate residuals - everything left in explanations, not associated with brain age delta
    regional_loadings = np.multiply(regional_loadings, np.linalg.norm(train_X, axis=0).reshape(np.shape(train_X)[1],1))
    residuals = ss.inverse_transform(train_X - component_scores.dot(regional_loadings.T))

    ##############################################################################################################
    # create surrogates and run PCA
    ##############################################################################################################
    num = round(np.shape(explanations)[1]/4)

    explanation_maps = []
    residual_maps = []
    surrogate_explanation_maps = []
    surrogate_residuals_maps = []
    random_explanation_maps = []
    random_residuals_maps = []

    # separate out metrics and hemispheres (need to be calculated separately)
    for n_metric, (metric,pal) in enumerate(zip(['thickness', 'area'], ['Reds', 'Greens'])):

        for hemi in ['lh','rh']:
            print('***********************************')
            print('{:} maps: {:}'.format(metric, hemi))
            print('***********************************')

            if hemi=='lh':
                start = (n_metric+1)*(n_metric*num)
                end = (n_metric+1)*(n_metric*num)+num
            else:
                start = num+(n_metric+1)*(n_metric*num)
                end = 2*num+(n_metric+1)*(n_metric*num)

            # explanations
            ######################################################
            exp_map = explanations.iloc[:,start:end].to_numpy()

            # fit PCA to explanations
            pca.fit(ss.fit_transform(exp_map))
            # save out
            print('')
            print('PCA on explanation data')
            print('see: {:}{:}-{:}-explanations-explainedVariance-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))
            print('see: {:}{:}-{:}-explanations-principal-components-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))
            print('')
            pd.DataFrame(pca.explained_variance_ratio_, columns=['explained_variance']).to_csv('{:}{:}-{:}-explanations-explainedVariance-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))
            pd.DataFrame(pca.components_).to_csv('{:}{:}-{:}-explanations-principal-components-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))

            # surrogates with matched spatial autocorrelation
            surro_exp_maps = bootstrap_surrogates(exp_map,
                                 '{:}{:}_{:}_parcelGeodesicDistMat.txt'.format(surrogates, hemi, parc),
                                 n_boot=repeats, n_subjects=100, n_samples=10)
            # variance explained by principal components in each random sample
            surro_exp_pcs = run_pcas(surro_exp_maps, n_comps=n_comps)
            # save out
            print('')
            print('PCA on surrogate data')
            print('see:{:}{:}-{:}-explanations-explainedVariance-surrogates-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))
            pd.DataFrame(surro_exp_pcs, columns=np.arange(n_comps)+1).to_csv('{:}{:}-{:}-explanations-explainedVariance-surrogates-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc), index=None)


            # surrogates with randomly shuffled data
            random_exp_maps = []
            for r in np.arange(repeats):
                random_exp_maps.append(np.array([x[np.random.rand(exp_map.shape[1]).argsort()] for x in exp_map]))
            # variance explained by principal components
            random_exp_pcs = run_pcas(random_exp_maps, n_comps=n_comps)
            # save out
            print('')
            print('PCA on random data')
            print('see:{:}{:}-{:}-explanations-explainedVariance-randoms-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))
            pd.DataFrame(random_exp_pcs, columns=np.arange(n_comps)+1).to_csv('{:}{:}-{:}-explanations-explainedVariance-randoms-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc), index=None)


            # residuals
            ######################################################
            res_map = residuals[:, start:end]

            # fit PCA to residuals
            pca.fit(ss.fit_transform(res_map))
            # save out
            print('')
            print('PCA on residuals data')
            print('see: {:}{:}-{:}-residuals-explainedVariance-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))
            print('see: {:}{:}-{:}-residuals-principal-components-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))
            print('')
            pd.DataFrame(pca.explained_variance_ratio_, columns=['explained_variance']).to_csv('{:}{:}-{:}-residuals-explainedVariance-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))
            pd.DataFrame(pca.components_).to_csv('{:}{:}-{:}-residuals-principal-components-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))


            # surrogates with matched spatial autocorrelation
            surro_res_maps = bootstrap_surrogates(res_map,
                                 '{:}{:}_{:}_parcelGeodesicDistMat.txt'.format(surrogates, hemi, parc),
                                 n_boot=repeats, n_subjects=100, n_samples=10)
            # variance explained by principal components in each random sample
            surro_res_pcs = run_pcas(surro_res_maps, n_comps=n_comps)
            # save out
            print('')
            print('PCA on surrogate data')
            print('see:{:}{:}-{:}-residuals-explainedVariance-surrogates-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))
            pd.DataFrame(surro_res_pcs, columns=np.arange(n_comps)+1).to_csv('{:}{:}-{:}-residuals-explainedVariance-surrogates-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc), index=None)


            # surrogates with randomly shuffled data
            random_res_maps = []
            for r in np.arange(repeats):
                random_res_maps.append(np.array([x[np.random.rand(res_map.shape[1]).argsort()] for x in res_map]))
            # variance explained by principal components
            random_res_pcs = run_pcas(random_res_maps, n_comps=n_comps)
            # save out
            print('')
            print('PCA on random data')
            print('see:{:}{:}-{:}-residuals-explainedVariance-randoms-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))
            pd.DataFrame(random_res_pcs, columns=np.arange(n_comps)+1).to_csv('{:}{:}-{:}-residuals-explainedVariance-randoms-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc), index=None)


            # save some example maps for visualisation later
            print('see: {:}{:}-{:}-explanations-example-surrogates-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))
            print('see: {:}{:}-{:}-explanations-example-randoms-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))
            print('see: {:}{:}-{:}-residuals-example-surrogates-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))
            print('see: {:}{:}-{:}-residuals-example-randoms-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc))

            np.savetxt('{:}{:}-{:}-explanations-example-surrogates-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc), surro_exp_maps[0], delimiter=',')
            np.savetxt('{:}{:}-{:}-explanations-example-randoms-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc), random_exp_maps[0], delimiter=',')
            np.savetxt('{:}{:}-{:}-residuals-example-surrogates-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc), surro_res_maps[0], delimiter=',')
            np.savetxt('{:}{:}-{:}-residuals-example-randoms-{:}-{:}-{:}-{:}-{:}.csv'.format(surrogates, metric, hemi, model, run_combat, regress, run_pca, parc), random_res_maps[0], delimiter=',')


def run_pcas(maps, n_comps=10):
    ss = StandardScaler()
    pca = PCA(n_components = n_comps)

    variance_explained = []
    for k in maps:
        pca.fit(ss.fit_transform(k))
        variance_explained.append(pca.explained_variance_ratio_)

    variance_explained = np.stack(variance_explained)

    return variance_explained


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import yaml, os

from functions.misc import create_surrogates

from brainsmash.workbench.geo import cortex
from brainsmash.workbench.geo import parcellate

from tqdm import tqdm

def main(model='nonlinear', n_surrogates = 500):

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
            parcfile = '{:}{:}.HCP-MMP1-fsaverage5-noHipp.annot'.format(parcdir, hemi)
        else:
            parcfile = '{:}{:}.custom500-fsaverage5.annot'.format(parcdir, hemi)

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
    # Load deconfounded data, create surrogates
    ##############################################################################################################
    # load (deconfounded) brainage explanations
    deconfounded_model_explanations = pd.read_csv('{:}{:}-model-deconfounded-feature-explanations-{:}-{:}-{:}-{:}.csv'.format(genpath, model, run_combat, regress, run_pca, parc))
    deconfounded_model_explanations = deconfounded_model_explanations.drop(['ID', 'Age', 'Male', 'Site', 'fold'], axis='columns').values

    n_sub, n_feat = np.shape(deconfounded_model_explanations)

    if os.path.exists('{:}{:}-model-{:}surrogates-{:}-{:}-{:}-{:}.npy'.format(surrogates, model, n_surrogates, run_combat, regress, run_pca, parc )):
        print('surrogate maps already calculated')
    else:
        # create surrogate maps
        fold_surrogates = get_all_surrogates(deconfounded_model_explanations, n_surrogates, surrogates, parc)
        np.save('{:}{:}-model-fold-{:}surrogates-{:}-{:}-{:}-{:}.npy'.format(surrogates, model, n_surrogates, run_combat, regress, run_pca, parc), fold_surrogates)


def get_all_surrogates(data, n_surrogates, path_to_surrogates, parc):
    """wrapper to create surrogate maps"""
    n_sub, n_feat = np.shape(data)
    fold_surrogates = np.zeros((n_sub, n_feat, n_surrogates))
    num = round(n_feat/4)

    # separate out thickness and area maps
    for n_metric, metric in enumerate(['thickness', 'area']):
        # create surrogates separately for lH and rh
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

            # deconfounded explanations
            exp_map = data[:, start:end]

            # get surrogates with matched SA
            surrogate_exp_maps = create_surrogates(exp_map,
                                         '{:}{:}_{:}_parcelGeodesicDistMat.txt'.format(path_to_surrogates, hemi, parc),
                                         n_repeats=n_surrogates, n_jobs=10)

            for n,i in enumerate(surrogate_exp_maps):
                fold_surrogates[:,start:end,n] = i

    return fold_surrogates


if __name__ == '__main__':
    main()

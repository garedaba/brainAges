import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os, glob
import yaml

from functions.surfaces import load_surf_data, parcellateSurface
from functions.models import get_ensemble_model, get_linear_model, get_nonlinear_model, get_model_explanations
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
    fold = np.zeros((n_subs, 1))
    feature_explanations = np.zeros((3, n_subs, n_features*2))
    if run_pca=='PCA':
        pca_explanations = np.zeros((3, n_subs, preprocessing_params['pca_comps']*2))

    # cross-validation
    for n, (train_idx, test_idx) in enumerate(skf.split(np.arange(n_subs), subject_data.Site)):
        print('')
        print('FOLD {:}:------------------------------------------------'.format(n+1))

        # age data for train and test sets
        train_y, test_y = subject_data.Age[train_idx], subject_data.Age[test_idx]

        # run any required processing separetely on metric data (thickness and area)
        train_x, test_x, pca_models = pre_process_metrics(metric_names, metric_data, train_idx, test_idx, preprocessing_params)

        # run models in each fold
        fold[test_idx] = n+1
        print('')
        # get pre-specified models as pipelines with randomised nested CV parameter search
        for m, (model_name, model) in enumerate(zip(['linear', 'nonlinear', 'ensemble'], [get_linear_model(), get_nonlinear_model(), get_ensemble_model()])):
            # FIT
            print('fitting {:} model'.format(model_name))
            model.fit(train_x, train_y)
            # PREDICT
            preds[test_idx, m] = model.predict(test_x)
            # EXPLAIN
            print('calculating {:} model explanations for test data'.format(model_name))
            exp_features = round(n_features/10)
            model_explanations = get_model_explanations(model, train_x, test_x, num_features=exp_features)
            # if PCA has been performed, transform importances back to feature space
            # - sum of individual PC component maps, weighted by contribution in an individual
            if run_pca == 'PCA':
                pca_explanations[m, test_idx, :] = model_explanations
                feature_explanations[m, test_idx, :] = np.concatenate((model_explanations[:,:pca_models[0].n_components_].dot(pca_models[0].components_),
                                                                     model_explanations[:,pca_models[0].n_components_:].dot(pca_models[1].components_)), axis=1)
            else:
                feature_explanations[m, test_idx, :] = model_explanations

    #####################################################################################################
    # RESULTS
    #####################################################################################################
    print('---------------------------------------------------------')
    print('compiling results')
    print('---------------------------------------------------------')
    # collate data
    preds = pd.DataFrame(preds, columns = ['linear_preds', 'nonlinear_preds', 'ensemble_preds'])
    fold = pd.DataFrame(fold.astype(int), columns=['fold'])
    predictions = pd.concat((subject_data, fold, preds), axis=1)

    # saving
    print('model predictions: {:}model_predictions-{:}-{:}-{:}.csv'.format(outpath, regress, run_pca, parc))
    print('')
    predictions.to_csv('{:}model_predictions-{:}-{:}-{:}.csv'.format(outpath, regress, run_pca, parc), index=False)

    # accuracies and AUC
    n_fold = len(np.unique(predictions.fold))
    models = ['linear', 'nonlinear', 'ensemble']

    fold_mae = np.zeros((n_fold, len(models)))
    fold_r2 = np.zeros((n_fold, len(models)))

    for n, f in enumerate(np.unique(predictions.fold)):
        for m, model in enumerate(models):
            fold_mae[n, m] = mean_absolute_error(predictions.Age[predictions.fold==f], predictions[model+'_preds'][predictions.fold==f])
            fold_r2[n, m] = r2_score(predictions.Age[predictions.fold==f], predictions[model+'_preds'][predictions.fold==f])

    fold_mae = pd.DataFrame(fold_mae, columns=models)
    fold_mae.insert(0, 'fold', np.unique(predictions.fold))

    fold_r2 = pd.DataFrame(fold_r2, columns=models)
    fold_r2.insert(0, 'fold', np.unique(predictions.fold))

    # saving
    print('model accuracy (MAE): {:}MAE-{:}-{:}-{:}.csv'.format(outpath, regress, run_pca, parc))
    fold_mae.to_csv('{:}MAE-{:}-{:}-{:}.csv'.format(outpath, regress, run_pca, parc), index=False)
    print('model accuracy (R2): {:}R2-{:}-{:}-{:}.csv'.format(outpath, regress, run_pca, parc))
    fold_r2.to_csv('{:}R2-{:}-{:}-{:}.csv'.format(outpath, regress, run_pca, parc), index=False)
    print('')

    # explainations
    for m, model_name in enumerate(zip(['linear', 'nonlinear', 'ensemble'])):
        exp = pd.DataFrame(feature_explanations[m])
        fold = pd.DataFrame(fold.astype(int), columns=['fold'])
        feat_exp = pd.concat((subject_data, fold, exp), axis=1)
        print('model explanations: {:}{:}-model-feature-explanations-{:}-{:}-{:}.csv'.format(genpath, model_name, regress, run_pca, parc))
        print('')
        feat_exp.to_csv('{:}{:}-model-feature-explanations-{:}-{:}-{:}.csv'.format(genpath, model_name, regress, run_pca, parc), index=False)
        if run_pca=='PCA':
            pcexp = pd.DataFrame(pca_explanations[m])
            fold = pd.DataFrame(fold.astype(int), columns=['fold'])
            pcexp = pd.concat((subject_data, fold, pcexp), axis=1)
            print('model explanations (PCA): {:}{:}-model-pca-explanations-{:}-{:}-{:}.csv'.format(genpath, model_name, regress, run_pca, parc))
            print('')
            pcexp.to_csv('{:}{:}-model-pca-explanations-{:}-{:}-{:}.csv'.format(genpath, model_name, regress, run_pca, parc), index=False)

    #####################################################################################################
    # PLOTTING
    #####################################################################################################
    print('---------------------------------------------------------')
    print('plotting')
    print('---------------------------------------------------------')

    for mapping, pal in zip(['Age', 'Site'], ['inferno', 'tab20b']):
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(18,5), sharex=True, sharey=False)
        for ax, model in zip([ax1,ax2,ax3], ['linear', 'nonlinear', 'ensemble']):
            plot_age_scatters(predictions.Age, predictions[model + '_preds'], predictions[mapping], ax=ax, palette=pal)
            ax.set_title(model, fontsize=16)
        plt.tight_layout()
        print('saving model scatters to: {:}model_predictions-{:}-{:}-by{:}-{:}.png'.format(outpath, regress, run_pca, mapping, parc))
        plt.savefig('{:}model_predictions-{:}-{:}-by{:}-{:}.png'.format(outpath, regress, run_pca, mapping, parc))

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4), sharey=False)
    for dat, val, ax, pal, lims in zip([fold_mae, fold_r2], ['MAE', 'r2'], [ax1, ax2], ['inferno', 'inferno'], [(0,4), (0,1)]):

        plot_df = dat.melt(id_vars='fold', var_name='model', value_name=val)

        sns.boxplot(x='model', y=val, data=plot_df,  palette=pal, ax=ax)

        ax.set_xticklabels(['linear', 'nonlinear', 'ensemble'], fontsize=16)
        ax.set_xlabel('')
        ax.set_ylabel(val, fontsize=18)
        ax.tick_params(axis="both", labelsize=18)
        ax.set_ylim(lims)
        sns.despine(top=False, right=False)

    plt.tight_layout()
    print('')
    print('saving model accuracies to: {:}model_accuracies-{:}-{:}-{:}.png'.format(outpath, regress, run_pca, parc))
    plt.savefig('{:}model_accuracies-{:}-{:}-{:}.png'.format(outpath, regress, run_pca, parc))




#####################################################################################################
# HELPER FUNCTIONS
#####################################################################################################
# plot function for later
def plot_age_scatters(true, predicted, colors, ax=None, palette=None):

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(4,4))
    if palette is None:
        palette = 'viridis'

    ax.scatter(true, predicted, edgecolor='gray', s=50, alpha=0.5, c=colors, cmap=palette)
    ax.set_xlim(0,25)
    ax.set_ylim(0,25)
    lims = [0,25]
    ax.plot(lims, lims, 'k-', lw=1, alpha=0.75, zorder=0)

    ax.set_xlabel('age', fontsize=16)
    ax.set_ylabel('predicted age', fontsize=16)

    ax.tick_params(axis="both", labelsize=18)


if __name__ == '__main__':
    main()

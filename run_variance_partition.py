import pandas as pd
import numpy as np

import yaml, os

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from functions.misc import partition_variance, deconfound


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

  # linear model
  lm = LinearRegression()

  # set up models
  for model in ['linear', 'nonlinear', 'ensemble']:

      #####################################################################################################
      # LOADING
      # load previously calculated model predictions and explanations
      #####################################################################################################

      # load data - cross-validated model explanations and predictions
      model_explanations = pd.read_csv('{:}{:}-model-feature-explanations-{:}-{:}-{:}-{:}.csv'.format(genpath, model, run_combat, regress, run_pca, parc))
      model_predictions = pd.read_csv('{:}model_predictions-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc))

      # metadata
      subject_info = model_explanations.loc[:, model_explanations.columns.isin(['ID', 'Age', 'Male', 'Site', 'fold'])]
      # feature importances
      explanations =  model_explanations.loc[:, ~model_explanations.columns.isin(['ID', 'Age', 'Male', 'Site', 'fold'])]
      # brain age deltas
      deltas = model_predictions[model+'_uncorr_preds'] - model_predictions.Age

      # metric data
      metric_names = ['thickness', 'area']
      metric_data = []
      for metric in metric_names:
          chkfile = genpath + metric + '-parcellated-data-' + parc + '.csv'
          metric_data.append(np.loadtxt(chkfile, delimiter=','))

      #####################################################################################################
      # VARIANCE PARTITION
      # how much variance in brain age delta is explained by confounders
      #####################################################################################################
      # PCA
      dat = pd.DataFrame(pca.fit_transform(explanations))
      print('{:}: PCA w/ {:} components = {:.2f}% variance explained'.format(model, n_pca_comps, 100*pca.explained_variance_ratio_.sum()))

      # add in confounders
      dat['age'] = subject_info.Age
      dat['sex'] = subject_info.Male
      dat['meanthick'] = np.mean(metric_data[0],1)
      dat['meanarea'] = np.mean(metric_data[1],1)
      dat['deltas'] = deltas

      # confounders
      conf = ['age', 'sex', 'meanthick', 'meanarea']
      # predictors
      pred = list(dat.columns[~dat.columns.isin(['age', 'sex', 'meanthick','meanarea', 'deltas'])])
      # calculate variance partition
      var_table = partition_variance('deltas', conf, pred, dat)

      # save out
      var_table.to_csv('{:}variance-partition-{:}-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=None)

      #####################################################################################################
      # PARTIAL CORRELATIONS
      # calculate partial correlations for explanations and confounds
      #####################################################################################################
      # for plots
      # explanations given confound (x|Z)
      deconfounded_pcs = deconfound(pca.fit_transform(explanations), dat[conf]).data
      # delta given confounds (y|Z)
      deconfounded_delta = deconfound(deltas, dat[conf]).data
      # partial association between f(x | Z) and (y | Z)
      predicted_delta = lm.fit(deconfounded_pcs, deconfounded_delta).predict(deconfounded_pcs)
      # collate
      delta_given_confounds = pd.DataFrame(np.stack((deltas, deconfounded_delta, predicted_delta)).T, columns=['deltas', 'deconfounded_deltas', 'predicted_deltas'])

      # confounds given explanations (Z | x)
      deconfounded_confs = deconfound(dat[conf], pca.fit_transform(explanations)).data
      # (y | x)
      deconfounded_delta = deconfound(deltas, pca.transform(explanations)).data
      # (y | x) ~ f(Z | x)
      predicted_delta = lm.fit(deconfounded_confs, deconfounded_delta).predict(deconfounded_confs)
      # collate
      delta_given_explanations = pd.DataFrame(np.stack((deltas, deconfounded_delta, predicted_delta)).T, columns=['deltas', 'deconfounded_deltas', 'predicted_deltas'])

      # save out
      delta_given_confounds.to_csv('{:}partial-correlations-explanations-{:}-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=None)
      delta_given_explanations.to_csv('{:}partial-correlations-confounds-{:}-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=None)

if __name__ == '__main__':
    main()

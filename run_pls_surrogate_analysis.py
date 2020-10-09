import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import yaml, os

from sklearn.preprocessing import StandardScaler

from functions.pls_models import run_plsr, plsr_training_curve
from functions.misc import create_surrogates

from brainsmash.workbench.geo import cortex
from brainsmash.workbench.geo import parcellate

from tqdm import tqdm

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
	ss = StandardScaler()

	# surrogate analysis
	n_surrogates = 500

	# pls
	plsr_comps = 1

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
	# Load deconfounded data, create surrogates and perform PLS
	##############################################################################################################
	# load (deconfounded) brainage explanations

	deconfounded_model_explanations = np.load('{:}{:}-model-all-fold-deconfounded-feature-explanations-{:}-{:}-{:}-{:}.npy'.format(genpath, model, run_combat, regress, run_pca, parc))
	delta_deconf = np.load('{:}{:}-model-all-fold-deconfounded-delta-{:}-{:}-{:}-{:}.npy'.format(genpath, model, run_combat, regress, run_pca, parc))
	n_sub, n_feat, n_fold = np.shape(deconfounded_model_explanations)

	# folds
	cv_preds = pd.read_csv('{:}model_predictions-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc))
	cv_folds = cv_preds.fold.values

	surrogate_target_delta = np.zeros((len(delta_deconf)))
	surrogate_predicted_delta = np.zeros((len(delta_deconf), n_surrogates))
	surrogate_explained_delta_var = np.zeros((plsr_comps, n_fold, n_surrogates))
	surrogate_subject_scores = np.zeros((np.shape(deconfounded_model_explanations)[0], plsr_comps, n_surrogates))

	# train, test curves
	comp_choice = np.arange(10)+1
	surrogate_train_accuracy = np.zeros((len(comp_choice), n_fold, n_surrogates))
	surrogate_test_accuracy = np.zeros((len(comp_choice), n_fold, n_surrogates))

	for f in np.arange(n_fold):
		print('')
		print('************* FOLD: {:} **********'.format(f+1))

		if os.path.exists('{:}{:}-model-fold-{:}-{:}surrogates-{:}.npy'.format(surrogates, model, f+1, n_surrogates, parc)):
			print('surrogate maps already calculated')
			fold_surrogates = np.load('{:}{:}-model-fold-{:}-{:}surrogates-{:}.npy'.format(surrogates, model, f+1, n_surrogates, parc))
		else:
			# create surrogate maps
			fold_surrogates = get_all_surrogates(deconfounded_model_explanations[:,:,f], n_surrogates, surrogates, parc)
			np.save('{:}{:}-model-fold-{:}-{:}surrogates-{:}.npy'.format(surrogates, model, f+1, n_surrogates, parc), fold_surrogates)

		# perform PLS for 1 comp
		print('')
		print('for n components = {:}'.format(plsr_comps))
		print('cross-validating PLS model with surrogate data')

		# PLS
		train_Y, test_Y = delta_deconf[cv_folds!=f+1, f], delta_deconf[cv_folds==f+1, f]
		surrogate_target_delta[cv_folds==f+1] = test_Y

		for n in tqdm(np.arange(n_surrogates)):
			train_data = fold_surrogates[cv_folds!=f+1,:,n]
			test_data = fold_surrogates[cv_folds==f+1,:,n]

			# preprocess with scaling
			train_X, test_X = ss.fit_transform(train_data), ss.transform(test_data)

			# run
			regional_loadings, component_scores, component_loadings, coefs, weights = run_plsr(train_X, train_Y, n_comps=plsr_comps)

			# collect
			surrogate_predicted_delta[cv_folds==f+1, n] = test_X.dot(coefs.reshape(-1))

			# explained variance
			surrogate_explained_delta_var[:,f,n] = component_loadings**2
			# predicted subject scores
			surrogate_subject_scores[cv_folds==f+1,:,n] = test_X.dot(weights)

			# training_curve
			surrogate_fold_train_accuracy, surrogate_fold_test_accuracy = plsr_training_curve(train_X, train_Y, test_X, test_Y, components = comp_choice)
			surrogate_train_accuracy[:,f,n] = surrogate_fold_train_accuracy
			surrogate_test_accuracy[:,f,n] = surrogate_fold_test_accuracy

	# save out
	surrogate_predictions= pd.DataFrame(surrogate_predicted_delta)
	surrogate_predictions.insert(0,'fold',cv_folds)
	surrogate_predictions.insert(0,'target',surrogate_target_delta)
	surrogate_predictions.to_csv('{:}surrogate-{:}-model_predictions-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=None)
	# training curves
	train_acc = pd.DataFrame(np.mean(surrogate_train_accuracy, axis=1))
	train_acc.insert(0, 'components', comp_choice)
	train_acc.to_csv('{:}surrogate-{:}-model_training_curves-train-accuracy-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=None)
	test_acc = pd.DataFrame(np.mean(surrogate_test_accuracy, axis=1))
	test_acc.insert(0, 'components', comp_choice)
	test_acc.to_csv('{:}surrogate-{:}-model_training_curves-test-accuracy-{:}-{:}-{:}-{:}.csv'.format(outpath, model, run_combat, regress, run_pca, parc), index=None)

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
										 n_repeats=n_surrogates)

			for n,i in enumerate(surrogate_exp_maps):
				fold_surrogates[:,start:end,n] = i

	return fold_surrogates


if __name__ == '__main__':
	main()

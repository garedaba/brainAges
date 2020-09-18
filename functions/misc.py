import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from neurocombat_sklearn import CombatModel
from brainsmash.mapgen.base import Base
from joblib import Parallel, delayed

from copy import deepcopy

# metric data processing
def pre_process_metrics(metric_names, metric_data, subject_data, train_idx, test_idx, params):
    """Takes list of metric data arrays and returned single feature set with requested processing

    metric_names : list of strings, names of metrics
    metric_data : list of arrays, n x p metric data
    subject_data : pandas df, containing subject metadata, age, sex etc.
    train_idx : training data indices
    test_idx : test data indices
    params : dict, preprocessing configuration parameters

    returns :
    processed_train_data, processed_test_data :  arrays, processed feature sets
    """

    processed_train_data = []
    processed_test_data = []
    pca_model = []
    for metric_name, metric in zip(metric_names, metric_data):

        print('')
        print(metric_name)

        train_metric = metric[train_idx, :]
        test_metric = metric[test_idx, :]

        if params['combat']:
            print('performing comBat normalisation')
            train_metric, test_metric = harmonise(metric, subject_data, train_idx, test_idx, batch_covar='Site', discrete_covar='Male', continuous_covar='Age')

        if params['regress']:
            print('regressing global measures....')
            train_metric, test_metric = global_residualise(train_metric, test_metric)

        processed_train_data.append(train_metric)
        processed_test_data.append(test_metric)

    # combine metrics
    processed_train_data = np.concatenate(processed_train_data, axis=1)
    processed_test_data = np.concatenate(processed_test_data, axis=1)

    return processed_train_data, processed_test_data

# harmonise
def harmonise(data, subject_info, train_idx, test_idx, batch_covar='Site', discrete_covar=None, continuous_covar=None):
    """ Perform comBat harmonisation using the neurocombat-sklearn package: https://github.com/Warvito/neurocombat_sklearn
        combat model is estimated in training data and applied to both training and test

        Parameters
        ----------
        data : n subject x p voxels/vertices array containing brain tissue metrics
        subject_info : pd.DatFrame, n subject x q variables
        train_idx, test_idx : row index of train and test samples
        batch_covar : str, matching column label of unwanted variance, e.g.: Site
        discrete_covar : list of str, columns w/ discrete data whose variance we want to keep in data
        continuous_covar : list of str, columns w/ continuous data whose variance we want to keep in data

        Returns
        -------
        harmonised_train_data, harmonised_test_data : corrected tissue metrics
        """
    combat = CombatModel()

    train_batch = subject_info[batch_covar].iloc[train_idx][:,np.newaxis]
    test_batch = subject_info[batch_covar].iloc[test_idx][:,np.newaxis]

    if isinstance(discrete_covar, list):
        train_disc = subject_info[discrete_covar].iloc[train_idx].values
        test_disc = subject_info[discrete_covar].iloc[test_idx].values
    else:
        train_disc = subject_info[discrete_covar].iloc[train_idx].values.reshape(-1,1) if discrete_covar is not None else None
        test_disc = subject_info[discrete_covar].iloc[test_idx].values.reshape(-1,1) if discrete_covar is not None else None

    if  isinstance(continuous_covar, list):
        train_cont = subject_info[continuous_covar].iloc[train_idx].values
        train_cont = subject_info[continuous_covar].iloc[test_idx].values
    else:
        train_cont = subject_info[continuous_covar].iloc[train_idx].values.reshape(-1,1) if continuous_covar is not None else None
        test_cont = subject_info[continuous_covar].iloc[test_idx].values.reshape(-1,1) if continuous_covar is not None else None


    combat.fit(data[train_idx,:], train_batch, discrete_covariates=train_disc, continuous_covariates=train_cont)
    harmonised_train_data = combat.transform(data[train_idx,:], train_batch, train_disc, train_cont)
    harmonised_test_data = combat.transform(data[test_idx,:], test_batch, test_disc, test_cont)

    return harmonised_train_data, harmonised_test_data


# residulise
def global_residualise(train_data, test_data):
    """ Regress variation due to global metric (mean thickness etc) from brain tissue metrics (n x p matrix).
        Returns deconfounded data. Data is scaled using linear regression on a voxel (vertex) wise basis

        Parameters
        ----------
        train_data : n subject x p voxels/vertices array containing brain tissue metrics
        test_data : as above for data to be transformed

        Returns
        -------
        deconfounded_train_data, deconfounded_test_data : tissue metrics corrected for confound
        """

    lr = LinearRegression()
    mean_train_data = np.mean(train_data, axis=0)

    c_train = np.mean(train_data, axis=1).reshape(-1,1) # same number of regions for all subjects so mean is equiv. to total for regresion
    c_test = np.mean(test_data, axis=1).reshape(-1,1)

    deconfounded_train_data = train_data - lr.fit(c_train, train_data).predict(c_train)
    deconfounded_train_data += mean_train_data

    deconfounded_test_data = (test_data - lr.predict(c_test)) + mean_train_data

    return deconfounded_train_data, deconfounded_test_data


def create_surrogates(subject_idx, features, distMat, n_samples=10):
    """ Create surrogate maps matched for spatial autocorrelation (SA) using BrainSmash
        Takes a list of subjects and creates n_samples surrogate maps based on SA of each subject's feature map

        parameters
        ----------
        subjects_idx : subject indices to select feature maps from
        features : array, n_subject x p variable feature matrix
        distMat : precalculated geometric distance matrix with matched parcellation to subject features
        n_samples : number of surrogate maps to produce per subject in subjects_idx

        returns
        -------
        all_surrogates : array, (len(subject_idx) x n_samples)-by-p variables surrogate maps
    """
    surrogate_maps = []

    for p in subject_idx:
        gen = Base(features[p,:], distMat, resample=True)
        surrogate_maps.append(gen(n=n_samples))

    all_surrogates = np.vstack(surrogate_maps)

    return all_surrogates


def bootstrap_surrogates(features, distMat, n_boot=10, n_subjects=100, n_samples=10, n_jobs=-1):
    """ Randomly sample n_subjects from features and calculate n_samples surrogate maps, repeat n_boot times

        parameters
        ----------
        features : array, n_subjects x p_variables
        distMat : precalculated geometric distance matrix with matched parcellation to subject features
        n_boot : number of times to run surrogate generation
        n_subjects : how many subject maps to use in surogate generation
        n_samples : number of maps per subject feature map
        n_jobs : number of CPUs

        returns:
        --------
        all_surrogates : n_boot x (n_subjects x n_samples) x p_features, surrogate map data
    """
    print('calculating surrogates based on {:} samples of {:} subjects, with {:} maps each'.format(n_boot, n_subjects, n_samples))
    # random selection of subjects
    p_idx = []
    for i in np.arange(n_boot):
        p_idx.append(np.random.choice(len(features), size=n_subjects, replace=False))

    all_surrogates = Parallel(n_jobs=n_jobs, verbose=2)(delayed(create_surrogates)
                                                    (j, features, distMat, n_samples=n_samples) for j in p_idx)

    return all_surrogates

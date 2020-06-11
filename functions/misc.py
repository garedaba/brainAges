import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from copy import deepcopy

# metric data processing
def pre_process_metrics(metric_names, metric_data, train_idx, test_idx, params):
    """Takes list of metric data arrays and returned single feature set with requested processing

    metric_names : list of strings, names of metrics
    metric_data : list of arrays, n x p metric data
    train_idx : training data indices
    test_idx : test data indices
    params : dict, preprocessing configuration parameters

    returns :
    processed_train_data, processed_test_data :  arrays, processed feature sets
    pca_models : list, pca models if run_pca = 'PCA'
    """

    processed_train_data = []
    processed_test_data = []
    pca_models=[]

    for metric_name, metric in zip(metric_names, metric_data):

        print('')
        print(metric_name)

        train_metric = metric[train_idx, :]
        test_metric = metric[test_idx, :]

        if params['regress']:
            print('regressing global measures....')
            train_metric, test_metric = global_residualise(train_metric, test_metric)

        if params['pca']:
            ncomps = params['pca_comps']
            pca = PCA(whiten=True, n_components=ncomps)
            train_metric = pca.fit_transform(train_metric)
            test_metric = pca.transform(test_metric)
            print('performing PCA whitening....n = {:} components explained {:.2f}% variance'.format(ncomps, 100*pca.explained_variance_ratio_.sum()))
            # save for later
            pca_models.append(deepcopy(pca))

        processed_train_data.append(train_metric)
        processed_test_data.append(test_metric)

    # combine metrics
    processed_train_data = np.concatenate(processed_train_data, axis=1)
    processed_test_data = np.concatenate(processed_test_data, axis=1)

    return processed_train_data, processed_test_data, pca_models

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

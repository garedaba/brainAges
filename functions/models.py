import sys

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform, norm, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, DotProduct
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import numpy as np

import shap

def get_linear_model(params):
    """output a sparse linear regressor with randomised parameter search over nested 3-fold CV
    params: dict, containing details on PCA if required

    returns:
    model: sklearn estimator
    """

    ss = StandardScaler()
    lr = ElasticNet(selection='random', random_state=42)  # EN

    if params['pca']:
        pca = PCA(n_components=params['pca_comps'], whiten=True)
        lr_model = Pipeline(steps=(['scale', ss], ['pca', pca], ['model', lr])) # pipeline
    else:
        lr_model = Pipeline(steps=(['scale', ss], ['model', lr]))     # pipeline

    lr_model_params = {
            'model__alpha': loguniform(1e-1, 1e3),
            'model__l1_ratio': uniform(0.1, .9)
    }

    # model: classifier with randomised parameter search over nested 3-fold CV
    linear_model = RandomizedSearchCV(lr_model, lr_model_params, n_iter=100, cv=5)

    return clone(linear_model)

def get_nonlinear_model(params):
    """output a nonlinear GPR model
    params: dict, containing details on PCA if required

    returns:
    model: sklearn estimator
    """
    kernel = 1 * RBF(length_scale=10.0, length_scale_bounds=(1.0, 1000.0)) + WhiteKernel(10.0, noise_level_bounds=(5.0,1000))

    ss = StandardScaler()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True, n_restarts_optimizer=10)

    if params['pca']:
        pca = PCA(n_components=params['pca_comps'], whiten=True)
        nonlinear_model = Pipeline(steps=(['scale', ss], ['pca', pca], ['model', gpr])) # pipeline
    else:
        nonlinear_model = Pipeline(steps=(['scale', ss], ['model', gpr]))

    return clone(nonlinear_model)

def get_ensemble_model(params):
    """output a nonlinear XGBoost regressor with randomised parameter search over nested 3-fold CV
    params: dict, containing details on PCA if required

    returns:
    model: sklearn estimator
    """
    ss = StandardScaler()
    xgb_reg = xgb.XGBRegressor(objective="reg:squarederror",n_jobs=1, base_score=12, learning_rate=0.05, random_state=42)

    if params['pca']:
        pca = PCA(n_components=params['pca_comps'], whiten=True)
        xgb_model = Pipeline(steps=(['scale', ss], ['pca', pca], ['model', xgb_reg])) # pipeline
    else:
        xgb_model = Pipeline(steps=(['scale', ss], ['model', xgb_reg]))

    xgb_model_params = {
        "model__n_estimators": [100,250,500],
        "model__colsample_bytree": uniform(0.5, 0.5), # default 1
        "model__min_child_weight": randint(1,6),        #deafult 1
        "model__max_depth": randint(2, 5),            # default 3, 3-10 -
        "model__subsample": uniform(0.5, 0.5),        # default 1
        "model__reg_lambda": loguniform(1e1,1e2)       # l2 reg, default 1
    }

    # model: classifier with randomised parameter search over nested 3-fold CV (more iters to account for large space)
    ensemble_model = RandomizedSearchCV(xgb_model, xgb_model_params, n_iter=250, cv=5, verbose=1, n_jobs=5)

    return clone(ensemble_model)


def get_model_explanations(model, train_data, test_data, background_samples=10, nsamples='auto', num_features=50):
    """runs KernelSHAP on model and returns subjectwise model explanations

    model : trained sklearn estimator
    train_data : n x p array of training data used in model training
    test_data : n x p array of test data
    background_samples : number of kmeans clusters used to summarise training data, fewer=faster
    nsamples : number of times to reevaluate model to estimate Shapley values, more=lower variance
    num_features : number of features to include in local model
    """
    explainer = shap.KernelExplainer(model.predict, shap.kmeans(train_data, background_samples), link='identity')
    explanations = explainer.shap_values(test_data, nsamples=nsamples, l1_reg='num_features({:})'.format(num_features))

    return explanations

def get_age_corrected_model_explanations(model, train_data, train_age, test_data, age=12, background_samples=10, nsamples='auto', num_features=50):
    """runs KernelSHAP on model and returns subjectwise model explanations. In contrast to get_model_explanations()
    this model uses a background set drawn from the training data matched by age. As such the expected value of the
    explainer is close the test sample age and, as all explanations are relative to it, the shap values describe
    how features drive the decision away from the subject age, not the group mean age.

    model : trained sklearn estimator
    train_data : n x p array of training data used in model training
    train_age : n array of training ages used for sample selection
    test_data : n x p array of test data
    age : float, age of test sample
    background_samples : number of age-matched samples to draw from training data, fewer=faster
    nsamples : number of times to reevaluate model to estimate Shapley values, more=lower variance
    num_features : number of features to include in local model
    """
    ranked_samples = np.argsort(abs(train_age.values - age))[:background_samples]
    explainer = shap.KernelExplainer(model.predict, train_data[ranked_samples,:], link='identity')
    explanations = explainer.shap_values(test_data, nsamples=nsamples, l1_reg='num_features({:})'.format(num_features), silent=True)

    return explanations


def correct_age_predictions(train_preds, train_age, test_preds, test_age):
    """fits a linear model to predicted age residuals to account for regression to the mean effect commonly
    seen in brain age estimation models

    train_preds, array, model predictions of 'brain age' in training samples
    train_age, array, true age of training samples
    test_preds, array, model predictions of 'brain age' in test sample
    test_age, array, true age of testing samples

    returns:
    corrected_predictions, array, age-corrected model predictions
    """
    lr = LinearRegression()

    train_resids = np.array(train_preds - train_age)
    test_resids = np.array(test_preds - test_age)

    # fit model
    lr.fit(train_age[:,np.newaxis], train_resids)

    # predict test residuals using age
    pred_resid = lr.predict(test_age[:,np.newaxis])

    # correct model predictions
    corrected_predictions = test_preds - pred_resid

    return corrected_predictions

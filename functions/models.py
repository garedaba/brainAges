import sys

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.base import clone
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform, norm, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, DotProduct

import xgboost as xgb
import numpy as np

import shap



def get_linear_model():
    """output a sparse linear regressor with randomised parameter search over nested 3-fold CV

    returns:
    model: sklearn estimator
    """

    ss = StandardScaler()
    lr = ElasticNet(selection='random', random_state=42)  # EN

    lr_model = Pipeline(steps=(['scale', ss], ['model', lr]))     # pipeline

    lr_model_params = {
            'model__alpha': loguniform(1e-1, 1e3),
            'model__l1_ratio': uniform(0.1, .9)
    }

    # model: classifier with randomised parameter search over nested 3-fold CV
    linear_model = RandomizedSearchCV(lr_model, lr_model_params, n_iter=250, cv=5)

    return clone(linear_model)

def get_nonlinear_model():
    """output a nonlinear GPR model

    returns:
    model: sklearn estimator
    """
    kernel = 1 * RBF([5.0]) + WhiteKernel()

    ss = StandardScaler()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True, n_restarts_optimizer=10)

    nonlinear_model = Pipeline(steps=(['scale', ss], ['model', gpr]))

    return clone(nonlinear_model)

def get_ensemble_model():
    """output a nonlinear XGBoost classifier with randomised parameter search over nested 3-fold CV

    returns:
    model: sklearn estimator
    """
    ss = StandardScaler()
    xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", base_score=12, random_state=42)

    xgb_model = Pipeline(steps=(['scale', ss], ['model', xgb_reg]))

    xgb_model_params = {
        "model__colsample_bytree": uniform(0.5, 0.5), # default 1
        "model__gamma": loguniform(1e-2, 1e0),        # default 0
        "model__learning_rate": uniform(0.03, 0.37),  # default 0.3
        "model__max_depth": randint(2, 6),            # default 3
        "model__n_estimators": randint(50, 150),      # default 100
        "model__subsample": uniform(0.5, 0.5),        # default 1
    }

    # model: classifier with randomised parameter search over nested 3-fold CV (more iters to account for large space)
    ensemble_model = RandomizedSearchCV(xgb_model, xgb_model_params, n_iter=250, cv=5)

    return clone(ensemble_model)


def get_model_explanations(model, train_data, test_data, background_samples=20, nsamples=5000, num_features=50):
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

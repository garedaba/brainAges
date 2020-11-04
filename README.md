# brainAges
Examining individual variation in brain age estimates during typical development.


### Requirements
_Python 3.6.10_  
Required packages include: `numpy`, `scipy`, `shap`, `scikit-learn`, `nibabel`

All installed packages are shown in req.txt
To clone environment try: `conda create -n new environment --file req.txt`

### Neuroimaging data
PING data is available from the [NIMH Data Archive](https://nda.nih.gov/about.html) subject to data use agreement. Study link: https://nda.nih.gov/study.html?id=905

### Analysis
_1. `run_brain_age_models.py`_  
- Load surface data, parcellate and preprocess  
- Train and test brain age models
- Calculate individual model explanations  

Output:
- Cross-validated model accuracies
- Age predictions  
- Model explanations

_2. `run_variance_partition.py`_
- Estimate variance explained in brain age delta by confounding variables

Output:
- % variance explained in delta by confounds and explanations

_3. `run_deconfound_data.py`_
- For each model, remove variance associated with confounding variables from delta and model explanations using linear regression
- Performed within 5-fold cross-validation folds   

Output:
- Model explanations and brain age delta estimates with variance due to confounds removed

_4. `run_surrogates.py`_
- Use `BrainSMASH` to generate random surrogate maps with matched spatial autocorrelation  

Output:
- Surrogate maps (n_subjects x p_features x s surrogates)

_5. `run_explanation_correlations.py`_
- Measure mean similarity of model explanations within subjects (across train/test folds)
- Measure mean similarity between each subject and every other
- Measure mean similarity between each subject and set of random surrogates

Output:
- Mean cosine similarity for 'within', 'between' and 'random' comparisons

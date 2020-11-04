# brainAges
Examining individual variation in brain age estimates during typical development.


### Requirements
_Python 3.6.10_  
Required packages include: `numpy`, `scipy`, `shap`, `scikit-learn`, `nibabel`

All installed packages are shown in req.txt
To clone environment try: `conda create -n new environment --file req.txt`

### Neuroimaging data
PING data is available from the [NIMH Data Archive](https://nda.nih.gov/about.html) subject to data use agreement. Study link: https://nda.nih.gov/study.html?id=905 (requires NDAR login)

### Analysis
**1. `run_brain_age_models.py`**  
Load surface data, parcellate and preprocess then train and test brain age models and calculate individual model explanations  

  _Output:_
- Cross-validated model accuracies
- Age predictions  
- Model explanations

**2. `run_variance_partition.py`**  
Estimate variance explained in brain age delta by confounding variables

  _Output:_
- % variance explained in delta by confounds and explanations

**3. `run_deconfound_data.py`**  
For each model, remove variance associated with confounding variables from delta and model explanations using linear regression (performed within 5-fold cross-validation folds)

  _Output:_
- Model explanations and brain age delta estimates with variance due to confounds removed

**4. `run_surrogates.py`**  
Use `BrainSMASH` to generate random surrogate maps with matched spatial autocorrelation  

  _Output:_
- Surrogate maps (n_subjects x p_features x s surrogates)

**5. `run_explanation_correlations.py`**  
Measure mean similarity of model explanations within subjects (across train/test folds), mean similarity between each subject and every other and mean similarity between each subject and set of random surrogates

  _Output:_
- Mean cosine similarity for 'within', 'between' and 'random' comparisons

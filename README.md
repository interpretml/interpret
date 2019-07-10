# InterpretML - Alpha Release

![License](https://img.shields.io/github/license/microsoft/interpret.svg?style=flat-square)
![Python Version](https://img.shields.io/pypi/pyversions/interpret.svg?style=flat-square)
![Package Version](https://img.shields.io/pypi/v/interpret.svg?style=flat-square)
![Build Status](https://img.shields.io/azure-devops/build/ms/interpret/151/master.svg?style=flat-square)
![Maintenance](https://img.shields.io/maintenance/yes/2019.svg?style=flat-square)

*Equal contributions: Samuel Jenkins & Harsha Nori & Paul Koch & Rich Caruana*

<br/>

> ### In the beginning machines learned in darkness, and data scientists struggled in the void to explain them.
>
> ### Let there be light.

<br/>

InterpretML is an open-source python package for training interpretable models and explaining blackbox systems. Interpretability is essential for:
- Model debugging - Why did my model make this mistake?
- Detecting bias - Does my model discriminate?
- Human-AI cooperation - How can I understand and trust the model's decisions?
- Regulatory compliance - Does my model satisfy legal requirements?
- High-risk applications - Healthcare, finance, judicial, ...

Historically, the most intelligible models were not very accurate, and the most accurate models were not intelligible. Microsoft Research has developed an algorithm called the Explainable Boosting Machine (EBM)<sup>[*](#ebm-footnote)</sup> which has both high accuracy and intelligibility. EBM uses modern machine learning techniques like bagging and boosting to breathe new life into traditional GAMs (Generalized Additive Models). This makes them as accurate as random forests and gradient boosted trees, and also enhances their intelligibility and editability.

<br/>

[*Notebook for reproducing table*](https://nbviewer.jupyter.org/github/Microsoft/interpret/blob/master/benchmarks/EBM%20Classification%20Comparison.ipynb)

| Dataset/AUROC | Domain  | Logistic Regression | Random Forest | XGBoost        | Explainable Boosting Machine |
|---------------|---------|:-------------------:|:-------------:|:--------------:|:----------------------------:|
| Adult Income  | Finance | .907±.003           | .903±.002     | .922±.002      | **_.928±.002_**              |
| Heart Disease | Medical | .895±.030           | .890±.008     | .870±.014      | **_.916±.010_**              |
| Breast Cancer | Medical | **_.995±.005_**     | .992±.009     | **_.995±.006_**| **_.995±.006_**              |
| Telecom Churn | Business| .804±.015           | .824±.002     | .850±.006      | **_.851±.005_**              |
| Credit Fraud  | Security| .979±.002           | .950±.007     | **_.981±.003_**| .975±.005                    |

<br/>

In addition to EBM, InterpretML also supports methods like LIME, SHAP, linear models, partial dependence, decision trees and rule lists.  The package makes it easy to compare and contrast models to find the best one for your needs.

<a name="ebm-footnote">*</a> *EBM is a fast implementation of GA<sup>2</sup>M. Details on the algorithm can be found [here](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/KDD2015FinalDraftIntelligibleModels4HealthCare_igt143e-caruanaA.pdf).*

---

## Installation

Python 3.5+ | Linux, Mac OS X, Windows
```sh
pip install numpy scipy pyscaffold
pip install -U interpret
```

## Getting Started

Let's fit an Explainable Boosting Machine

```python
from interpret.glassbox import ExplainableBoostingClassifier

ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)

# EBM supports pandas dataframes, numpy arrays, and handles "string" data natively.
```

Understand the model
```python
from interpret import show

ebm_global = ebm.explain_global()
show(ebm_global)
```
![Global Explanation Image](examples/assets/readme_ebm_global_specific.PNG?raw=true)

<br/>

Understand individual predictions
```python
ebm_local = ebm.explain_local(X_test, y_test)
show(ebm_local)
```
![Local Explanation Image](examples/assets/readme_ebm_local_specific.PNG?raw=true)

<br/>

And if you have multiple models, compare them
```python
show([logistic_regression, decision_tree])
```
![Dashboard Image](examples/assets/readme_dashboard.PNG?raw=true)

<br/>

## Example Notebooks

- [Interpretable models for binary classification](https://nbviewer.jupyter.org/github/Microsoft/interpret/blob/master/examples/notebooks/Interpretable%20Classification%20Methods.ipynb)
- [Interpretable models for regression](https://nbviewer.jupyter.org/github/Microsoft/interpret/blob/master/examples/notebooks/Interpretable%20Regression%20Methods.ipynb)
- [Blackbox interpretability for binary classification](https://nbviewer.jupyter.org/github/Microsoft/interpret/blob/master/examples/notebooks/Explaining%20Blackbox%20Classifiers.ipynb)
- [Blackbox interpretability for regression](https://nbviewer.jupyter.org/github/Microsoft/interpret/blob/master/examples/notebooks/Explaining%20Blackbox%20Regressors.ipynb)

## Roadmap

Currently we're working on:
- Multiclass Classification Support
- Missing Values Support
- Improved Categorical Encoding

...and lots more! Get in touch to find out more.

## Contributing

If you are interested contributing directly to the code base, please see [CONTRIBUTING.md](./CONTRIBUTING.md).

## Acknowledgements

Many people have supported us along the way. Check out [ACKNOWLEDGEMENTS.md](./ACKNOWLEDGEMENTS.md)!

<br/>

We also build on top of many great packages. Please check them out!

[plotly](https://github.com/plotly/plotly.py) |
[dash](https://github.com/plotly/dash) | 
[scikit-learn](https://github.com/scikit-learn/scikit-learn) | 
[lime](https://github.com/marcotcr/lime) |
[shap](https://github.com/slundberg/shap) |
[salib](https://github.com/SALib/SALib) |
[skope-rules](https://github.com/scikit-learn-contrib/skope-rules) |
[gevent](https://github.com/gevent/gevent) | 
[joblib](https://github.com/joblib/joblib) |
[pytest](https://github.com/pytest-dev/pytest) | 
[jupyter](https://github.com/jupyter/notebook) 


## Contact us

There are multiple ways to get in touch:
- Email us at interpret@microsoft.com
- Or, feel free to raise a GitHub issue


## Reporting Security Issues (we had to include this...)

Security issues and bugs should be reported privately, via email, to the Microsoft Security
Response Center (MSRC) at [secure@microsoft.com](mailto:secure@microsoft.com). You should
receive a response within 24 hours. If for some reason you do not, please follow up via
email to ensure we received your original message. Further information, including the
[MSRC PGP](https://technet.microsoft.com/en-us/security/dn606155) key, can be found in
the [Security TechCenter](https://technet.microsoft.com/en-us/security/default).

<br/>
<br/>
<br/>
<br/>
<br/>

<br/>
<br/>
<br/>
<br/>
<br/>

<br/>
<br/>
<br/>
<br/>
<br/>

<br/>
<br/>
<br/>
<br/>
<br/>

<br/>
<br/>
<br/>
<br/>
<br/>

<br/>
<br/>
<br/>
<br/>
<br/>

<br/>
<br/>
<br/>
<br/>
<br/>

<br/>
<br/>
<br/>
<br/>
<br/>


> ### If a tree fell in your random forest, would anyone notice?

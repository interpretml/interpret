# InterpretML

<a href="https://githubtocolab.com/interpretml/interpret/blob/develop/docs/interpret/python/examples/interpretable-classification.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/interpretml/interpret/develop?labpath=docs%2Finterpret%2Fpython%2Fexamples%2Finterpretable-classification.ipynb)
![License](https://img.shields.io/github/license/interpretml/interpret.svg?style=flat-square)
![Python Version](https://img.shields.io/pypi/pyversions/interpret.svg?style=flat-square)
![Package Version](https://img.shields.io/pypi/v/interpret.svg?style=flat-square)
![Conda](https://img.shields.io/conda/v/conda-forge/interpret)
![Build Status](https://github.com/interpretml/interpret/actions/workflows/ci.yml/badge.svg?branch=develop)
[![codecov](https://codecov.io/github/interpretml/interpret/branch/develop/graph/badge.svg?token=aPlXLsPEZD)](https://codecov.io/github/interpretml/interpret)
![Maintenance](https://img.shields.io/maintenance/yes/2099?style=flat-square)
<br/>
> ### In the beginning machines learned in darkness, and data scientists struggled in the void to explain them. 
> ### Let there be light.

InterpretML is an open-source package that incorporates state-of-the-art machine learning interpretability techniques under one roof. With this package, you can train interpretable glassbox models and explain blackbox systems. InterpretML helps you understand your model's global behavior, or understand the reasons behind individual predictions.

Interpretability is essential for:
- Model debugging - Why did my model make this mistake?
- Feature Engineering - How can I improve my model?
- Detecting fairness issues - Does my model discriminate?
- Human-AI cooperation - How can I understand and trust the model's decisions?
- Regulatory compliance - Does my model satisfy legal requirements?
- High-risk applications - Healthcare, finance, judicial, ...

![](https://github.com/interpretml/interpretml.github.io/blob/master/interpret-highlight.gif)

# Installation

Python 3.7+ | Linux, Mac, Windows
```sh
pip install interpret
# OR
conda install -c conda-forge interpret
```

# Introducing the Explainable Boosting Machine (EBM)

EBM is an interpretable model developed at Microsoft Research<sup>[*](#citations)</sup>. It uses modern machine learning techniques like bagging, gradient boosting, and automatic interaction detection to breathe new life into traditional GAMs (Generalized Additive Models). This makes EBMs as accurate as state-of-the-art techniques like random forests and gradient boosted trees. However, unlike these blackbox models, EBMs produce exact explanations and are editable by domain experts.

| Dataset/AUROC | Domain  | Logistic Regression | Random Forest | XGBoost         | Explainable Boosting Machine |
|---------------|---------|:-------------------:|:-------------:|:---------------:|:----------------------------:|
| Adult Income  | Finance | .907±.003           | .903±.002     | .927±.001       | **_.928±.002_**              |
| Heart Disease | Medical | .895±.030           | .890±.008     | .851±.018       | **_.898±.013_**              |
| Breast Cancer | Medical | **_.995±.005_**     | .992±.009     | .992±.010       | **_.995±.006_**              |
| Telecom Churn | Business| .849±.005           | .824±.004     | .828±.010       | **_.852±.006_**              |
| Credit Fraud  | Security| .979±.002           | .950±.007     | **_.981±.003_** | **_.981±.003_**              |

[*Notebook for reproducing table*](https://nbviewer.jupyter.org/github/interpretml/interpret/blob/develop/docs/benchmarks/ebm-classification-comparison.ipynb)

# Supported Techniques

| Interpretability Technique  | Type               |
|-----------------------------|--------------------|
| [Explainable Boosting](https://interpret.ml/docs/ebm.html)        | glassbox model     |
| [APLR](https://interpret.ml/docs/aplr.html)                       | glassbox model     |
| [Decision Tree](https://interpret.ml/docs/dt.html)                | glassbox model     |
| [Decision Rule List](https://interpret.ml/docs/dr.html)           | glassbox model     |
| [Linear/Logistic Regression](https://interpret.ml/docs/lr.html)   | glassbox model     |
| [SHAP Kernel Explainer](https://interpret.ml/docs/shap.html)      | blackbox explainer |
| [LIME](https://interpret.ml/docs/lime.html)                       | blackbox explainer |
| [Morris Sensitivity Analysis](https://interpret.ml/docs/msa.html) | blackbox explainer |
| [Partial Dependence](https://interpret.ml/docs/pdp.html)          | blackbox explainer |

# Train a glassbox model

Let's fit an Explainable Boosting Machine

```python
from interpret.glassbox import ExplainableBoostingClassifier

ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)

# or substitute with LogisticRegression, DecisionTreeClassifier, RuleListClassifier, ...
# EBM supports pandas dataframes, numpy arrays, and handles "string" data natively.
```

Understand the model
```python
from interpret import show

ebm_global = ebm.explain_global()
show(ebm_global)
```
![Global Explanation Image](./docs/readme/ebm-global.png?raw=true)

<br/>

Understand individual predictions
```python
ebm_local = ebm.explain_local(X_test, y_test)
show(ebm_local)
```
![Local Explanation Image](./docs/readme/ebm-local.png?raw=true)

<br/>

And if you have multiple model explanations, compare them
```python
show([logistic_regression_global, decision_tree_global])
```
![Dashboard Image](./docs/readme/dashboard.png?raw=true)

<br/>

If you need to keep your data private, use Differentially Private EBMs (see [DP-EBMs](https://proceedings.mlr.press/v139/nori21a/nori21a.pdf))

```python
from interpret.privacy import DPExplainableBoostingClassifier, DPExplainableBoostingRegressor

dp_ebm = DPExplainableBoostingClassifier(epsilon=1, delta=1e-5) # Specify privacy parameters
dp_ebm.fit(X_train, y_train)

show(dp_ebm.explain_global()) # Identical function calls to standard EBMs
```

<br/>
<br/>

For more information, see the [documentation](https://interpret.ml/docs).

<br/>

EBMs include pairwise interactions by default. For 3-way interactions and higher see this notebook: https://interpret.ml/docs/python/examples/custom-interactions.html

<br/>

Interpret EBMs can be fit on datasets with 100 million samples in several hours. For larger workloads consider using distributed EBMs on Azure SynapseML: [classification EBMs](https://learn.microsoft.com/en-us/fabric/data-science/explainable-boosting-machines-classification) and [regression EBMs](https://learn.microsoft.com/en-us/fabric/data-science/explainable-boosting-machines-regression)

<br/>
<br/>

# Acknowledgements

InterpretML was originally created by (equal contributions): Samuel Jenkins, Harsha Nori, Paul Koch, and Rich Caruana

EBMs are fast derivative of GA2M, invented by: Yin Lou, Rich Caruana, Johannes Gehrke, and Giles Hooker

Many people have supported us along the way. Check out [ACKNOWLEDGEMENTS.md](./ACKNOWLEDGEMENTS.md)!

We also build on top of many great packages. Please check them out!

[plotly](https://github.com/plotly/plotly.py) |
[dash](https://github.com/plotly/dash) |
[scikit-learn](https://github.com/scikit-learn/scikit-learn) |
[lime](https://github.com/marcotcr/lime) |
[shap](https://github.com/slundberg/shap) |
[salib](https://github.com/SALib/SALib) |
[skope-rules](https://github.com/scikit-learn-contrib/skope-rules) |
[treeinterpreter](https://github.com/andosa/treeinterpreter) |
[gevent](https://github.com/gevent/gevent) |
[joblib](https://github.com/joblib/joblib) |
[pytest](https://github.com/pytest-dev/pytest) |
[jupyter](https://github.com/jupyter/notebook)

# <a name="citations">Citations</a>

<details open>
  <summary><strong>InterpretML</strong></summary>
  <hr/>

  <details open>
    <summary>
      <em>"InterpretML: A Unified Framework for Machine Learning Interpretability" (H. Nori, S. Jenkins, P. Koch, and R. Caruana 2019)</em>
    </summary>
    <br/>
    <pre>
@article{nori2019interpretml,
  title={InterpretML: A Unified Framework for Machine Learning Interpretability},
  author={Nori, Harsha and Jenkins, Samuel and Koch, Paul and Caruana, Rich},
  journal={arXiv preprint arXiv:1909.09223},
  year={2019}
}
    </pre>
    <a href="https://arxiv.org/pdf/1909.09223.pdf">Paper link</a>
  </details>

  <hr/>
</details>

<details>
  <summary><strong>Explainable Boosting</strong></summary>
  <hr/>

  <details>
    <summary>
      <em>"Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission" (R. Caruana, Y. Lou, J. Gehrke, P. Koch, M. Sturm, and N. Elhadad 2015)</em>
    </summary>
    <br/>
    <pre>
@inproceedings{caruana2015intelligible,
  title={Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission},
  author={Caruana, Rich and Lou, Yin and Gehrke, Johannes and Koch, Paul and Sturm, Marc and Elhadad, Noemie},
  booktitle={Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={1721--1730},
  year={2015},
  organization={ACM}
}
    </pre>
    <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/KDD2015FinalDraftIntelligibleModels4HealthCare_igt143e-caruanaA.pdf">Paper link</a>
  </details>

  <details>
    <summary>
      <em>"Accurate intelligible models with pairwise interactions" (Y. Lou, R. Caruana, J. Gehrke, and G. Hooker 2013)</em>
    </summary>
    <br/>
    <pre>
@inproceedings{lou2013accurate,
  title={Accurate intelligible models with pairwise interactions},
  author={Lou, Yin and Caruana, Rich and Gehrke, Johannes and Hooker, Giles},
  booktitle={Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={623--631},
  year={2013},
  organization={ACM}
}
    </pre>
    <a href="https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf">Paper link</a>
  </details>

  <details>
    <summary>
      <em>"Intelligible models for classification and regression" (Y. Lou, R. Caruana, and J. Gehrke 2012)</em>
    </summary>
    <br/>
    <pre>
@inproceedings{lou2012intelligible,
  title={Intelligible models for classification and regression},
  author={Lou, Yin and Caruana, Rich and Gehrke, Johannes},
  booktitle={Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={150--158},
  year={2012},
  organization={ACM}
}
    </pre>
    <a href="https://www.cs.cornell.edu/~yinlou/papers/lou-kdd12.pdf">Paper link</a>
  </details>

  <details>
    <summary>
      <em>"Interpretability, Then What? Editing Machine Learning Models to Reflect Human Knowledge and Values" (Zijie J. Wang, Alex Kale, Harsha Nori, Peter Stella, Mark E. Nunnally, Duen Horng Chau, Mihaela Vorvoreanu, Jennifer Wortman Vaughan, Rich Caruana 2022)</em>
    </summary>
    <br/>
    <pre>
@article{wang2022interpretability,
  title={Interpretability, Then What? Editing Machine Learning Models to Reflect Human Knowledge and Values},
  author={Wang, Zijie J and Kale, Alex and Nori, Harsha and Stella, Peter and Nunnally, Mark E and Chau, Duen Horng and Vorvoreanu, Mihaela and Vaughan, Jennifer Wortman and Caruana, Rich},
  journal={arXiv preprint arXiv:2206.15465},
  year={2022}
}
    </pre>
    <a href="https://arxiv.org/pdf/2206.15465.pdf">Paper link</a>
  </details>

  <details>
    <summary>
      <em>"Axiomatic Interpretability for Multiclass Additive Models" (X. Zhang, S. Tan, P. Koch, Y. Lou, U. Chajewska, and R. Caruana 2019)</em>
    </summary>
    <br/>
    <pre>
@inproceedings{zhang2019axiomatic,
  title={Axiomatic Interpretability for Multiclass Additive Models},
  author={Zhang, Xuezhou and Tan, Sarah and Koch, Paul and Lou, Yin and Chajewska, Urszula and Caruana, Rich},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={226--234},
  year={2019},
  organization={ACM}
}
    </pre>
    <a href="https://arxiv.org/pdf/1810.09092.pdf">Paper link</a>
  </details>

  <details>
    <summary>
      <em>"Distill-and-compare: auditing black-box models using transparent model distillation" (S. Tan, R. Caruana, G. Hooker, and Y. Lou 2018)</em>
    </summary>
    <br/>
    <pre>
@inproceedings{tan2018distill,
  title={Distill-and-compare: auditing black-box models using transparent model distillation},
  author={Tan, Sarah and Caruana, Rich and Hooker, Giles and Lou, Yin},
  booktitle={Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society},
  pages={303--310},
  year={2018},
  organization={ACM}
}
    </pre>
    <a href="https://arxiv.org/pdf/1710.06169">Paper link</a>
  </details>

  <details>
    <summary>
      <em>"Purifying Interaction Effects with the Functional ANOVA: An Efficient Algorithm for Recovering Identifiable Additive Models" (B. Lengerich, S. Tan, C. Chang, G. Hooker, R. Caruana 2019)</em>
    </summary>
    <br/>
    <pre>
@article{lengerich2019purifying,
  title={Purifying Interaction Effects with the Functional ANOVA: An Efficient Algorithm for Recovering Identifiable Additive Models},
  author={Lengerich, Benjamin and Tan, Sarah and Chang, Chun-Hao and Hooker, Giles and Caruana, Rich},
  journal={arXiv preprint arXiv:1911.04974},
  year={2019}
}
    </pre>
    <a href="https://arxiv.org/pdf/1911.04974.pdf">Paper link</a>
  </details>

  <details>
    <summary>
      <em>"Interpreting Interpretability: Understanding Data Scientists' Use of Interpretability Tools for Machine Learning" (H. Kaur, H. Nori, S. Jenkins, R. Caruana, H. Wallach, J. Wortman Vaughan 2020)</em>
    </summary>
    <br/>
    <pre>
@inproceedings{kaur2020interpreting,
  title={Interpreting Interpretability: Understanding Data Scientists' Use of Interpretability Tools for Machine Learning},
  author={Kaur, Harmanpreet and Nori, Harsha and Jenkins, Samuel and Caruana, Rich and Wallach, Hanna and Wortman Vaughan, Jennifer},
  booktitle={Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems},
  pages={1--14},
  year={2020}
}
    </pre>
    <a href="https://www.microsoft.com/en-us/research/publication/interpreting-interpretability-understanding-data-scientists-use-of-interpretability-tools-for-machine-learning/">Paper link</a>
  </details>

  <details>
    <summary>
      <em>"How Interpretable and Trustworthy are GAMs?" (C. Chang, S. Tan, B. Lengerich, A. Goldenberg, R. Caruana 2020)</em>
    </summary>
    <br/>
    <pre>
@article{chang2020interpretable,
  title={How Interpretable and Trustworthy are GAMs?},
  author={Chang, Chun-Hao and Tan, Sarah and Lengerich, Ben and Goldenberg, Anna and Caruana, Rich},
  journal={arXiv preprint arXiv:2006.06466},
  year={2020}
}
    </pre>
    <a href="https://arxiv.org/pdf/2006.06466.pdf">Paper link</a>
  </details>

  <hr/>
</details>

<details>
  <summary><strong>Differential Privacy</strong></summary>
  <hr/>

  <details>
    <summary>
      <em>"Accuracy, Interpretability, and Differential Privacy via Explainable Boosting" (H. Nori, R. Caruana, Z. Bu, J. Shen, J. Kulkarni 2021)</em>
    </summary>
    <br/>
    <pre>
@inproceedings{pmlr-v139-nori21a,
  title = 	 {Accuracy, Interpretability, and Differential Privacy via Explainable Boosting},
  author =       {Nori, Harsha and Caruana, Rich and Bu, Zhiqi and Shen, Judy Hanwen and Kulkarni, Janardhan},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {8227--8237},
  year = 	 {2021},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR}
}
    </pre>
    <a href="https://proceedings.mlr.press/v139/nori21a/nori21a.pdf">Paper link</a>
  </details>

  <hr/>
</details>

<details>
  <summary><strong>LIME</strong></summary>
  <hr/>

  <details>
    <summary>
      <em>"Why should i trust you?: Explaining the predictions of any classifier" (M. T. Ribeiro, S. Singh, and C. Guestrin 2016)</em>
    </summary>
    <br/>
    <pre>
@inproceedings{ribeiro2016should,
  title={Why should i trust you?: Explaining the predictions of any classifier},
  author={Ribeiro, Marco Tulio and Singh, Sameer and Guestrin, Carlos},
  booktitle={Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining},
  pages={1135--1144},
  year={2016},
  organization={ACM}
}
    </pre>
    <a href="https://arxiv.org/pdf/1602.04938.pdf">Paper link</a>
  </details>

  <hr/>
</details>

<details>
  <summary><strong>SHAP</strong></summary>
  <hr/>

  <details>
    <summary>
      <em>"A Unified Approach to Interpreting Model Predictions" (S. M. Lundberg and S.-I. Lee 2017)</em>
    </summary>
    <br/>
    <pre>
@incollection{NIPS2017_7062,
 title = {A Unified Approach to Interpreting Model Predictions},
 author = {Lundberg, Scott M and Lee, Su-In},
 booktitle = {Advances in Neural Information Processing Systems 30},
 editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
 pages = {4765--4774},
 year = {2017},
 publisher = {Curran Associates, Inc.},
 url = {https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf}
}
    </pre>
    <a href="https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf">Paper link</a>
  </details>

  <details>
    <summary>
      <em>"Consistent individualized feature attribution for tree ensembles" (Lundberg, Scott M and Erion, Gabriel G and Lee, Su-In 2018)</em>
    </summary>
    <br/>
    <pre>
@article{lundberg2018consistent,
  title={Consistent individualized feature attribution for tree ensembles},
  author={Lundberg, Scott M and Erion, Gabriel G and Lee, Su-In},
  journal={arXiv preprint arXiv:1802.03888},
  year={2018}
}
    </pre>
    <a href="https://arxiv.org/pdf/1802.03888">Paper link</a>
  </details>

  <details>
    <summary>
      <em>"Explainable machine-learning predictions for the prevention of hypoxaemia during surgery" (S. M. Lundberg et al. 2018)</em>
    </summary>
    <br/>
    <pre>
@article{lundberg2018explainable,
  title={Explainable machine-learning predictions for the prevention of hypoxaemia during surgery},
  author={Lundberg, Scott M and Nair, Bala and Vavilala, Monica S and Horibe, Mayumi and Eisses, Michael J and Adams, Trevor and Liston, David E and Low, Daniel King-Wai and Newman, Shu-Fang and Kim, Jerry and others},
  journal={Nature Biomedical Engineering},
  volume={2},
  number={10},
  pages={749},
  year={2018},
  publisher={Nature Publishing Group}
}
    </pre>
    <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6467492/pdf/nihms-1505578.pdf">Paper link</a>
  </details>

  <hr/>
</details>

<details>
  <summary><strong>Sensitivity Analysis</strong></summary>
  <hr/>

  <details>
    <summary>
      <em>"SALib: An open-source Python library for Sensitivity Analysis" (J. D. Herman and W. Usher 2017)</em>
    </summary>
    <br/>
    <pre>
@article{herman2017salib,
  title={SALib: An open-source Python library for Sensitivity Analysis.},
  author={Herman, Jonathan D and Usher, Will},
  journal={J. Open Source Software},
  volume={2},
  number={9},
  pages={97},
  year={2017}
}
    </pre>
    <a href="https://www.researchgate.net/profile/Will_Usher/publication/312204236_SALib_An_open-source_Python_library_for_Sensitivity_Analysis/links/5ac732d64585151e80a39547/SALib-An-open-source-Python-library-for-Sensitivity-Analysis.pdf?origin=publication_detail">Paper link</a>
  </details>

  <details>
    <summary>
      <em>"Factorial sampling plans for preliminary computational experiments" (M. D. Morris 1991)</em>
    </summary>
    <br/>
    <pre>
@article{morris1991factorial,
  title={},
  author={Morris, Max D},
  journal={Technometrics},
  volume={33},
  number={2},
  pages={161--174},
  year={1991},
  publisher={Taylor \& Francis Group}
}
    </pre>
    <a href="https://abe.ufl.edu/Faculty/jjones/ABE_5646/2010/Morris.1991%20SA%20paper.pdf">Paper link</a>
  </details>

  <hr/>
</details>

<details>
  <summary><strong>Partial Dependence</strong></summary>
  <hr/>

  <details>
    <summary>
      <em>"Greedy function approximation: a gradient boosting machine" (J. H. Friedman 2001)</em>
    </summary>
    <br/>
    <pre>
@article{friedman2001greedy,
  title={Greedy function approximation: a gradient boosting machine},
  author={Friedman, Jerome H},
  journal={Annals of statistics},
  pages={1189--1232},
  year={2001},
  publisher={JSTOR}
}
    </pre>
    <a href="https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451">Paper link</a>
  </details>

  <hr/>
</details>

<details>
  <summary><strong>Open Source Software</strong></summary>
  <hr/>

  <details>
    <summary>
      <em>"Scikit-learn: Machine learning in Python" (F. Pedregosa et al. 2011)</em>
    </summary>
    <br/>
    <pre>
@article{pedregosa2011scikit,
  title={Scikit-learn: Machine learning in Python},
  author={Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and others},
  journal={Journal of machine learning research},
  volume={12},
  number={Oct},
  pages={2825--2830},
  year={2011}
}
    </pre>
    <a href="https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf">Paper link</a>
  </details>

  <details>
    <summary>
      <em>"Collaborative data science" (Plotly Technologies Inc. 2015)</em>
    </summary>
    <br/>
    <pre>
@online{plotly, 
  author = {Plotly Technologies Inc.}, 
  title = {Collaborative data science}, 
  publisher = {Plotly Technologies Inc.}, 
  address = {Montreal, QC}, 
  year = {2015}, 
  url = {https://plot.ly}
}
    </pre>
    <a href="https://plot.ly">Link</a>
  </details>
  
  <details>
    <summary>
      <em>"Joblib: running python function as pipeline jobs" (G. Varoquaux and O. Grisel 2009)</em>
    </summary>
    <br/>
    <pre>
@article{varoquaux2009joblib,
  title={Joblib: running python function as pipeline jobs},
  author={Varoquaux, Ga{\"e}l and Grisel, O},
  journal={packages. python. org/joblib},
  year={2009}
}
    </pre>
    <a href="https://joblib.readthedocs.io/en/latest/">Link</a>
  </details>
  
  <hr/>
</details>

# Videos

- [The Science Behind InterpretML: Explainable Boosting Machine](https://www.youtube.com/watch?v=MREiHgHgl0k)
- [How to Explain Models with InterpretML Deep Dive](https://www.youtube.com/watch?v=WwBeKMQ0-I8)
- [Black-Box and Glass-Box Explanation in Machine Learning](https://youtu.be/7uzNKY8pEhQ)
- [Explainable AI explained!  By-design interpretable models with Microsofts InterpretML](https://www.youtube.com/watch?v=qPn9m30ojfc)
- [Interpreting Machine Learning Models with InterpretML](https://www.youtube.com/watch?v=ERNuFfsknhk)
- [Machine Learning Model Interpretability using AzureML & InterpretML (Explainable Boosting Machine)](https://www.youtube.com/watch?v=0ocVtXU8o1I)
- [A Case Study of Using Explainable Boosting Machines](https://uncch.hosted.panopto.com/Panopto/Pages/Embed.aspx?id=063d6839-e8db-40e0-8df4-b0fc012e709b&start=0)
- [From SHAP to EBM: Explain your Gradient Boosting Models in Python](https://www.youtube.com/watch?v=hnZjw77-1rE)

# External links

- [Machine Learning Interpretability in Banking: Why It Matters and How Explainable Boosting Machines Can Help](https://www.prometeia.com/en/trending-topics-article/machine-learning-interpretability-in-banking-why-it-matters-and-how-explainable-boosting-machines-can-help)
- [Interpretable Machine Learning – Increase Trust and Eliminate Bias](https://ficonsulting.com/insight-post/interpretable-machine-learning-increase-trust-and-eliminate-bias/)
- [Enhancing Trust in Credit Risk Models: A Comparative Analysis of EBMs and GBMs](https://2os.medium.com/enhancing-trust-in-credit-risk-models-a-comparative-analysis-of-ebms-and-gbms-25e02810300f)
- [Explainable AI: unlocking value in FEC operations](https://analytiqal.nl/2024/01/22/fec-value-from-explainable-ai/)
- [Interpretable or Accurate? Why Not Both?](https://towardsdatascience.com/interpretable-or-accurate-why-not-both-4d9c73512192)
- [The Explainable Boosting Machine. As accurate as gradient boosting, as interpretable as linear regression.](https://towardsdatascience.com/the-explainable-boosting-machine-f24152509ebb)
- [Exploring explainable boosting machines](https://leinadj.github.io/2023/04/09/Exploring-Explainable-Boosting-Machines.html)
- [Performance And Explainability With EBM](https://blog.oakbits.com/ebm-algorithm.html)
- [InterpretML: Another Way to Explain Your Model](https://towardsdatascience.com/interpretml-another-way-to-explain-your-model-b7faf0a384f8)
- [A gentle introduction to GA2Ms, a white box model](https://www.fiddler.ai/blog/a-gentle-introduction-to-ga2ms-a-white-box-model)
- [Explaining Non-Parametric Additive Models](https://gablabc.github.io/posts/2025/01/NonParametricAdditive/)
- [Model Interpretation with Microsoft’s Interpret ML](https://medium.com/@sand.mayur/model-interpretation-with-microsofts-interpret-ml-85aa0ad697ae)
- [Explaining Model Pipelines With InterpretML](https://medium.com/@mariusvadeika/explaining-model-pipelines-with-interpretml-a9214f75400b)
- [Explain Your Model with Microsoft’s InterpretML](https://medium.com/@Dataman.ai/explain-your-model-with-microsofts-interpretml-5daab1d693b4)
- [On Model Explainability: From LIME, SHAP, to Explainable Boosting](https://everdark.github.io/k9/notebooks/ml/model_explain/model_explain.nb.html)
- [Dealing with Imbalanced Data (Mortgage loans defaults)](https://mikewlange.github.io/ImbalancedData-/index.html)
- [The right way to compute your Shapley Values](https://towardsdatascience.com/the-right-way-to-compute-your-shapley-values-cfea30509254)
- [The Art of Sprezzatura for Machine Learning](https://towardsdatascience.com/the-art-of-sprezzatura-for-machine-learning-e2494c0db727)
- [Mixing Art into the Science of Model Explainability](https://towardsdatascience.com/mixing-art-into-the-science-of-model-explainability-312b8216fa95)
- [Automatic Piecewise Linear Regression](https://link.springer.com/article/10.1007/s00180-024-01475-4)
- [MCTS EDA which makes sense](https://www.kaggle.com/code/ambrosm/mcts-eda-which-makes-sense/notebook)
- [Explainable Boosting machines for Tabular data](https://www.kaggle.com/code/parulpandey/explainable-boosting-machines-for-tabular-data)

# Papers that use or compare EBMs

- [Challenging the Performance-Interpretability Trade-off: An Evaluation of Interpretable Machine Learning Models](https://arxiv.org/pdf/2409.14429)
- [GAMFORMER: In-context Learning for Generalized Additive Models](https://arxiv.org/pdf/2410.04560v1)
- [Glass Box Machine Learning and Corporate Bond Returns](https://download.ssrn.com/2024/12/7/5047456.pdf?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQC4Tahz9gLK1PnaT2OcGVvU95LIXVjM6MGGLiR8muLBHwIgDhBNcW1HFouaDPID%2FgMGtfAOIqCtr0JTXSsBYYGjXZEqvQUIYhAEGgwzMDg0NzUzMDEyNTciDBSJqXVnefWlpTvEvSqaBfTjYUxrqKGMDY1QImthn4LR848sVKt0vRNJSuhmBUct5KZ%2FYHehm4HVsRgxd%2FYezoCGxoo%2Bee1rhSCW7WVwdPrNzvAb34a410A6DkywgnsGsKhvMltoeYudsXrL2SlqY6fP5z8mmzELDBjHhNRbpjaPx%2BmHQiv8PrE6bqrQz%2Fe18Aj9JVsAlUAiJ0s9AiK5kaqJU6yZJJvaFh4AWHd8IsOj0QBU%2BHoTYj5ff2XzM9PtWUQbZccke066NcJEuUSE7fj5OREKYi%2BCXG6zNWu8y4BubfSzjL1pLRDuBB%2Fc6zQNWu%2Bz9sibZRzBvx1mmhTrL2ffhWCeTlCaSj1EsM3VuyNz3d8z2MFTpxn4hBqcR3lk0daL4qllASz3UMp%2FPwteZszzpE9moZwEwJWmR4TrH3KFF7x2bZaQPLvgt%2BqnstHyJXOgaEmXDX0yEyGQyDQh9RpD4n%2FlBTSHsXme0OfjEt5AscRnBqRpsX3ZG9Bx0cK5ibthJ5M%2FwsExm76cF8tPdyOCqozaF%2B9l1sJsWK1h4jHYrbdyZZZtKj786Ed2CAOZ0M%2FzY1hyHHAO47jOlNs9Ju2qLhsdXmcbFRiBfA2IN4UxClTyrAaYA2fQhpFHpRdM5CMk8%2Fe4h8Mt1PVO7K95BihB2P5O%2BCiWWzrlwAVKa5KKipxmRTDTjl2VIKFkwLeCwPHJFpQCL1ZknhGyIiD9hLpS%2BykAXnHyN4U7dN6rtOf8FGqDO0QUA2ZvTaT6DXrUps1Wf2iNqk%2Facl3RyB3nNwRw4igeUx9UysRRFsz2Rs93kZJq17yGiCLP38xF%2FniKj8doporrNbBjkMhBpAXKyqDpQ7JQ%2Bwu8GPAWdPiD%2FumnjvuuoG0UkmGX81izy5zCREKOraIuDzqOMjD6%2BIu7BjqxAYNSOCdUDYw30aHTCSDBoj%2B4LzQk2ZMdumvEyvNKv%2ButPmI1J6LsIwV8s1q7dgqpe2%2FA%2BLX4ZbWaT0OPR3idfz7ADSNZp6Ykfinfj6FSf673Kj%2FOm4etpiZxcMy8MDD65CHXCDy0%2FhMz9JAwmDyD2Dmb%2Fhvta226gohBnpuBA8KCoP946GdYEzyaXGvByrO4BOmvB%2BlRifauewFHndym7seWzQtJ%2B514aguu1JT%2F7XJVUw%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241218T174139Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAUPUUPRWE45GYNSPF%2F20241218%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=2c2b0fd799efafee1598fbc8ba7dd050451f67845f5cfa3b01a5b3ed0db7db1c&abstractId=5047456)
- [Data Science with LLMs and Interpretable Models](https://arxiv.org/pdf/2402.14474v1.pdf)
- [DimVis: Interpreting Visual Clusters in Dimensionality Reduction With Explainable Boosting Machine](https://arxiv.org/pdf/2402.06885.pdf)
- [Distill knowledge of additive tree models into generalized linear models](https://detralytics.com/wp-content/uploads/2023/10/Detra-Note_Additive-tree-ensembles.pdf)
- [Explainable Boosting Machines with Sparsity - Maintaining Explainability in High-Dimensional Settings](https://arxiv.org/abs/2311.07452)
- [Cost of Explainability in AI: An Example with Credit Scoring Models](https://link.springer.com/chapter/10.1007/978-3-031-44064-9_26)
- [Interpretable Machine Learning Leverages Proteomics to Improve Cardiovascular Disease Risk Prediction and Biomarker Identification](https://www.medrxiv.org/content/10.1101/2024.01.12.24301213v1.full.pdf)
- [Interpretable Additive Tabular Transformer Networks](https://openreview.net/pdf/d2f0db2646418b24bb322fc1f4082fd9e65409c2.pdf)
- [Signature Informed Sampling for Transcriptomic Data](https://www.biorxiv.org/content/biorxiv/early/2023/10/31/2023.10.26.564263.full.pdf)
- [Interpretable Survival Analysis for Heart Failure Risk Prediction](https://arxiv.org/pdf/2310.15472.pdf)
- [LLMs Understand Glass-Box Models, Discover Surprises, and Suggest Repairs](https://arxiv.org/pdf/2308.01157.pdf)
- [Model Interpretability in Credit Insurance](http://hdl.handle.net/10400.5/27507)
- [Federated Boosted Decision Trees with Differential Privacy](https://arxiv.org/pdf/2210.02910.pdf)
- [Differentially private and explainable boosting machine with enhanced utility](https://www.sciencedirect.com/science/article/abs/pii/S0925231224011950?via%3Dihub#preview-section-abstract)
- [Balancing Explainability and Privacy in Bank Failure Prediction: A Differentially Private Glass-Box Approach](https://ieeexplore.ieee.org/abstract/document/10818483)
- [GAM(E) CHANGER OR NOT? AN EVALUATION OF INTERPRETABLE MACHINE LEARNING MODELS](https://arxiv.org/pdf/2204.09123.pdf)
- [GAM Coach: Towards Interactive and User-centered Algorithmic Recourse](https://arxiv.org/pdf/2302.14165.pdf)
- [Missing Values and Imputation in Healthcare Data: Can Interpretable Machine Learning Help?](https://arxiv.org/pdf/2304.11749v1.pdf)
- [Practice and Challenges in Building a Universal Search Quality Metric](https://www.researchgate.net/profile/Nuo-Chen-38/publication/370126720_Practice_and_Challenges_in_Building_a_Universal_Search_Quality_Metric/links/6440a0f239aa471a524cb77d/Practice-and-Challenges-in-Building-a-Universal-Search-Quality-Metric.pdf?origin=publication_detail)
- [Explaining Phishing Attacks: An XAI Approach to Enhance User Awareness and Trust](https://www.researchgate.net/profile/Giuseppe-Desolda/publication/370003878_Explaining_Phishing_Attacks_An_XAI_Approach_to_Enhance_User_Awareness_and_Trust/links/643922a8e881690c4bd50ced/Explaining-Phishing-Attacks-An-XAI-Approach-to-Enhance-User-Awareness-and-Trust.pdf)
- [Revealing the Galaxy-Halo Connection Through Machine Learning](https://arxiv.org/pdf/2204.10332.pdf)
- [How the Galaxy–Halo Connection Depends on Large-Scale Environment](https://arxiv.org/pdf/2402.07995.pdf)
- [Explainable Artificial Intelligence for COVID-19 Diagnosis Through Blood Test Variables](https://link.springer.com/content/pdf/10.1007/s40313-021-00858-y.pdf)
- [A diagnostic support system based on interpretable machine learning and oscillometry for accurate diagnosis of respiratory dysfunction in silicosis](https://www.biorxiv.org/content/10.1101/2025.01.08.632001v1.full.pdf)
- [Using Explainable Boosting Machines (EBMs) to Detect Common Flaws in Data](https://link.springer.com/chapter/10.1007/978-3-030-93736-2_40)
- [Differentially Private Gradient Boosting on Linear Learners for Tabular Data Analysis](https://assets.amazon.science/fa/3a/a62ba73f4bbda1d880b678c39193/differentially-private-gradient-boosting-on-linear-learners-for-tabular-data-analysis.pdf)
- [Differentially private and explainable boosting machine with enhanced utility](https://www.sciencedirect.com/science/article/abs/pii/S0925231224011950)
- [Concrete compressive strength prediction using an explainable boosting machine model](https://www.sciencedirect.com/science/article/pii/S2214509523000244/pdfft?md5=171c275b6bcae8897cef03d931e908e2&pid=1-s2.0-S2214509523000244-main.pdf)
- [Proxy endpoints - bridging clinical trials and real world data](https://pdf.sciencedirectassets.com/272371/1-s2.0-S1532046424X00064/1-s2.0-S1532046424001412/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEIn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIBYgAN6aOVrDnvQ1932tPndUyJ0Dm1nHdMVLiekPVduQAiAzbYe7W%2Bd6Dj8ee42ZeZnQxJwEjEjuGdiUEPx0a2G43SqyBQgSEAUaDDA1OTAwMzU0Njg2NSIMyMkCUNFeDTCUCppMKo8FiVShykb8phR%2F8aWUGE9gfnE5y7X3Jj1ZA2CVldH13T67s536bdTBhjIMF18rV0YP9iMi6B5aGr%2F286ovIJl332fxZ6iQNBIOPTm8kXQDUqvZbknYldiZqUPs69kuC%2FcKnJd1BWnv2SEZwbRuX94rWnRDPDaSoJx%2FVS6o4qsbFjp9%2BMYZr%2BvJzWHKrXAI4W%2Fh9%2BsIa0yvlac3IMWzAeD23HzDNmF0nqjJ6BSZzmDNW4HRIGBTrTUTO40TzQzhaOY7wyGA0Zv8SpWIULI%2FrY8z8EOX%2FU6OhqgyIMKv%2FSx3rUpMi5CrC1WcpnL97j%2FDAijNi4vMfG1b%2BBQIFRu2EmUky76k4w3FYxkCpYj4n4mk9H%2B%2Bc9C%2BdjKjUiayi%2FisIZUD7ISNhQ9oov0kXI1IVTCGKKQC9jqHOvdiA8YbVuMdEzy1Lkx%2B1kiEo79qvSlpTe2BtWAOm2Iequ01XoaMv%2FQb4ajhWKKSkTafzDAxc58aayP1YH49UzQ68Me7ecdHpx3JUHyYnxJGQ82wRpPkfZJA5wCmOUVI%2FBLuwFJyczG0LpALN5IpIqZz%2B8DvDR0xjRoN49dVwhrTSQ9BesvXbi2LKVm1ptacaaKqyx0PwLjQYKOd%2BPI3zCvRxEiM3IKSNFRLsUTyPNEE4E8pMFNxfyEX59yvTQrHwM62P7hvxHs%2BY6CxUGZTKBQwDAgxttJmiO%2BvjCRbTBXZg1WrQdXCkxntBXb15Mnqxo4lyPzUUkLdLAFK%2BLSwzBIcvSw2qG81Y8qhWmBgBT9vfAoSrjxsILFrB3nnz7u9XNNpRxb5Z9NuNG92%2Fpd%2F%2F5VespMY8Q0iwsNqazZ4M4H8UB34JgtrUEY27WrIsDWzLR%2FAYAxU%2BZHrFzCrsae5BjqyASqDBsNqjEkho%2FbuQDT%2F0vGx%2BgAqrksvVX0GrzNgvqnuPyvw6%2F%2B40ZJP5EA4axfltOYb2tNjd18Ngy2A3cd6J57v1G7wYyuSFIUfHGN5LA8BXK7p0x1mNcwN3pKHtAf260gjpsWMG7anvpK%2F3YupTz498C1lAmurJD%2BLN41lq05wBr403cchE41yzqAKHVKVpNq9s6oGHJmq0KJRvk%2FfjZr8oLhod5gtrwLKvLGqULf50L0%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241105T092058Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYUKUJCDYI%2F20241105%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=15f8e40964e2c750ae15e43aa8e7f7c76eef6a76b792e41434d14bed42b31432&hash=d4a3e49b29443e5eea9e5a44c0dc11b3f30b21addbe6d6d20d523c68db23cd23&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1532046424001412&tid=spdf-4fbabbd8-becb-4526-98d3-c7517914e457&sid=8ab2a095350fc74edc4b8765ecd8c0260edcgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f165f0b050207505b0151&rr=8ddbc55a0d60a380&cc=us)
- [Machine Learning Model Reveals Determinators for Admission to Acute Mental Health Wards From Emergency Department Presentations](https://onlinelibrary.wiley.com/doi/epdf/10.1111/inm.13402)
- [Towards Cleaner Cities: Estimating Vehicle-Induced PM2.5 with Hybrid EBM-CMA-ES Modeling](https://www.mdpi.com/2305-6304/12/11/827)
- [Predicting Robotic Hysterectomy Incision Time: Optimizing Surgical Scheduling with Machine Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741200/pdf/e2024.00040.pdf)
- [Using machine learning to assist decision making in the assessment of mental health patients presenting to emergency departments](https://journals.sagepub.com/doi/full/10.1177/20552076241287364)
- [Proposing an inherently interpretable machine learning model for shear strength prediction of reinforced concrete beams with stirrups](https://pdf.sciencedirectassets.com/287527/1-s2.0-S2214509523X00035/1-s2.0-S2214509524005011/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjECUaCXVzLWVhc3QtMSJGMEQCIB0r0KsYBZufOjbCVtUtozwn1QKMdLt2tbbfhuJKjWlXAiB5Dfr7p0yyj%2FSfypTLmjPL8WbjGAB3tRACFjyyqQbbfiq8BQiu%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMqBpZ2HmN91c%2BJPqpKpAFZtvqQjCScZa4FN%2FeubsPzOk5c%2B58LliO4Zr%2Bn1pm3vtW4I9I1vA29pkhT5was1N3ccPPIm2jNLwJ%2FHiZej7A2SmFv13Ro3sTvhqG%2F6A9Xx70Nx9jOlDPJUmCypKadKp0FGfuhZQuxeN0b%2F1QUUQZG4RpxC%2FXorRRHmb%2FrXcOWBwu4PmLZAkWmTKpncjDI7oj8eh8yBe6%2FA3JkJ14ZyBgR7JnPzR2ZqMdIhvlKoyMn6EnL1Azq2y3qwEMdzSCvz3wH3sT4pClc2vPs6ruQS4CdT3E7BHrf42Q0VnUXWjuy7gt9iRr0vaWR3tD%2FxyrrEKw7XuMHO9L4rQ4Pfn1dhGZ2J8H5ocwJGSh13U5fY6noyaTNViqvHx1oHNMWL03QpkJxmUxYquBWepcDjxEc32V6eGF7Ecm8Vij3s20wdRNcHqxGFKlUCgph48CKUA79iwSGQCkWQh7bq%2FTtowTbSPud7l8xeG1MvfIVy%2B6yzrjqygvPBQs3qkvdoWUrKXe57bhr2jEkKlSdYyp2TJMD6yoYRdTPyFx5xb0KgIt6KQTPmfbqYXkd3FFz3uc0HmWC5NQz6qP9UzNcBhcK8dXo3Dw042pl0HLO1njFaa%2BBfbT89VUVUIqjrAcmHweIl1v7Eyldzr%2BGBXIlsxPO3gPzyPLF2LTggc6dA%2Bswxmgmkv%2B7n5pU5%2F5sxvEhemb%2Fqu%2B8d47O%2Bn6RH8fL4eLGGL2d0dvFvyE7gEwt%2BaU9HsIN0IHqyH5VmaTF5zaKy%2Fn%2BhkF8yGpe5Hq5yNOUGrfQgfyFn4Kqd%2FTVajxIFzk8DEY%2F%2FFtyGJ%2B8BrHV4P%2FYs8R4XcBzPQtyrTuUC1CGmF01Tc2gnnEo4pVPaIjfBk9B%2BXVMc3Mu4Ywy4L%2BsgY6sgFK3hFIXjIfoVjqrIlBvsGYaFiZB1bVKBVy3DRiBgozzYmIVhipN%2FS%2BPok1oETqvYVvLqEVkGcb5W7nUIK16lFgjwDq6ePuxdqSafgOw5jVQroNsDCPRz8B%2F4fg7kv6gs4R9SX7gCaQ2V7L6NxqJDUUqsCMtIYq05Qx43dGByqLoVEz9USpRBmTLQwpGvOmUaGNNwTsCwmt5gRP8UX3CnkwI%2FydxmhrXLEdaUIFVwJbIor9&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240604T221639Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY4E2DAHPF%2F20240604%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=eece32da8855b55208baecc0ce041e79aa03be1c292b58c67ce0215de36cbdb4&hash=46dd1da122f4cea242c6444a811fb16dde5cb8465e88552ac3eaeee97b975e9b&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S2214509524005011&tid=spdf-45c1c4d1-dd97-4c0d-a04f-c30843a79e78&sid=1fea53ed2d5cf1443e4a7c4-33f4bf6475e1gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f155c5f060d565b01055d&rr=88eb49dd2a5f7688&cc=us)
- [A hybrid machine learning approach for predicting fiber-reinforced polymer-concrete interface bond strength](https://download.ssrn.com/eaai/e646e179-ec4a-4987-80b5-8d6bbf43ceda-meca.pdf?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBMaCXVzLWVhc3QtMSJHMEUCIFVH%2Ba5TT2NOEqgCl7GMhXBXBZWE9VzzcRFT6kYXzdxYAiEA4yvXsrzNQnNq%2BkJRB0rw1d2p35f418pIO%2FT3PHKoZ%2BoqxgUI%2B%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgwzMDg0NzUzMDEyNTciDCD0kCrKAqamcwb9LCqaBb4zlqjDhNBhf%2Frbe%2FX3lzSjvS58HiJQtbOHmzaM7putg93e7Wk8nPesoiupTH8uB5ejDC7stGJElRZp5ulT5M6CokoMu82ERn15kMpkgptj3MVEmsY9VTCP%2BCbROJ6v4YcAttOOAEzOc2M6li6o0w4IsF8DNXEIJr%2FJvjB3IDYPkrmpIiHl25h3AzfxPuOF01E2rgucLnY0xTyKGnPBBDZ%2FPtcuqlk2NKun3Q9HbcKj8EPJP%2FPupMW3IQvMnhcdJqqLHXs6wL1P42NTw5vtZO2W5WiEC1CNGDFUTSFRdb9hjhpH4JsYl8X%2BSFT6mZ31K2HTWeuigs5nXp1JN8r8r4O021yiVxHAJ6Chnddr0Z19iM5yOZA4H1EhO1rxxL0VF%2F%2F8Ac3GxuEfkBiug5wuL7aNlBNX6720pYfHH%2FgyrqdU5KSDIp8VYw3KgEij0LkizBHQIoolC48VAEMNc%2F8iWOdZpAVYprhEbABbff8%2BW6c4y1N9vmLTkjZkJtZODpzpQVjrHkL9hAOvmXZocEEN6maRoVJx3DlcTHrfQr8%2BQnPQnmajb5x0FHo44xxBIUt7UB4FOc6beDprle%2F7BO2SNEPLw6rJ9e3WJeVaYch46iqk2tiWFroNHDXlQ73CbzV59AEVtLAR29eIf7uyz%2BU0fOAXG5oAsJyB7YXUjH%2Bh79sxJgBq3%2FoqkEja06CFPRhWeqxixc8y9bEU%2FvvjhfbcWcxGY%2Be%2FwnXbemUbSyr26Y5xvADyicKIMexZNjeHBJ9MKMifQ9oh%2FjmudjxtMLbTpA6EAxMelLjhWcoURF0XeTttMEzEuTjO1OXUwMeXSPZ9roJqH3DB4PHi%2B8UIUG1JoVocv7wDu5ZVlMzgmDr0ti1BShKr9szxagq34jCEkJe8BjqxAbm7bsef33J3AImECx0GZeL0R2tFJZ7ctogL261zP7RqJ4T71rDMbpyfX6HfGuNEbWVROKHUexpuH8FZBodmn%2FjDjZSviK1oxQ1L5TDA2rwMsodnThreIad8vSXqxAzx9qng%2BeN2llXkNdIB7WEnkttzcJ24pZqwYnPI%2FsOznTq%2BDJ88mdNPtzph%2FGdVQcR99tV3waapotTEnUjjoqTTSh9aMgi1jIYMGMrJj6Jb4N%2FhWA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250114T024208Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAUPUUPRWEXKDDLJZE%2F20250114%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=11c6f325f84736d5324ab155663c94231696de52be8910a03bb5e9c18f0d1689&abstractId=5055231)
- [Using explainable machine learning and fitbit data to investigate predictors of adolescent obesity](https://www.nature.com/articles/s41598-024-60811-2)
- [Interpretable Predictive Value of Including HDL-2b and HDL-3 in an Explainable Boosting Machine Model for Multiclass Classification of Coronary Artery Stenosis Severity in Acute Myocardial Infarction Patients](https://watermark.silverchair.com/ztae100.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA2owggNmBgkqhkiG9w0BBwagggNXMIIDUwIBADCCA0wGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMnDqoUBnqG9Zyr0dAAgEQgIIDHT2M3owEzTRAV3KZzrOpzyqOYgClio-CQrzB5731fvsEe9ZWO_QfqQAKdaPyyOsEKjacd25hWs-_OvgXCqc36R4yFWu46PFOCApII2s3hbHYI1XEQozWfdyosgaQf_e7_5RIqIfwTEHt19LoYZuaDYjCqq2vmWOMZb6dNI6mz-h3Zd6BgbyYAFgRHiJfU94NU0Crf_AbbTx2jW3HqMBLYPn-ysUiyQYILNmqlKAAlw81ZjBwzusaQFsiJMCxwGyFHks7nwtnUQ8J5PU5Jelp8_fQ8x5_dlZvzvdkI9MR87zUkk4hm2XL0uyfvH92-7VV_2gMe-rU3aJZhbHJu2hENPDh_OmoDe7SOC-5EwPsgIDoDr_dgSgyhBMIbOk_TrSM4oEN6dbtvfLSDXQUWDV4semLuPjqz7WyiQz4PPt1mXuaf12X5xyVsf1Mms4UpGAKLyoCdJ-zDJ9csOPCefIsV2Bzs-KzaD63HWFLJuCU0hWIaK0QOcJATnpQb1PhFiAF6YZ_cCYTxkuAcrQyHS-WCEefNy8hB8PQXhNljtw0J499qdnLcNOM1gAQ3-o21KaTrEFs-DyvZwWmaGn8Zw1bK1CG8yVxWOh6_wjJpGjMMenstzrKFcLbJADs1yf3PuNGZds0g-Qf4NDcgsturcr0V1nLHVRFazWZhUKSeRnLjPzA5i3lVKnmwKjKa_50i0LMSIXNFS-dmvHs-qVUb8FO0_aKZ6egckXkoGG8w3Jox4MhhY2-B28Z0wbJOj8_DojCCtAmAPC0T5emRsuk1rkuRXIoMtFDWN0l7fr7RVkuy1TEd3mpa5UuU7Qo-wu_yqi6ibwLupjGeVN__7SeteoBSh8yFJgYN4BEiYmdkEX7DgKaMC90h5GakNJ7zeAPR9PFnQVRORoof04qMWK4aGod2igso1-qsCup-kVWmPy8zrQKlqxE4OCeqUpKQgZMUUAlFu643iuRnQuLnahXhui45TY8lS56XGCLqkwSG594lMoAXAYZ9tVFM4fAVwQJ3EWkJfHRRCWWGZfLwBPsdUnNEziGg4QIdrKhe-Fu7nLF)
- [Estimate Deformation Capacity of Non-Ductile RC Shear Walls Using Explainable Boosting Machine](https://arxiv.org/pdf/2301.04652.pdf)
- [Introducing the Rank-Biased Overlap as Similarity Measure for Feature Importance in Explainable Machine Learning: A Case Study on Parkinson’s Disease](https://www.researchgate.net/publication/362808061_Introducing_the_Rank-Biased_Overlap_as_Similarity_Measure_for_Feature_Importance_in_Explainable_Machine_Learning_A_Case_Study_on_Parkinson's_Disease)
- [Targeting resources efficiently and justifiably by combining causal machine learning and theory](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9768181/pdf/frai-05-1015604.pdf)
- [Extractive Text Summarization Using Generalized Additive Models with Interactions for Sentence Selection](https://arxiv.org/pdf/2212.10707.pdf)
- [Death by Round Numbers: Glass-Box Machine Learning Uncovers Biases in Medical Practice](https://www.medrxiv.org/content/medrxiv/early/2022/11/28/2022.04.30.22274520.full.pdf)
- [Post-Hoc Interpretation of Transformer Hyperparameters with Explainable Boosting Machines](https://www.cs.jhu.edu/~xzhan138/papers/BLACK2022.pdf)
- [Interpretable machine learning for predicting pathologic complete response in patients treated with chemoradiation therapy for rectal adenocarcinoma](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9771385/pdf/frai-05-1059033.pdf)
- [Exploring the Balance between Interpretability and Performance with carefully designed Constrainable Neural Additive Models](https://www.sciencedirect.com/science/article/pii/S1566253523001987)
- [Estimating Discontinuous Time-Varying Risk Factors and Treatment Benefits for COVID-19 with Interpretable ML](https://arxiv.org/pdf/2211.08991.pdf)
- [StratoMod: Predicting sequencing and variant calling errors with interpretable machine learning](https://www.biorxiv.org/content/10.1101/2023.01.20.524401v1.full.pdf)
- [Interpretable machine learning algorithms to predict leaf senescence date of deciduous trees](https://pdf.sciencedirectassets.com/271723/1-s2.0-S0168192323X00112/1-s2.0-S0168192323003143/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIArPrCug2%2BpvA%2F87dfMYdbINsntWDDgNHeCOn72Yfad3AiBHzR9BvMkRvZrjQZ1DoY1YMkD6VsQw45zqo5ykkClnHSq8BQiL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMdV4IhgM83azwHKyjKpAFFMABPkhGjjH1i3y26weF5LN6ZuxfgcDlklmnpEZDoEntreay08vlEU7%2F3CLeNSqYgaq5txCiVztJDv2TBcxDUt0PP4faNrHUWIQdfDksvDs3EE7VEupaqhVjMNi0%2F%2ByLRw2OzjMPpz7H5sd3i4%2F%2FK2%2FJlpAWHlr4RFJ9BXMMPbLDEqhjIJIl5ZzaeLeeijXKrTtvJ6iYwTic%2FHJ23m7Fdnkh94HKkFTOWeglJzGT7FSc5Wnc7DgExrL7EBLvu9YVusMUf9rFYIU%2BKaVyxIa7WDUN48cWjwdGLjYV9XPy%2FP2lRKjeiiNMYbdknQzJfSzh0HWxx0Aq6zlXdkJUbvSgqFoDC2npaUGXjNupSLNIzcMFWr8lUvUFIBm1ZigETFDZrB4zEJFQVxXV%2Bsztpcs1tMO%2F8LAG3MNI%2BI%2Fp7lT3bj%2F%2BZg6S7d6ROGS96XMS3Am3WffiwNIxueTGrWmRWxS75EQexcJmrQ4ELU%2By3vOXxIvqftT68w6%2BnBryUB5kGE%2B6GljxUFD5y7hZFLM0tfFW9XEZF5PjDbz%2Fx%2Bi0dxEiwvN2mzNpSAWiiy6ZBT31GSRRMtTe9Sm4U%2B8DwSR0fymXmme5fKLGzkySq0xPuFhzN6LyLCoxtbob%2BRyLALNdP8E31enPu%2B1xl5Isg%2FXHINRM29SYzK0u1PlPK78ng%2Bqt4mUlLD7jlzIeBKa1vz%2BU8%2F1ZYvEofc8i6q691PqjYl%2FZK5lFQO1EEremVOv4i2nEYwmGtjtCAk1WFChnamFlEdWyJIerN5pKI4YvsGF%2FwXG8aHuYBg41CfGftl%2FwlJ77dPOQ8QHgp5BZFheyeYwEMijnbz4terE7kVpdvBKOk5lBxtiJILI0ftU%2F4F0k825M%2Ft4w%2FqzIpgY6sgFspzJ6vfwqmIKbmprTCY6NBr4uAZU%2FPUWraWxu3hCydMZTVOjlrab%2Bv5NSdCqWKHvK7Yn89JtE9um3P8Gyev9BFPXT6LykCtjNOulKUQnywvl8ngKdbujNjLAyZb4D0p4dFRFsE2sUTUWNvs%2BVwA%2BYdn4%2BwPkMN5PU0KR78myJ7LyYJGodNLOXcBSV%2FXa396TmeXagW3ihm2U7H%2FvXm1IZmOz%2FflT5y6CEy%2FegChXEVpb6&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230808T111525Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY64PTFOFS%2F20230808%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=e35040e1985923b74081dbdac33f7250949695d95e631d68a8fe20684b3746bc&hash=59ce65176ba4b931ecc905ef2a0bb80561947d73205e8ad2561d63a95552a4fb&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0168192323003143&tid=spdf-41137a89-2992-4585-8512-4303f8dedb0c&sid=b0b6f2a791aeb640d1897e968c8092375869gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=10145807525053555255&rr=7f375764f8ea2338&cc=us)
- [Comparing Explainable Machine Learning Approaches With Traditional Statistical Methods for Evaluating Stroke Risk Models: Retrospective Cohort Study](https://cardio.jmir.org/2023/1/e47736/PDF)
- [An Explainable AI Approach using Graph Learning to Predict ICU Length of Stay](https://shichangzh.github.io/preprints/LoS_XAI_ISR.pdf)
- [Cross Feature Selection to Eliminate Spurious Interactions and Single Feature Dominance Explainable Boosting Machines](https://arxiv.org/ftp/arxiv/papers/2307/2307.08485.pdf)
- [Multi-Objective Optimization of Performance and Interpretability of Tabular Supervised Machine Learning Models](https://arxiv.org/pdf/2307.08175v1.pdf)
- [An explainable model to support the decision about the therapy protocol for AML](https://arxiv.org/pdf/2307.02631.pdf)
- [Assessing wind field characteristics along the airport runway glide slope: an explainable boosting machine-assisted wind tunnel study](https://www.nature.com/articles/s41598-023-36495-5)
- [Trustworthy Academic Risk Prediction with Explainable Boosting Machines](https://link.springer.com/chapter/10.1007/978-3-031-36272-9_38)
- [Binary ECG Classification Using Explainable Boosting Machines for IoT Edge Devices](https://ieeexplore.ieee.org/document/9970834)
- [Explainable artificial intelligence toward usable and trustworthy computer-aided diagnosis of multiple sclerosis from Optical Coherence Tomography](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10406231/)
- [An Interpretable Machine Learning Model with Deep Learning-based Imaging Biomarkers for Diagnosis of Alzheimer’s Disease](https://arxiv.org/pdf/2308.07778.pdf)
- [Prediction of Alzheimer Disease on the DARWIN Dataset with Dimensionality Reduction and Explainability Techniques](https://www.scitepress.org/Papers/2024/130174/130174.pdf)
- [Explainable Boosting Machine for Predicting Alzheimer’s Disease from MRI Hippocampal Subfields](https://link.springer.com/chapter/10.1007/978-3-030-86993-9_31)
- [Comparing explainable machine learning approaches with traditional statistical methods for evaluating stroke risk models: retrospective cohort study](https://pureadmin.qub.ac.uk/ws/portalfiles/portal/495863198/JMIR_Cardio.pdf)
- [Explainable Artificial Intelligence for Cotton Yield Prediction With Multisource Data](https://ieeexplore.ieee.org/document/10214067)
- [Preoperative detection of extraprostatic tumor extension in patients with primary prostate cancer utilizing](https://insightsimaging.springeropen.com/articles/10.1186/s13244-024-01876-5)
- [Monotone Tree-Based GAMI Models by Adapting XGBoost](https://arxiv.org/ftp/arxiv/papers/2309/2309.02426.pdf)
- [Neural Graphical Models](https://arxiv.org/pdf/2210.00453.pdf)
- [FAST: An Optimization Framework for Fast Additive Segmentation in Transparent ML](https://arxiv.org/pdf/2402.12630v1.pdf)
- [The Quantitative Analysis of Explainable AI for Network Anomaly Detection](https://studenttheses.uu.nl/bitstream/handle/20.500.12932/45996/Thesis_SinievanderBen_6021794.pdf?sequence=1&isAllowed=y)
- [Enhancing Predictive Battery Maintenance Through the Use of Explainable Boosting Machine](https://link.springer.com/chapter/10.1007/978-3-031-44146-2_6)
- [Improved Differentially Private Regression via Gradient Boosting](https://arxiv.org/pdf/2303.03451.pdf)
- [Explainable Artificial Intelligence in Job Recommendation Systems](http://essay.utwente.nl/96974/1/Tran_MA_EEMCS.pdf)
- [Diagnosis uncertain models for medical risk prediction](https://arxiv.org/pdf/2306.17337.pdf)
- [Extending Explainable Boosting Machines to Scientific Image Data](https://arxiv.org/pdf/2305.16526.pdf)
- [Pest Presence Prediction Using Interpretable Machine Learning](https://arxiv.org/pdf/2205.07723.pdf)
- [Key Thresholds and Relative Contributions of Knee Geometry, Anteroposterior Laxity, and Body Weight as Risk Factors for Noncontact ACL Injury](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10184233/pdf/10.1177_23259671231163627.pdf)
- [A clinical prediction model for 10-year risk of self-reported osteoporosis diagnosis in pre- and perimenopausal women](https://pubmed.ncbi.nlm.nih.gov/37273115/)
- [epitope1D: Accurate Taxonomy-Aware B-Cell Linear Epitope Prediction](https://www.biorxiv.org/content/10.1101/2022.10.17.512613v1.full.pdf)
- [Explainable Boosting Machines for Slope Failure Spatial Predictive Modeling](https://www.mdpi.com/2072-4292/13/24/4991/htm)
- [Micromodels for Efficient, Explainable, and Reusable Systems: A Case Study on Mental Health](https://arxiv.org/pdf/2109.13770.pdf)
- [Identifying main and interaction effects of risk factors to predict intensive care admission in patients hospitalized with COVID-19](https://www.medrxiv.org/content/10.1101/2020.06.30.20143651v1.full.pdf)
- [Leveraging interpretable machine learning in intensive care](https://link.springer.com/article/10.1007/s10479-024-06226-8#Tab10)
- [Development of prediction models for one-year brain tumour survival using machine learning: a comparison of accuracy and interpretability](https://www.pure.ed.ac.uk/ws/portalfiles/portal/343114800/1_s2.0_S0169260723001487_main.pdf)
- [Using Interpretable Machine Learning to Predict Maternal and Fetal Outcomes](https://arxiv.org/pdf/2207.05322.pdf)
- [Calibrate: Interactive Analysis of Probabilistic Model Output](https://arxiv.org/pdf/2207.13770.pdf)
- [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/pdf/2004.13912.pdf)
- [TabSRA: An Attention based Self-Explainable Model for Tabular Learning](https://www.esann.org/sites/default/files/proceedings/2023/ES2023-37.pdf)
- [Evaluating the Efficacy of Instance Incremental vs. Batch Learning in Delayed Label Environments: An Empirical Study on Tabular Data Streaming for Fraud Detection](https://arxiv.org/pdf/2409.10111v1)
- [Improving Neural Additive Models with Bayesian Principles](https://arxiv.org/pdf/2305.16905.pdf)
- [NODE-GAM: Neural Generalized Additive Model for Interpretable Deep Learning](https://arxiv.org/pdf/2106.01613.pdf)
- [Scalable Interpretability via Polynomials](https://arxiv.org/pdf/2205.14108v1.pdf)
- [Polynomial Threshold Functions of Bounded Tree-Width: Some Explainability and Complexity Aspects](https://arxiv.org/pdf/2501.08297)
- [Neural Basis Models for Interpretability](https://arxiv.org/pdf/2205.14120.pdf)
- [ILMART: Interpretable Ranking with Constrained LambdaMART](https://arxiv.org/pdf/2206.00473.pdf)
- [Integrating Co-Clustering and Interpretable Machine Learning for the Prediction of Intravenous Immunoglobulin Resistance in Kawasaki Disease](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9097874)
- [Distilling Reinforcement Learning Policies for Interpretable Robot Locomotion: Gradient Boosting Machines and Symbolic Regression](https://arxiv.org/pdf/2403.14328)
- [Proxy Endpoints - Bridging clinical trials and real world data](https://deliverypdf.ssrn.com/delivery.php?ID=100104064008112111075114086019087126028049030043069035029115016108019006060084089121082084037060084007106031067094003062092094027085086025068093071031052079088007024075059029108100000124020112107075035009017105116086086122095064020024067066064103085015070113092118127102118080007103101&EXT=pdf&INDEX=TRUE)
- [Application of boosted trees to the prognosis prediction of COVID‐19](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11111612/pdf/HSR2-7-e2104.pdf)
- [Explainable Gradient Boosting for Corporate Crisis Forecasting in Italian Businesses](https://assets-eu.researchsquare.com/files/rs-4426436/v1_covered_0583163e-fa83-4b34-9a7e-eae573b17bd8.pdf?c=1715832940)
- [Revisiting differentially private XGBoost: Are random decision trees really better than greedy ones?](https://openreview.net/pdf?id=bCynxWndWY)
- [Investigating Trust in Human-Machine Learning Collaboration: A Pilot Study on Estimating Public Anxiety from Speech](https://dl.acm.org/doi/pdf/10.1145/3462244.3479926)
- [pureGAM: Learning an Inherently Pure Additive Model](https://www.microsoft.com/en-us/research/uploads/prod/2022/07/pureGAM-camera-ready.pdf)
- [GAMI-Net: An Explainable Neural Network based on Generalized Additive Models with Structured Interactions](https://arxiv.org/pdf/2003.07132v1.pdf)
- [Interpretable Machine Learning based on Functional ANOVA Framework: Algorithms and Comparisons](https://arxiv.org/ftp/arxiv/papers/2305/2305.15670.pdf)
- [Using Model-Based Trees with Boosting to Fit Low-Order Functional ANOVA Models](https://arxiv.org/ftp/arxiv/papers/2207/2207.06950.pdf)
- [Interpretable generalized additive neural networks](https://pdf.sciencedirectassets.com/271700/AIP/1-s2.0-S0377221723005027/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDYA80LSoQY%2FmGTGsi8cQ2BzHoFU7410ljuYQwqt9ht0gIgBg4NSEN4e5jKUouf04uZCPIMh8NHH22jrY3opOG%2Fa1wqsgUIXhAFGgwwNTkwMDM1NDY4NjUiDFuCZlUMsD25uY5h3CqPBW0KcZyo1I0j19n0O26WHoCoxeimG0I7m02rUpQug4EiDYFVkx%2FRqfC4eL2Y0z7iO%2B95NIQ9UrOd3zWWZZPGpKCgHpU1GA4JwHSKNJDi8G2q%2FGm18%2Fl8B9jN4Lq3klUfU3HcjJh%2B4O1aZTJb3PmqDxKn%2BFQIftfS13xNcyqGnGBlw3yaSp3ZXoV55tKSX6b%2Fp5ZuXWORWiC2JlANxa0exR%2FkBeE75gfILdU8bH2TJ1wozoB0yTZwDAl1%2Bc4exGhVdhZRpvr9W6q%2BTG4tx6qhglAwv1uUQN8Zt1z8GEFHMTrtSv5pNJIpLqqMxp62UeufPMesYyoO5RfKjRS96PxYs1S%2FC5zfz0V63kkFtmSVn4IzVQ%2B9tLq%2FEWQ3BvTs8B0cH%2FOm6W8wn4nGk3HywJiUWvGexXahMqDW9o2pq7CWOSoFCKjjkOyxBXAzP0OX5LeCCgOF11BbhNcDSiIqlWQhqsk0738appUu99Yh12XmMWyu6YXAv1jvgrpaliMRkliAu9by418e6%2FBBA%2B%2BfcDQC3VkEv3NpSQklitMaIT2Y624jhM09ntjdC4IcONNRVE3Q5sHIh6DZsBHrPj9oKqpu9nPKnDBrKoAFdnQ%2BkLQ%2B8JAXyCHwd3YBUXQStlYTpUExESOnFFJ36HGJ%2FbkkFC5Ac9W%2BALq%2FkBYIvtPFNBWIGUSC%2BUgSH0kC%2BJqoyYUNHjfYZ3fxCDwI%2BAugNT3UtXtT%2BrnCKlH3f68ZAyOdkFLiHRQevc2%2FRBXJ5gAqCgZFDUVM%2BVjgB%2BInE458PRMxLuRwFHJarOqZhoDvC68ar2q3YDPyqmyUZxaiVrqn2xlJGdh0lcTVwNourzqlY2l4v87nYm7ncxJO%2FiiBQArtSRTOWmkw6JigpQY6sQEhgDdw23Gwj9rSPFlCHfUzj%2B%2BfdgeX3LZpuPITkl6%2BYwjKw0wXpR4c0Rj0IsCH1EAxJcxSLXhSSHgInlZR41EreEAByudeYNtxD0iAQcR3L4RlqTVuI6V3IIcxNltdg5rDAJwUqsGqhMrZOH0uJqQXvLJwxgfkOkckdjdnrfT%2FmOh%2BNjLCR4KTvwTJIC2YAmNHQco2TLKbC27i118DSoKwrYUvb%2BkCTwD3TMkxIf%2BrW5s%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230707T133754Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYTYKBXRE7%2F20230707%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=389ecd144af85f4eae42ab9684f9d56696191a9d8d33c44386ee6af520187724&hash=e798cbd4d80d01d56a2a1ea75a3947b027daecaea5f6e6674a1dc2dbea97dab3&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0377221723005027&tid=spdf-79e45837-627e-4a9f-88f5-7359ecb4ca63&sid=7e54333b754ff04056483e557e54be0269ddgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0b1a5101565e5607565a&rr=7e307c188e87e7cb&cc=mx)
- [A Concept and Argumentation based Interpretable Model in High Risk Domains](https://arxiv.org/pdf/2208.08149.pdf)
- [Analyzing the Differences between Professional and Amateur Esports through Win Probability](https://dl.acm.org/doi/pdf/10.1145/3485447.3512277)
- [Explainable machine learning with pairwise interactions for the classification of Parkinson’s disease and SWEDD from clinical and imaging features](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9132761/pdf/11682_2022_Article_688.pdf)
- [Interpretable Prediction of Goals in Soccer](https://statsbomb.com/wp-content/uploads/2019/10/decroos-interpretability-statsbomb.pdf)
- [Extending the Tsetlin Machine with Integer-Weighted Clauses for Increased Interpretability](https://arxiv.org/pdf/2005.05131.pdf)
- [In Pursuit of Interpretable, Fair and Accurate Machine Learning for Criminal Recidivism Prediction](https://arxiv.org/pdf/2005.04176.pdf)
- [From Shapley Values to Generalized Additive Models and back](https://arxiv.org/pdf/2209.04012.pdf)
- [Developing A Visual-Interactive Interface for Electronic Health Record Labeling](https://arxiv.org/pdf/2209.12778.pdf)
- [Development and Validation of an Interpretable 3-day Intensive Care Unit Readmission Prediction Model Using Explainable Boosting Machines](https://www.medrxiv.org/content/10.1101/2021.11.01.21265700v1.full.pdf)
- [Death by Round Numbers and Sharp Thresholds: How to Avoid Dangerous AI EHR Recommendations](https://www.medrxiv.org/content/10.1101/2022.04.30.22274520v1.full.pdf)
- [Building a predictive model to identify clinical indicators for COVID-19 using machine learning method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9037972/pdf/11517_2022_Article_2568.pdf)
- [Using Innovative Machine Learning Methods to Screen and Identify Predictors of Congenital Heart Diseases](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8777022/pdf/fcvm-08-797002.pdf)
- [Impact of Accuracy on Model Interpretations](https://arxiv.org/pdf/2011.09903.pdf)
- [Machine Learning Algorithms for Identifying Dependencies in OT Protocols](https://www.mdpi.com/1996-1073/16/10/4056)
- [Causal Understanding of Why Users Share Hate Speech on Social Media](https://arxiv.org/pdf/2310.15772.pdf)
- [Explainable Boosting Machine: A Contemporary Glass-Box Model to Analyze Work Zone-Related Road Traffic Crashes](https://www.mdpi.com/2313-576X/9/4/83)
- [Efficient and Interpretable Traffic Destination Prediction using Explainable Boosting Machines](https://arxiv.org/pdf/2402.03457.pdf)
- [Explainable Artificial Intelligence Paves the Way in Precision Diagnostics and Biomarker Discovery for the Subclass of Diabetic Retinopathy in Type 2 Diabetics](https://www.mdpi.com/2218-1989/13/12/1204)
- [A proposed tree-based explainable artificial intelligence approach for the prediction of angina pectoris](https://www.nature.com/articles/s41598-023-49673-2)
- [Explainable Boosting Machine: A Contemporary Glass-Box Strategy for the Assessment of Wind Shear Severity in the Runway Vicinity Based on the Doppler Light Detection and Ranging Data](https://www.mdpi.com/2073-4433/15/1/20)
- [On the Physical Nature of Lya Transmission Spikes in High Redshift Quasar Spectra](https://arxiv.org/pdf/2401.04762.pdf)
- [GRAND-SLAMIN’ Interpretable Additive Modeling with Structural Constraints](https://openreview.net/pdf?id=F5DYsAc7Rt)
- [Identification of groundwater potential zones in data-scarce mountainous region using explainable machine learning](https://www.sciencedirect.com/science/article/pii/S0022169423013598)
- [Explainable Classification Techniques for Quantum Dot Device Measurements](https://arxiv.org/pdf/2402.13699v1.pdf)

# Books that cover EBMs

- [Machine Learning for High-Risk Applications](https://www.oreilly.com/library/view/machine-learning-for/9781098102425/)
- [Interpretable Machine Learning with Python](https://www.amazon.com/Interpretable-Machine-Learning-Python-hands-dp-180323542X/dp/180323542X/)
- [Explainable Artificial Intelligence: An Introduction to Interpretable Machine Learning](https://www.amazon.com/Explainable-Artificial-Intelligence_-An-Introduction-to-Interpretable-XAI/dp/3030833550)
- [Applied Machine Learning Explainability Techniques](https://www.amazon.com/Applied-Machine-Learning-Explainability-Techniques/dp/1803246154)
- [The eXplainable A.I.: With Python examples](https://www.amazon.com/eXplainable-I-Python-examples-ebook/dp/B0B4F98MN6)
- [Platform and Model Design for Responsible AI: Design and build resilient, private, fair, and transparent machine learning models](https://www.amazon.com/Platform-Model-Design-Responsible-transparent/dp/1803237074)
- [Explainable AI Recipes](https://www.amazon.com/Explainable-Recipes-Implement-Explainability-Interpretability-ebook/dp/B0BSF5NBY7)
- [Ensemble Methods for Machine Learning](https://www.amazon.com/Ensemble-Methods-Machine-Learning-Kunapuli/dp/1617297135)

# External tools

- [EBM to Onnx converter by SoftAtHome](https://github.com/interpretml/ebm2onnx)
- [EBM to SQL converter - ML 2 SQL](https://github.com/kaspersgit/ml_2_sql)
- [EBM to PMML converter - SkLearn2PMML](https://github.com/jpmml/sklearn2pmml)
- [EBM visual editor - GAM Changer](https://github.com/interpretml/gam-changer)
- [Interpreting Visual Clusters in Dimensionality Reduction - DimVis](https://github.com/parisa-salmanian/DimVis)

# Contact us

There are multiple ways to get in touch:
- Email us at interpret@microsoft.com
- Or, feel free to raise a GitHub issue

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

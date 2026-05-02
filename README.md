# InterpretML

<a href="https://githubtocolab.com/interpretml/interpret/blob/main/docs/interpret/python/examples/interpretable-classification.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/interpretml/interpret/main?labpath=docs%2Finterpret%2Fpython%2Fexamples%2Finterpretable-classification.ipynb)
![License](https://img.shields.io/github/license/interpretml/interpret.svg?style=flat-square)
![Python Version](https://img.shields.io/pypi/pyversions/interpret.svg?style=flat-square)
![Package Version](https://img.shields.io/pypi/v/interpret.svg?style=flat-square)
![Conda](https://img.shields.io/conda/v/conda-forge/interpret)
![Build Status](https://github.com/interpretml/interpret/actions/workflows/ci.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/github/interpretml/interpret/branch/main/graph/badge.svg?token=aPlXLsPEZD)](https://codecov.io/github/interpretml/interpret)
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

Python 3.10+ | Linux, Mac, Windows
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

[*Notebook for reproducing table*](https://nbviewer.jupyter.org/github/interpretml/interpret/blob/main/docs/benchmarks/ebm-classification-comparison.ipynb)

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
- [Rich Caruana – Friends Don’t Let Friends Deploy Black-Box Models](https://www.youtube.com/watch?v=2YKtNYBuojE)

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
- [Model Interpretation with Microsoft’s Interpret ML](https://medium.com/@sand.mayur/model-interpretation-with-microsofts-interpret-ml-85aa0ad697ae)
- [Explaining Model Pipelines With InterpretML](https://medium.com/@mariusvadeika/explaining-model-pipelines-with-interpretml-a9214f75400b)
- [Explain Your Model with Microsoft’s InterpretML](https://medium.com/@Dataman.ai/explain-your-model-with-microsofts-interpretml-5daab1d693b4)
- [On Model Explainability: From LIME, SHAP, to Explainable Boosting](https://everdark.github.io/k9/notebooks/ml/model_explain/model_explain.nb.html)
- [Dealing with Imbalanced Data (Mortgage loans defaults)](https://mikewlange.github.io/ImbalancedData-/index.html)
- [The Art of Sprezzatura for Machine Learning](https://towardsdatascience.com/the-art-of-sprezzatura-for-machine-learning-e2494c0db727)
- [MCTS EDA which makes sense](https://www.kaggle.com/code/ambrosm/mcts-eda-which-makes-sense/notebook)
- [Explainable Boosting machines for Tabular data](https://www.kaggle.com/code/parulpandey/explainable-boosting-machines-for-tabular-data)
- [Out with the opaque: achieving both model performance and transparency](https://www.theactuary.com/2026/01/12/out-opaque-achieving-both-model-performance-and-transparency)

# Papers that use or compare EBMs

- [Challenging the Performance-Interpretability Trade-off: An Evaluation of Interpretable Machine Learning Models](https://link.springer.com/article/10.1007/s12599-024-00922-2)
- [The hidden risk of round numbers and sharp thresholds in clinical practice](https://www.nature.com/articles/s41746-025-02079-y)
- [Improving Credit Card Fraud Detection with an Optimized Explainable Boosting Machine](https://arxiv.org/pdf/2602.06955)
- [TabArena: A Living Benchmark for Machine Learning on Tabular Data](https://arxiv.org/pdf/2506.16791)
- [Classifier Calibration at Scale: An Empirical Study of Model-Agnostic Post-Hoc Methods](https://arxiv.org/html/2601.19944v1)
- [Statistical Inference for Explainable Boosting Machines](https://arxiv.org/abs/2601.18857)
- [Explainable Boosting Machine for Predicting Claim Severity and Frequency in Car Insurance](https://arxiv.org/pdf/2503.21321)
- [Unveiling the drivers of the Baryon Cycles with Interpretable Multi-step Machine Learning and Simulations](https://arxiv.org/pdf/2504.09744v3)
- [Modelling Container Transhipment Throughput and Analysing Dynamics During Significant External Events: A Case Study of the Port of Busan and the China-to-U.S. Trade Route](https://eprints.soton.ac.uk/505005/1/_PDF_A_Modelling_Container_Transhipment_Throughput_and_Analysing_Dynamics_During_Significant_External_Events_-_A_Case_Study_of_the_Port_of_Busan_and_the_China-to-U.S._Trade_Route.pdf)
- [The Most Important Features in Generalized Additive Models Might Be Groups of Features](https://arxiv.org/pdf/2506.19937)
- [Explainable AI with EDA for V2I path loss prediction](https://www.nature.com/articles/s41598-026-34987-8)
- [A Comparative Analysis of Interpretable Machine Learning Methods](https://arxiv.org/pdf/2601.00428)
- [ParamBoost: Gradient Boosted Piecewise Cubic Polynomials](https://arxiv.org/pdf/2604.18864)
- [GAMFORMER: In-context Learning for Generalized Additive Models](https://arxiv.org/pdf/2410.04560v1)
- [Automatic Piecewise Linear Regression](https://link.springer.com/article/10.1007/s00180-024-01475-4)
- [Accuracy is not enough: explainable boosting machine model and identification of candidate biomarkers for real-time sepsis risk assessment in the emergency department](https://link.springer.com/article/10.1186/s12873-025-01402-w)
- [Explainable Boosting Machine approach identifies risk factors for acute renal failure](https://link.springer.com/article/10.1186/s40635-024-00639-2)
- [Domain-informed explainable boosting machines for trustworthy lateral spread predictions](https://arxiv.org/html/2603.17175v1)
- [An explainable boosting machine model for identifying artifacts caused by formalin-fixed paraffin embedding](https://www.biorxiv.org/content/10.64898/2026.03.10.710815v1.full)
- [Glass Box Machine Learning and Corporate Bond Returns](https://www.nber.org/system/files/working_papers/w33320/w33320.pdf)
- [Data Science with LLMs and Interpretable Models](https://arxiv.org/pdf/2402.14474v1.pdf)
- [DimVis: Interpreting Visual Clusters in Dimensionality Reduction With Explainable Boosting Machine](https://arxiv.org/pdf/2402.06885.pdf)
- [Distill knowledge of additive tree models into generalized linear models](https://detralytics.com/wp-content/uploads/2023/10/Detra-Note_Additive-tree-ensembles.pdf)
- [Explainable Boosting Machines with Sparsity - Maintaining Explainability in High-Dimensional Settings](https://arxiv.org/abs/2311.07452)
- [Cost of Explainability in AI: An Example with Credit Scoring Models](https://link.springer.com/chapter/10.1007/978-3-031-44064-9_26)
- [iTARGET: Interpretable Tailored Age Regression for Grouped Epigenetic Traits](https://arxiv.org/pdf/2501.02401v1)
- [Vehicle Fuel Optimization Under Real-World Driving Conditions: An Explainable Artificial Intelligence Approach](https://arxiv.org/pdf/2107.06031v3)
- [Explainable Boosting Machine for Structural Health Assessment: An Interpretable Approach to Data-Driven Structural Assessment](https://dpi-proceedings.com.destechpub.a2hosted.com/index.php/shm2025/article/view/37379/35953)
- [Interpretable Machine Learning Leverages Proteomics to Improve Cardiovascular Disease Risk Prediction and Biomarker Identification](https://www.medrxiv.org/content/10.1101/2024.01.12.24301213v1.full.pdf)
- [Interpretable Additive Tabular Transformer Networks](https://openreview.net/pdf/d2f0db2646418b24bb322fc1f4082fd9e65409c2.pdf)
- [Interpretable Survival Analysis for Heart Failure Risk Prediction](https://arxiv.org/pdf/2310.15472.pdf)
- [Investigating Protective and Risk Factors and Predictive Insights for Aboriginal Perinatal Mental Health: Explainable Artificial Intelligence Approach](https://www.jmir.org/2025/1/e68030)
- [Explainable Learning Framework for the Assessment and Prediction of Wind Shear-Induced Aviation Turbulence](https://www.mdpi.com/2073-4433/16/12/1318)
- [HearteXplain: explainable prediction of acute heart failure and identification of hematologic biomarkers using EBMs and Morris sensitivity analysis](https://www.nature.com/articles/s41598-025-23668-7)
- [Identification of a Novel Lipidomic Biomarker for Hepatocyte Carcinoma Diagnosis: Advanced Boosting Machine Learning Techniques Integrated with Explainable Artificial Intelligence](https://www.mdpi.com/2218-1989/15/11/716)
- [Interpretable machine learning for precision cognitive aging](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1560064/full)
- [LLMs Understand Glass-Box Models, Discover Surprises, and Suggest Repairs](https://arxiv.org/pdf/2308.01157.pdf)
- [Model Interpretability in Credit Insurance](http://hdl.handle.net/10400.5/27507)
- [Enhancing ML Interpretability for Credit Scoring](https://arxiv.org/html/2509.11389v1)
- [Transparent and Fair Profiling in Employment Services: Evidence from Switzerland](https://www.arxiv.org/pdf/2509.11847)
- [Federated Boosted Decision Trees with Differential Privacy](https://arxiv.org/pdf/2210.02910.pdf)
- [Differentially private and explainable boosting machine with enhanced utility](https://www.sciencedirect.com/science/article/abs/pii/S0925231224011950)
- [Balancing Explainability and Privacy in Bank Failure Prediction: A Differentially Private Glass-Box Approach](https://ieeexplore.ieee.org/abstract/document/10818483)
- [GAM(E) changer or not? An evaluation of interpretable machine learning models based on additive model constraints](https://arxiv.org/pdf/2204.09123.pdf)
- [GAM Coach: Towards Interactive and User-centered Algorithmic Recourse](https://arxiv.org/pdf/2302.14165.pdf)
- [Missing Values and Imputation in Healthcare Data: Can Interpretable Machine Learning Help?](https://arxiv.org/pdf/2304.11749v1.pdf)
- [Revealing the Galaxy-Halo Connection Through Machine Learning](https://arxiv.org/pdf/2204.10332.pdf)
- [How the Galaxy–Halo Connection Depends on Large-Scale Environment](https://arxiv.org/pdf/2402.07995.pdf)
- [A diagnostic support system based on interpretable machine learning and oscillometry for accurate diagnosis of respiratory dysfunction in silicosis](https://www.biorxiv.org/content/10.1101/2025.01.08.632001v1.full.pdf)
- [Using Explainable Boosting Machines (EBMs) to Detect Common Flaws in Data](https://link.springer.com/chapter/10.1007/978-3-030-93736-2_40)
- [Differentially Private Gradient Boosting on Linear Learners for Tabular Data Analysis](https://openreview.net/pdf?id=uPF2bs14E3p)
- [Concrete compressive strength prediction using an explainable boosting machine model](https://www.sciencedirect.com/science/article/pii/S2214509523000244/pdfft?md5=171c275b6bcae8897cef03d931e908e2&pid=1-s2.0-S2214509523000244-main.pdf)
- [When Interpretability Meets Generalization: Delta-GAM for Robust Extrapolation in Out-of-Distribution Settings](https://dl.acm.org/doi/pdf/10.1145/3711896.3737180)
- [Mitigating Cognitive Biases in Predicting Student Dropout: Global and Local Explainability with Explainable Boosting Machine](https://media.proquest.com/media/hms/PFT/1/YyZOd?_s=66xFGpyzvpyWM%2FMUrO68tlIKHDE%3D)
- [Proxy endpoints - bridging clinical trials and real world data](https://www.sciencedirect.com/science/article/pii/S1532046424001412)
- [Machine Learning Model Reveals Determinators for Admission to Acute Mental Health Wards From Emergency Department Presentations](https://onlinelibrary.wiley.com/doi/epdf/10.1111/inm.13402)
- [Interpretable Machine Learning Models for Predicting Perioperative Myocardial Injury in Non-Cardiac Surgery](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5379891)
- [Predicting Robotic Hysterectomy Incision Time: Optimizing Surgical Scheduling with Machine Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741200/pdf/e2024.00040.pdf)
- [Performance of explainable artificial intelligence in guiding the management of patients with a pancreatic cyst](https://www.sciencedirect.com/science/article/pii/S1424390324007300)
- [Explainable boosting machine for structural health assessment of reinforced concrete beams using crack width measurements](https://sites.utexas.edu/ferche/files/2025/10/Explainable-Boosting-Machine-for-Structural-Health-Assessment.pdf)
- [Predicting Blood Pressure Variability in Hemodialysis Using an Explainable Boosting Machine Model](https://academic.oup.com/ckj/article/18/12/sfaf349/8325109)
- [Validating Explainer Methods: A Functionally Grounded Approach for Numerical Forecasting](https://onlinelibrary.wiley.com/doi/epdf/10.1002/for.70060)
- [Using explainable machine learning and fitbit data to investigate predictors of adolescent obesity](https://www.nature.com/articles/s41598-024-60811-2)
- [Estimate Deformation Capacity of Non-Ductile RC Shear Walls Using Explainable Boosting Machine](https://arxiv.org/pdf/2301.04652.pdf)
- [Targeting resources efficiently and justifiably by combining causal machine learning and theory](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9768181/pdf/frai-05-1015604.pdf)
- [Extractive Text Summarization Using Generalized Additive Models with Interactions for Sentence Selection](https://arxiv.org/pdf/2212.10707.pdf)
- [Post-Hoc Interpretation of Transformer Hyperparameters with Explainable Boosting Machines](https://aclanthology.org/2022.blackboxnlp-1.5.pdf)
- [Interpretable machine learning for predicting pathologic complete response in patients treated with chemoradiation therapy for rectal adenocarcinoma](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9771385/pdf/frai-05-1059033.pdf)
- [Exploring the Balance between Interpretability and Performance with carefully designed Constrainable Neural Additive Models](https://www.sciencedirect.com/science/article/pii/S1566253523001987)
- [Interpretable machine learning algorithms to predict leaf senescence date of deciduous trees](https://www.sciencedirect.com/science/article/pii/S0168192323003143)
- [Cross Feature Selection to Eliminate Spurious Interactions and Single Feature Dominance Explainable Boosting Machines](https://arxiv.org/pdf/2307.08485)
- [Multi-Objective Optimization of Performance and Interpretability of Tabular Supervised Machine Learning Models](https://arxiv.org/pdf/2307.08175v1.pdf)
- [Assessing wind field characteristics along the airport runway glide slope: an explainable boosting machine-assisted wind tunnel study](https://www.nature.com/articles/s41598-023-36495-5)
- [Explainable Modeling for Wind Power Forecasting: A Glass-Box Approach with High Accuracy](https://arxiv.org/pdf/2310.18629)
- [Trustworthy Academic Risk Prediction with Explainable Boosting Machines](https://link.springer.com/chapter/10.1007/978-3-031-36272-9_38)
- [Binary ECG Classification Using Explainable Boosting Machines for IoT Edge Devices](https://ieeexplore.ieee.org/document/9970834)
- [Explainable artificial intelligence toward usable and trustworthy computer-aided diagnosis of multiple sclerosis from Optical Coherence Tomography](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10406231/)
- [An Interpretable Machine Learning Model with Deep Learning-based Imaging Biomarkers for Diagnosis of Alzheimer’s Disease](https://arxiv.org/pdf/2308.07778.pdf)
- [Explainable Boosting Machine for Predicting Alzheimer’s Disease from MRI Hippocampal Subfields](https://link.springer.com/chapter/10.1007/978-3-030-86993-9_31)
- [Explainable Artificial Intelligence for Cotton Yield Prediction With Multisource Data](https://ieeexplore.ieee.org/document/10214067)
- [Preoperative detection of extraprostatic tumor extension in patients with primary prostate cancer utilizing [68Ga]Ga-PSMA-11 PET/MRI](https://insightsimaging.springeropen.com/articles/10.1186/s13244-024-01876-5)
- [Monotone Tree-Based GAMI Models by Adapting XGBoost](https://arxiv.org/pdf/2309.02426)
- [Extending Explainable Boosting Machines to Scientific Image Data](https://arxiv.org/pdf/2305.16526.pdf)
- [Exploring explanation deficits in subclinical mastitis detection with explainable boosting machines](https://link.springer.com/article/10.1007/s44279-025-00246-z)
- [Pest Presence Prediction Using Interpretable Machine Learning](https://arxiv.org/pdf/2205.07723.pdf)
- [Key Thresholds and Relative Contributions of Knee Geometry, Anteroposterior Laxity, and Body Weight as Risk Factors for Noncontact ACL Injury](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10184233/pdf/10.1177_23259671231163627.pdf)
- [Explainable Boosting Machines for Slope Failure Spatial Predictive Modeling](https://www.mdpi.com/2072-4292/13/24/4991/htm)
- [Leveraging interpretable machine learning in intensive care](https://link.springer.com/article/10.1007/s10479-024-06226-8#Tab10)
- [Using Interpretable Machine Learning to Predict Maternal and Fetal Outcomes](https://arxiv.org/pdf/2207.05322.pdf)
- [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/pdf/2004.13912.pdf)
- [Improving Neural Additive Models with Bayesian Principles](https://arxiv.org/pdf/2305.16905.pdf)
- [NODE-GAM: Neural Generalized Additive Model for Interpretable Deep Learning](https://arxiv.org/pdf/2106.01613.pdf)
- [Scalable Interpretability via Polynomials](https://arxiv.org/pdf/2205.14108v1.pdf)
- [Neural Basis Models for Interpretability](https://arxiv.org/pdf/2205.14120.pdf)
- [ILMART: Interpretable Ranking with Constrained LambdaMART](https://arxiv.org/pdf/2206.00473.pdf)
- [Integrating Co-Clustering and Interpretable Machine Learning for the Prediction of Intravenous Immunoglobulin Resistance in Kawasaki Disease](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9097874)
- [Explainable Gradient Boosting for Corporate Crisis Forecasting in Italian Businesses](https://assets-eu.researchsquare.com/files/rs-4426436/v1_covered_0583163e-fa83-4b34-9a7e-eae573b17bd8.pdf?c=1715832940)
- [Investigating Trust in Human-Machine Learning Collaboration: A Pilot Study on Estimating Public Anxiety from Speech](https://dl.acm.org/doi/pdf/10.1145/3462244.3479926)
- [pureGAM: Learning an Inherently Pure Additive Model](https://www.microsoft.com/en-us/research/uploads/prod/2022/07/pureGAM-camera-ready.pdf)
- [GAMI-Net: An Explainable Neural Network based on Generalized Additive Models with Structured Interactions](https://arxiv.org/pdf/2003.07132v1.pdf)
- [Interpretable Machine Learning based on Functional ANOVA Framework: Algorithms and Comparisons](https://arxiv.org/pdf/2305.15670)
- [Using Model-Based Trees with Boosting to Fit Low-Order Functional ANOVA Models](https://arxiv.org/pdf/2207.06950)
- [Interpretable generalized additive neural networks](https://www.sciencedirect.com/science/article/pii/S0377221723005027)
- [Explainable machine learning with pairwise interactions for the classification of Parkinson’s disease and SWEDD from clinical and imaging features](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9132761/pdf/11682_2022_Article_688.pdf)
- [In Pursuit of Interpretable, Fair and Accurate Machine Learning for Criminal Recidivism Prediction](https://arxiv.org/pdf/2005.04176.pdf)
- [From Shapley Values to Generalized Additive Models and back](https://arxiv.org/pdf/2209.04012.pdf)
- [Developing A Visual-Interactive Interface for Electronic Health Record Labeling: An Explainable Machine Learning Approach](https://arxiv.org/pdf/2209.12778.pdf)
- [Development and Validation of an Interpretable 3-day Intensive Care Unit Readmission Prediction Model Using Explainable Boosting Machines](https://www.medrxiv.org/content/10.1101/2021.11.01.21265700v1.full.pdf)
- [EPS: An Explainable Post-Shot Expected Goal Metric for Evaluating Goalkeepers and Attackers](https://hal.science/hal-05258651/)
- [Prediction of surface rougness of additively manufactured and machined parts via machine learning](https://open.metu.edu.tr/handle/11511/115643)
- [Development of Explainable Machine Learning Models to Predict Outcomes After Platelet-Rich Plasma Injections for Knee Osteoarthritis](https://journals.sagepub.com/doi/full/10.1177/23259671251349743)
- [Knowledge-Guided Machine Learning: Illustrating the use of Explainable Boosting Machines to Identify Overshooting Tops in Satellite Imagery](https://arxiv.org/pdf/2507.03183)
- [Building a predictive model to identify clinical indicators for COVID-19 using machine learning method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9037972/pdf/11517_2022_Article_2568.pdf)
- [Using Innovative Machine Learning Methods to Screen and Identify Predictors of Congenital Heart Diseases](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8777022/pdf/fcvm-08-797002.pdf)
- [Explainable Boosting Machine: A Contemporary Glass-Box Model to Analyze Work Zone-Related Road Traffic Crashes](https://www.mdpi.com/2313-576X/9/4/83)
- [Efficient and Interpretable Traffic Destination Prediction using Explainable Boosting Machines](https://arxiv.org/pdf/2402.03457.pdf)
- [Explainable Artificial Intelligence Paves the Way in Precision Diagnostics and Biomarker Discovery for the Subclass of Diabetic Retinopathy in Type 2 Diabetics](https://www.mdpi.com/2218-1989/13/12/1204)
- [A proposed tree-based explainable artificial intelligence approach for the prediction of angina pectoris](https://www.nature.com/articles/s41598-023-49673-2)
- [Explainable Boosting Machine: A Contemporary Glass-Box Strategy for the Assessment of Wind Shear Severity in the Runway Vicinity Based on the Doppler Light Detection and Ranging Data](https://www.mdpi.com/2073-4433/15/1/20)
- [On the Physical Nature of Lyα Transmission Spikes in High Redshift Quasar Spectra](https://arxiv.org/pdf/2401.04762.pdf)
- [GRAND-SLAMIN’ Interpretable Additive Modeling with Structural Constraints](https://openreview.net/pdf?id=F5DYsAc7Rt)
- [Identification of groundwater potential zones in data-scarce mountainous region using explainable machine learning](https://www.sciencedirect.com/science/article/pii/S0022169423013598)
- [Explainable Classification Techniques for Quantum Dot Device Measurements](https://arxiv.org/pdf/2402.13699v1.pdf)

# Books that cover EBMs

- [Machine Learning for High-Risk Applications](https://www.oreilly.com/library/view/machine-learning-for/9781098102425/)
- [Explainable AI with Python](https://www.amazon.com/Explainable-AI-Python-Antonio-Cecco/dp/303192228X/ref=sr_1_1?crid=1D8M7T61VP2N9&dib=eyJ2IjoiMSJ9.Yuoa4g8YohqY3C8Umi1E0qOZfUgATZQR2GpBnxVUHuwvH1ml3gkgioEHWOHqod3AQtc8jPXxX6KvFVjsjuN3oZUzw8k5XALdGIsx8bk6ZCT4yRNEX8vgJHjhf7oVFRdbSvsRAWcDem2GI4I8F0xawxpgxYvpNDCzk9J8-FmIxeMN0tMST89WbTVTNO30crxXj5gok1PuZV_1IxYa6tyVhmtGlesWHcewNG0h7wfMIbk.FAlg6RW_KFog7rTG2O5z0425E88Z20EHAqSM2deROyI&dib_tag=se&keywords=Explainable+AI+with+Python&qid=1754559323&sprefix=%2Caps%2C147&sr=8-1)
- [Interpretable Machine Learning with Python](https://www.amazon.com/Interpretable-Machine-Learning-Python-hands-dp-180323542X/dp/180323542X/)
- [Explainable Artificial Intelligence: An Introduction to Interpretable Machine Learning](https://www.amazon.com/Explainable-Artificial-Intelligence_-An-Introduction-to-Interpretable-XAI/dp/3030833550)
- [Applied Machine Learning Explainability Techniques](https://www.amazon.com/Applied-Machine-Learning-Explainability-Techniques/dp/1803246154)
- [The eXplainable A.I.: With Python examples](https://www.amazon.com/eXplainable-I-Python-examples-ebook/dp/B0B4F98MN6)
- [Platform and Model Design for Responsible AI: Design and build resilient, private, fair, and transparent machine learning models](https://www.amazon.com/Platform-Model-Design-Responsible-transparent/dp/1803237074)
- [Explainable AI Recipes](https://www.amazon.com/Explainable-Recipes-Implement-Explainability-Interpretability-ebook/dp/B0BSF5NBY7)
- [Ensemble Methods for Machine Learning](https://www.amazon.com/Ensemble-Methods-Machine-Learning-Kunapuli/dp/1617297135)
- [Interpretability and Explainability in AI Using Python](https://www.amazon.com/Interpretability-Explainability-Using-Python-Decision-Making/dp/B0F536GGT5/ref=sr_1_1?crid=5QJBMJKZOJ4H&dib=eyJ2IjoiMSJ9.oiAm3_DaQcHqA3YNRGrC70d1KcpeDZReI29ATLUdCe0VWb6wKLo-U1iLlyW24-u0SIdRxce8m_E1urP9pl-Qwjm9JSfu6l8nX3Ws9itlpXw.AJkX9wz_VBTb3OSeiW22Fbt2NCI3_kM7zJ_TCTUcbt0&dib_tag=se&keywords=interpretability+and+explainability+in+ai&qid=1748138569&sprefix=interpretability+and+explainability+in+ai%2Caps%2C175&sr=8-1)

# External tools

- [R package for building EBMs through reticulate](https://github.com/bgreenwell/ebm)
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

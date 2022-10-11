# InterpretML - Alpha Release

![License](https://img.shields.io/github/license/microsoft/interpret.svg?style=flat-square)
![Python Version](https://img.shields.io/pypi/pyversions/interpret.svg?style=flat-square)
![Package Version](https://img.shields.io/pypi/v/interpret.svg?style=flat-square)
![Build Status](https://img.shields.io/azure-devops/build/ms/interpret/293/develop.svg?style=flat-square)
![Coverage](https://img.shields.io/azure-devops/coverage/ms/interpret/293/develop.svg?style=flat-square)
![LGTM Grade](https://img.shields.io/lgtm/grade/python/github/interpretml/interpret?style=flat-square)
![Maintenance](https://img.shields.io/maintenance/yes/2022?style=flat-square)
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

Python 3.6+ | Linux, Mac, Windows
```sh
pip install interpret
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

[*Notebook for reproducing table*](https://nbviewer.jupyter.org/github/interpretml/interpret/blob/master/benchmarks/EBM%20Classification%20Comparison.ipynb)

# Supported Techniques

| Interpretability Technique  | Type               |
|-----------------------------|--------------------|
| [Explainable Boosting](https://interpret.ml/docs/ebm.html)        | glassbox model     |
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
![Global Explanation Image](./examples/python/assets/readme_ebm_global_specific.PNG?raw=true)

<br/>

Understand individual predictions
```python
ebm_local = ebm.explain_local(X_test, y_test)
show(ebm_local)
```
![Local Explanation Image](./examples/python/assets/readme_ebm_local_specific.PNG?raw=true)

<br/>

And if you have multiple model explanations, compare them
```python
show([logistic_regression_global, decision_tree_global])
```
![Dashboard Image](./examples/python/assets/readme_dashboard.PNG?raw=true)

<br/>

If you need to keep your data private, use Differentially Private EBMs (see [DP-EBMs](http://proceedings.mlr.press/v139/nori21a/nori21a.pdf))

```python
from interpret.privacy import DPExplainableBoostingClassifier, DPExplainableBoostingRegressor

dp_ebm = DPExplainableBoostingClassifier(epsilon=1, delta=1e-5) # Specify privacy parameters
dp_ebm.fit(X_train, y_train)

show(dp_ebm.explain_global()) # Identical function calls to standard EBMs
```

<br/>
<br/>

For more information, see the [documentation](https://interpret.ml/docs/getting-started.html).
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
    <a href="http://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf">Paper link</a>
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
    <a href="http://www-personal.umich.edu/~harmank/Papers/CHI2020_Interpretability.pdf">Paper link</a>
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
    <a href="http://proceedings.mlr.press/v139/nori21a/nori21a.pdf">Paper link</a>
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
 url = {http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf}
}
    </pre>
    <a href="http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf">Paper link</a>
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
    <a href="http://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf">Paper link</a>
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

# External links

- [Interpretable or Accurate? Why Not Both?](https://towardsdatascience.com/interpretable-or-accurate-why-not-both-4d9c73512192)
- [The Explainable Boosting Machine. As accurate as gradient boosting, as interpretable as linear regression.](https://towardsdatascience.com/the-explainable-boosting-machine-f24152509ebb)
- [Performance And Explainability With EBM](https://blog.oakbits.com/ebm-algorithm.html)
- [InterpretML: Another Way to Explain Your Model](https://towardsdatascience.com/interpretml-another-way-to-explain-your-model-b7faf0a384f8)
- [A gentle introduction to GA2Ms, a white box model](https://blog.fiddler.ai/2019/06/a-gentle-introduction-to-ga2ms-a-white-box-model)
- [Model Interpretation with Microsoft’s Interpret ML](https://medium.com/@sand.mayur/model-interpretation-with-microsofts-interpret-ml-85aa0ad697ae)
- [Explaining Model Pipelines With InterpretML](https://medium.com/@mariusvadeika/explaining-model-pipelines-with-interpretml-a9214f75400b)
- [Explain Your Model with Microsoft’s InterpretML](https://medium.com/@Dataman.ai/explain-your-model-with-microsofts-interpretml-5daab1d693b4)
- [On Model Explainability: From LIME, SHAP, to Explainable Boosting](https://everdark.github.io/k9/notebooks/ml/model_explain/model_explain.nb.html)
- [Dealing with Imbalanced Data (Mortgage loans defaults)](https://mikewlange.github.io/ImbalancedData-/index.html)
- [The right way to compute your Shapley Values](https://towardsdatascience.com/the-right-way-to-compute-your-shapley-values-cfea30509254)
- [The Art of Sprezzatura for Machine Learning](https://towardsdatascience.com/the-art-of-sprezzatura-for-machine-learning-e2494c0db727)
- [Mixing Art into the Science of Model Explainability](https://towardsdatascience.com/mixing-art-into-the-science-of-model-explainability-312b8216fa95)

# Papers that use or compare EBMs

- [Federated Boosted Decision Trees with Differential Privacy](https://arxiv.org/pdf/2210.02910.pdf)
- [Pest Presence Prediction Using Interpretable Machine Learning](https://ieeexplore.ieee.org/document/9816284) - [preprint](https://arxiv.org/pdf/2205.07723.pdf)
- [GAM(E) CHANGER OR NOT? AN EVALUATION OF INTERPRETABLE MACHINE LEARNING MODELS](https://arxiv.org/pdf/2204.09123.pdf)
- [Revealing the Galaxy-Halo Connection Through Machine Learning](https://arxiv.org/pdf/2204.10332.pdf)
- [Explainable Artificial Intelligence for COVID-19 Diagnosis Through Blood Test Variables](https://link.springer.com/content/pdf/10.1007/s40313-021-00858-y.pdf)
- [Using Explainable Boosting Machines (EBMs) to Detect Common Flaws in Data](https://link.springer.com/chapter/10.1007/978-3-030-93736-2_40)
- [Explainable Boosting Machines for Slope Failure Spatial Predictive Modeling](https://www.mdpi.com/2072-4292/13/24/4991/htm)
- [Micromodels for Efficient, Explainable, and Reusable Systems: A Case Study on Mental Health](https://arxiv.org/pdf/2109.13770.pdf)
- [Identifying main and interaction effects of risk factors to predict intensive care admission in patients hospitalized with COVID-19](https://www.medrxiv.org/content/10.1101/2020.06.30.20143651v1.full.pdf)
- [Comparing the interpretability of machine learning classifiers for brain tumour survival prediction](https://deliverypdf.ssrn.com/delivery.php?ID=760122118067103094108090123091079011028032009009023085005014014002123105085114025022024005047078031019089073120012025117073002064031071072113006066035001068125027021087087083085026100009018045107092063001023068071002124070107120120007014102094103069089119026110104107005031095001092090&EXT=pdf&INDEX=TRUE)
- [Using Interpretable Machine Learning to Predict Maternal and Fetal Outcomes](https://arxiv.org/pdf/2207.05322.pdf)
- [Calibrate: Interactive Analysis of Probabilistic Model Output](https://arxiv.org/pdf/2207.13770.pdf)
- [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/pdf/2004.13912.pdf)
- [NODE-GAM: Neural Generalized Additive Model for Interpretable Deep Learning](https://arxiv.org/pdf/2106.01613.pdf)
- [Scalable Interpretability via Polynomials](https://arxiv.org/pdf/2205.14108v1.pdf)
- [Neural Basis Models for Interpretability](https://arxiv.org/pdf/2205.14120.pdf)
- [ILMART: Interpretable Ranking with Constrained LambdaMART](https://arxiv.org/pdf/2206.00473.pdf)
- [Integrating Co-Clustering and Interpretable Machine Learning for the Prediction of Intravenous Immunoglobulin Resistance in Kawasaki Disease](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9097874)
- [GAMI-Net: An Explainable Neural Network based on Generalized Additive Models with Structured Interactions](https://arxiv.org/pdf/2003.07132v1.pdf)
- [A Concept and Argumentation based Interpretable Model in High Risk Domains](https://arxiv.org/pdf/2208.08149.pdf)
- [Analyzing the Differences between Professional and Amateur Esports through Win Probability](https://dl.acm.org/doi/pdf/10.1145/3485447.3512277)
- [Explainable machine learning with pairwise interactions for the classifcation of Parkinson’s disease and SWEDD from clinical and imaging features](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9132761/pdf/11682_2022_Article_688.pdf)
- [Interpretable Prediction of Goals in Soccer](http://statsbomb.com/wp-content/uploads/2019/10/decroos-interpretability-statsbomb.pdf)
- [Extending the Tsetlin Machine with Integer-Weighted Clauses for Increased Interpretability](https://arxiv.org/pdf/2005.05131.pdf)
- [In Pursuit of Interpretable, Fair and Accurate Machine Learning for Criminal Recidivism Prediction](https://arxiv.org/pdf/2005.04176.pdf)
- [From Shapley Values to Generalized Additive Models and back](https://arxiv.org/pdf/2209.04012.pdf)
- [An Explainable Machine Learning Approach to Visual-Interactive Labeling: A Case Study on Non-communicable Disease Data](https://arxiv.org/pdf/2209.12778.pdf)
- [Development and Validation of an Interpretable 3-day Intensive Care Unit Readmission Prediction Model Using Explainable Boosting Machines](https://www.medrxiv.org/content/10.1101/2021.11.01.21265700v1.full.pdf)
- [Death by Round Numbers and Sharp Thresholds: How to Avoid Dangerous AI EHR Recommendations](https://www.medrxiv.org/content/10.1101/2022.04.30.22274520v1.full.pdf)
- [Building a predictive model to identify clinical indicators for COVID-19 using machine learning method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9037972/pdf/11517_2022_Article_2568.pdf)
- [Using Innovative Machine Learning Methods to Screen and Identify Predictors of Congenital Heart Diseases](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8777022/pdf/fcvm-08-797002.pdf)
- [Explainable Boosting Machine for Predicting Alzheimer’s Disease from MRI Hippocampal Subfields](https://link.springer.com/chapter/10.1007/978-3-030-86993-9_31)
- [Impact of Accuracy on Model Interpretations](https://arxiv.org/pdf/2011.09903.pdf)

# Books that discuss EBMs

- [Interpretable Machine Learning with Python](https://www.amazon.com/Interpretable-Machine-Learning-Python-hands/dp/180020390X)
- [Explainable Artificial Intelligence: An Introduction to Interpretable Machine Learning](https://www.amazon.com/Explainable-Artificial-Intelligence_-An-Introduction-to-Interpretable-XAI/dp/3030833550)
- [Machine Learning for High-Risk Applications](https://www.oreilly.com/library/view/machine-learning-for/9781098102425/)

# External tools

- [EBM to Onnx converter by SoftAtHome](https://github.com/interpretml/ebm2onnx)
- [GAM Changer](https://github.com/interpretml/gam-changer)
- [ML 2 SQL (experimental)](https://github.com/kaspersgit/ml_2_sql)

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

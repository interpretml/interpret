# InterpretML - Alpha Release

![License](https://img.shields.io/github/license/microsoft/interpret.svg?style=flat-square)
![Python Version](https://img.shields.io/pypi/pyversions/interpret.svg?style=flat-square)
![Package Version](https://img.shields.io/pypi/v/interpret.svg?style=flat-square)
![Build Status](https://img.shields.io/azure-devops/build/ms/interpret/293/master.svg?style=flat-square)
![Coverage](https://img.shields.io/azure-devops/coverage/ms/interpret/293/master.svg?style=flat-square)
![Maintenance](https://img.shields.io/maintenance/yes/2020?style=flat-square)
<br/>
> ### In the beginning machines learned in darkness, and data scientists struggled in the void to explain them. 
> ### Let there be light.

InterpretML is an open-source package that incorporates state-of-the-art machine learning interpretability techniques under one roof. With this package, you can train interpretable glassbox models and explain blackbox systems. InterpretML helps you understand your model's global behavior, or understand the reasons behind individual predictions.

Interpretability is essential for:
- Model debugging - Why did my model make this mistake?
- Detecting fairness issues - Does my model discriminate?
- Human-AI cooperation - How can I understand and trust the model's decisions?
- Regulatory compliance - Does my model satisfy legal requirements?
- High-risk applications - Healthcare, finance, judicial, ...

![](https://github.com/interpretml/assets/blob/master/interpret-highlight.gif)

# Installation

Python 3.5+ | Linux, Mac, Windows
```sh
pip install interpret
```

# Introducing the Explainable Boosting Machine (EBM)

EBM is an interpretable model developed at Microsoft Research<sup>[*](#citations)</sup>. It uses modern machine learning techniques like bagging, gradient boosting, and automatic interaction detection to breathe new life into traditional GAMs (Generalized Additive Models). This makes EBMs as accurate as state-of-the-art techniques like random forests and gradient boosted trees. However, unlike these blackbox models, EBMs produce lossless explanations and are editable by domain experts.

| Dataset/AUROC | Domain  | Logistic Regression | Random Forest | XGBoost        | Explainable Boosting Machine |
|---------------|---------|:-------------------:|:-------------:|:--------------:|:----------------------------:|
| Adult Income  | Finance | .907±.003           | .903±.002     | .922±.002      | **_.928±.002_**              |
| Heart Disease | Medical | .895±.030           | .890±.008     | .870±.014      | **_.916±.010_**              |
| Breast Cancer | Medical | **_.995±.005_**     | .992±.009     | **_.995±.006_**| **_.995±.006_**              |
| Telecom Churn | Business| .804±.015           | .824±.002     | .850±.006      | **_.851±.005_**              |
| Credit Fraud  | Security| .979±.002           | .950±.007     | **_.981±.003_**| .975±.005                    |

[*Notebook for reproducing table*](https://nbviewer.jupyter.org/github/interpretml/interpret/blob/master/benchmarks/EBM%20Classification%20Comparison.ipynb)

# Supported Techniques

|Interpretability Technique|Type|Examples|
|--|--|--------------------|
|Explainable Boosting|glassbox model|[Notebooks](https://nbviewer.jupyter.org/github/interpretml/interpret/blob/master/examples/python/notebooks/Interpretable%20Classification%20Methods.ipynb)|
|Decision Tree|glassbox model|[Notebooks](https://nbviewer.jupyter.org/github/interpretml/interpret/blob/master/examples/python/notebooks/Interpretable%20Classification%20Methods.ipynb)|
|Decision Rule List|glassbox model|Coming Soon|
|Linear/Logistic Regression|glassbox model|[Notebooks](https://nbviewer.jupyter.org/github/interpretml/interpret/blob/master/examples/python/notebooks/Interpretable%20Classification%20Methods.ipynb)|
|SHAP Kernel Explainer|blackbox explainer|[Notebooks](https://nbviewer.jupyter.org/github/interpretml/interpret/blob/master/examples/python/notebooks/Explaining%20Blackbox%20Classifiers.ipynb)|
|SHAP Tree Explainer|blackbox explainer|Coming Soon|
|LIME|blackbox explainer|[Notebooks](https://nbviewer.jupyter.org/github/interpretml/interpret/blob/master/examples/python/notebooks/Explaining%20Blackbox%20Classifiers.ipynb)|
|Morris Sensitivity Analysis|blackbox explainer|[Notebooks](https://nbviewer.jupyter.org/github/interpretml/interpret/blob/master/examples/python/notebooks/Explaining%20Blackbox%20Classifiers.ipynb)|
|Partial Dependence|blackbox explainer|[Notebooks](https://nbviewer.jupyter.org/github/interpretml/interpret/blob/master/examples/python/notebooks/Explaining%20Blackbox%20Classifiers.ipynb)|

In addition to these, InterpretML is extended by the following repositories:

- [**Interpret-Community**](https://github.com/interpretml/interpret-community): Experimental repository with additional interpretability methods and utility functions to handle real-world datasets and workflows.
- [**Interpret-Text**](https://github.com/interpretml/interpret-text): Supports a collection of interpretability techniques for models trained on text data.

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

And if you have multiple models, compare them
```python
show([logistic_regression, decision_tree])
```
![Dashboard Image](./examples/python/assets/readme_dashboard.PNG?raw=true)
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
      <em>"InterpretML: A Unified Framework for Machine Learning Interpretability" (H. Nori, S. Jenkins, P. Koch, and R.
        Caruana 2019)</em>
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
      <em>"Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission" (R. Caruana,
        Y. Lou, J. Gehrke, P. Koch, M. Sturm, and N. Elhadad 2015)</em>
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
      <em>"Accurate intelligible models with pairwise interactions" (Y. Lou, R. Caruana, J. Gehrke, and G. Hooker
        2013)</em>
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
  url = {https://plot.ly} }
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
- [How to Explain Models with IntepretML Deep Dive](https://www.youtube.com/watch?v=WwBeKMQ0-I8)

# External links

- [A gentle introduction to GA2Ms, a white box model](https://blog.fiddler.ai/2019/06/a-gentle-introduction-to-ga2ms-a-white-box-model)
- [On Model Explainability: From LIME, SHAP, to Explainable Boosting](https://everdark.github.io/k9/notebooks/ml/model_explain/model_explain.nb.html)
- [Benchmarking and MLI experiments on the Adult dataset](https://github.com/sayakpaul/Benchmarking-and-MLI-experiments-on-the-Adult-dataset/blob/master/Benchmarking_experiments_on_the_Adult_dataset_and_interpretability.ipynb)
- [Dealing with Imbalanced Data (Mortgage loans defaults)](https://mikewlange.github.io/ImbalancedData-/index.html)
- [Kaggle PGA Tour analysis by GAM](https://www.kaggle.com/juyamagu/pga-tour-analysis-by-gam)
- [Explaining Model Pipelines With InterpretML](https://medium.com/@mariusvadeika/explaining-model-pipelines-with-interpretml-a9214f75400b)
- [Explain Your Model with Microsoft’s InterpretML](https://medium.com/@Dataman.ai/explain-your-model-with-microsofts-interpretml-5daab1d693b4)
- [Model Interpretation with Microsoft’s Interpret ML](https://medium.com/@sand.mayur/model-interpretation-with-microsofts-interpret-ml-85aa0ad697ae)

# Papers using or comparing EBMs

- [Identifying main and interaction effects of risk factors to predict intensive care admission in patients hospitalized with COVID-19](https://www.medrxiv.org/content/10.1101/2020.06.30.20143651v1.full.pdf)
- [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/pdf/2004.13912.pdf)
- [Integrating Co-Clustering and Interpretable Machine Learning for the Prediction of Intravenous Immunoglobulin Resistance in Kawasaki Disease](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9097874)
- [GAMI-Net: An Explainable Neural Network based on Generalized Additive Models with Structured Interactions](https://arxiv.org/pdf/2003.07132v1.pdf)
- [Interpretable Prediction of Goals in Soccer](http://statsbomb.com/wp-content/uploads/2019/10/decroos-interpretability-statsbomb.pdf)
- [Extending the Tsetlin Machine with Integer-Weighted Clauses for Increased Interpretability](https://arxiv.org/pdf/2005.05131.pdf)
- [In Pursuit of Interpretable, Fair and Accurate Machine Learning for Criminal Recidivism Prediction](https://arxiv.org/pdf/2005.04176.pdf)
- [Galaxy Zoo: Probabilistic Morphology through Bayesian CNNs and Active Learning](https://arxiv.org/pdf/1905.07424.pdf)

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

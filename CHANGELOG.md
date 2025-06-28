# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and the versioning is mostly derived from [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [v0.6.13] - 2025-06-28
### Added
- support for early termination of EBM training using a callback mechanism

## [v0.6.12] - 2025-06-17
### Changed
- support for numpy 2.3.x
- increased default number of interaction terms

## [v0.6.11] - 2025-06-05
### Changed
- increased max_rounds to 50,000
- possibly faster prediction in some scenarios (unverified)
- remove obsolete dash components from requirements

## [v0.6.10] - 2025-03-26
### Added
- reorder_classes function which allows reordering of the classes after fitting
- support for ARM based Linux
### Changed
- changed default to max_leaves=2 for classification
- changed default to n_jobs=2
- changed default to outer_bags=14
### Fixed
- restrict to dash 2.x since visualizations are not working on dash 3.x

## [v0.6.9] - 2025-01-06
### Added
- refitting of the intercept term after fitting the rest of the model to improve the intercept value
- new options for handling missing values: "low", "high", "separate", and "gain"
- use Fischer (1958) for handling categorical values. This is the same method employed by LightGBM.
- added new parameters to control overfitting of nominal categoricals: gain\_scale, min\_cat\_samples, cat\_smooth
### Changed
- enable AVX-512 by default
- modified default EBM parameters: outer\_bags=16, n\_jobs=-1
### Fixed
- fixed memory leak in the purification function

## [v0.6.8] - 2024-12-09
### Fixed
- resolved new scikit-learn requirement for having \_\_sklearn\_tags\_\_
- changed position of ClassifierMixin and RegressorMixin inheritance to satisfy scikit-learn check
- reliable handling of sparse arrays (previously only sparse matrices worked)

## [v0.6.7] - 2024-11-27
### Changed
- minimum python version increased to 3.9
- minimum numpy version increased to 1.25
### Fixed
- removed scipy dependency to resolve Issue #588

## [v0.6.6] - 2024-11-20
### Changed
- added predict_with_uncertainty function by @degenfabian in PR #584
- handle mono-classification in SHAP by @degenfabian in PR #582
- improvements to tree building in C++
### Fixed
- issue that develop/debug options were not being honored in Windows when 1<n_jobs in joblib
- fix several bugs in C++ from negative hessians or negative gain values caused by floating point noise

## [v0.6.5] - 2024-10-23
### Changed
- default EBM parameters changed to improve model performance
- switch to using exact versions of exp/log instead of the previously used approximate versions
### Fixed
- fix issue where very large feature values fail in the UI PR #581 by @degenfabian

## [v0.6.4] - 2024-09-28
### Added
- support for regularization parameters reg_alpha, and reg_lambda in EBMs
- support for the parameter max_delta_step in EBMs
- improved fitting speed for most of the alternative objectives

## [v0.6.3] - 2024-08-07
### Added
- visualizations for the APRL (Automatic Piecewise Linear Regression) package by @mathias-von-ottenbreit
### Changed
- early_stopping_tolerance default changed to 1e-5 to reduce EBMs fitting time slightly
- shuffle initial feature order within each bag and during greedy boosting
### Fixed
- fixed numpy 2.0 issue in the Marginal class

## [v0.6.2] - 2024-06-22
### Added
- pass optional kwargs to DecisionTreeClassifier in PR #537 by @busFred
- support for multiclass purification
- support for higher dimensional purification
- allow higher levels of purification than would be supported via the tolerance parameter
### Changed
- numpy 2.0 support for EBMs
- update documentation regarding monotonicity in PR #531 by @Krzys25
- moved purification utility from "interpret/glassbox/_ebm/_research" to "interpret.utils"
### Fixed
- possible fix for issue #543 where merge_ebms was creating unexpected NaN values

## [v0.6.1] - 2024-04-14
### Fixed
- added compatibility with numpy 2.0 thanks to @DerWeh in PR #525
- fixed bug that was preventing SIMD from being used in python
- removed approximate division in SIMD since the approximation was too inaccurate
### Changed
- EBM fitting time reduced

## [v0.6.0] - 2024-03-16
### Added
- Documentation on recommended hyperparameters to help users optimize their models.
- Support for monotone_constraints during model fitting, although post-processed monotonization is still suggested/preferred.
- The EBMModel class now includes _more_tags for better integration with the scikit-learn API, thanks to contributions from @DerWeh.
### Changed
- Default max_rounds parameter increased from 5,000 to 25,000, for improved model accuracy.
- Numerous code simplifications, additional tests, and enhancements for scikit-learn compatibility, thanks to @DerWeh.
- The greedy boosting algorithm has been updated to support variable-length greedy sections, offering more flexibility during model training.
- Full compatibility with Python 3.12.
- Removal of the DecisionListClassifier from our documentation, as the skope-rules package seems to no longer be actively maintained.
### Fixed
- The sweep function now properly returns self, correcting an oversight identified by @alvanli.
- Default exclude parameter set to None, aligning with scikit-learn's expected defaults, fixed by @DerWeh.
- A potential bug when converting features from categorical to continuous values has been addressed.
- Updated to handle the new return format for TreeShap in the SHAP 0.45.0 release.
### Breaking Changes
- replaced the greediness \_\_init\_\_ parameter with greedy_ratio and cyclic_progress parameters for better control of the boosting process
  (see documentation for notes on greedy_ratio and cyclic_progress)
- replaced breakpoint_iteration_ with best_iteration_, which now contains the number of boosting steps rather than the number of boosting rounds

## [v0.5.1] - 2024-02-08
### Added
- Added new init parameter: interaction_smoothing_rounds
- Added new init parameter: min_hessian
- synthetic dataset generator (make_synthetic) for testing GAMs and for documentation
### Changed
- default parameters have been modified to improve the accuracy of EBMs
- changed boosting internals to use LogitBoost to improve accuracy
- changed interaction detection to use hessians to improve interaction selection
- enabled smoothing_rounds by default to improve the smoothness of EBMs
- added the ability to specify interactions via feature names or negative indexing
- improved the speed of Morris sensitivity and partial dependence
- python 3.12 support for core EBMs. Some of our optional dependencies do not yet support python 3.12 though
- made early stopping more consistent and changed the early_stopping_tolerance to be a percentage
### Fixed
- avoid displaying a scroll bar by default in jupyter notebook cells
- removed the dependency on deprecated distutils
### Breaking Changes
- changed the internal representation for classifiers that have just 1 class

## [v0.5.0] - 2023-12-13
### Added
- added support for AVX-512 in PyPI installations to improve fitting speed
- introduced an option to disable SIMD optimizations through the debug_mode function in python
- exposed public utils.link_func and utils.inv_link functions
### Changed
- the interpret-core package now installs the dependencies required to build and predict EBMs
  by default without needing to specify the [required] pip install flag
- experimental/private support for OVR multiclass EBMs
- added bagged_intercept_ attribute to store the intercepts for the bagged models
### Fixed
- resolved an issue in merge_ebms where the merge would fail if all EBMs in the 
  merge contained features with only one bin (issue #485)
- resolved multiple future warnings from other packages
### Breaking Changes
- changed how monoclassification (degenerate classification with 1 class) is expressed
- replaced predict_and_contrib function with simpler eval_terms function that returns 
  only the per-term contribution values. If you need both the contributions and predictions use:
  interpret.utils.inv_link(ebm.eval_terms(X).sum(axis=1) + ebm.intercept_, ebm.link_)
- separate to_json into to_jsonable (for python objects) and to_json (for files) functions
- create a new link function string for multiclass that is separate from binary classification
- for better scikit-learn compliance, removed the decision_function from the ExplainableBoostingRegressor

## [v0.4.4] - 2023-08-26
### Added
- added the following model editing functions: copy, remove_terms, remove_features, sweep, scale
- added experimental support for a JSON exporter function: to_json

## [v0.4.3] - 2023-08-04
### Changed
- Training speed improvements due to the use of SIMD on Intel processors. 
  Results may vary, but expect approx 2.75x faster for classification and 1.3x faster for RMSE regression
- Changed from using 64-bit floats to using 32-bit floats internally. Regression performed on datasets with large 
  targets that sum to greater than 3.4E+38 will overflow.
### Fixed
- Fixed an issue with the monotonize function that would occur when monotonizing a feature with missing values
- Resolved issue where excluding the 1st feature would cause an exception

## [v0.4.2] - 2023-05-31
### Added
- support for specifying outer bags
### Changed
- exceptions raised in the joblib child processes will be re-raised in the main process rather than be expressed as a TerminatedWorkerError
- small additional improvements in memory compression
- small improvements in maximizing the benefit of the privacy budget for Differentially Private EBMs
### Fixed
- fixed segfault that was occurring in the Anaconda build
- fixed a bug that would prevent Differentially Private EBMs from using the exclude parameter

## [v0.4.1] - 2023-05-16
### Added
- support for visualizations in streamlit
### Fixed
- fixed dangling pointer issue in call to CalcInteractionStrength

## [v0.4.0] - 2023-05-11
### Added
- alternative objective functions: poisson_deviance, tweedie_deviance, gamma_deviance, pseudo_huber, rmse_log (log link)
- greediness __init__ parameter that allows selecting a behavior between cyclic boosting and greedy boosting
- smoothing_rounds __init__ parameter
- added type hints to the EBM __init__ parameters and class attributes
- init_score parameter to allow boosting and prediction on top of a previous model
- multiclass support in merge_ebms
- ability to monotonize features using post process model editing
### Changed
- default BaseLinear regressor is changed from Lasso to LinearRegression class
- placed limits on the amount of memory used to find interactions with high cardinality categoricals
### Fixed
- validation_size of 0 is now handled by disabling early_stopping and using the final model
### Breaking Changes
- replaced the __init__ param "mains" with "exclude"
- removed the binning __init__ param as this functionality was already fully supported in feature_types
- removed the unused zero_val_count attribute and n_samples attribute
- renamed the noise_scale_ attribute to noise_scale_boosting_ and added noise_scale_binning_ to DPEBMs

## [v0.3.2] - 2023-03-14
### Fixed
- fix the issue that the shared library would only work on newer linux versions

## [v0.3.1] - 2023-03-13
### Added
- Mac m1 support in conda-forge
- SPOTGreedy prototype selection (PR #392)
### Fixed
- fix visualization when both cloud and non-cloud environments are detected (PR #210)
- fix ShapTree bug where it was treating classifiers as regressors
- resolve scikit-learn warnings occurring when models were trained using Pandas DataFrames
- change the defaults to prefer 'continuous' over 'nominal' when a feature has 1 or 2 unique float64 values
### Breaking Changes
- in the blackbox and greybox explainers, change from accepting a predict_fn to 
  accepting either a model or a predict_fn
- feature type 'categorical' has been renamed to 'nominal' for the remaining 
  feature_type parameters in the package (EBMs were already using 'nominal')
- removed the unused sampler parameters to the Explainer classes

## [v0.3.0] - 2022-11-16
### Added
- Full Complexity EBMs with higher order interactions supported: GA3M, GA4M, GA5M, etc... 
  3-way and higher-level interactions lose exact global interpretability, but retain exact local explanations
  Higher level interactions need to be explicitly specified. No automatic FAST detection yet
- Mac m1 support
- support for ordinals
- merge_ebms now supports merging models with interactions, including higher-level interactions
- added classic composition option during Differentially Private binning
- support for different kinds of feature importances (avg_weight, min_max)
- exposed interaction detection API (FAST algorithm)
- API to calculate and show the importances of groups of features and terms.
### Changed
- memory efficiency: About 20x less memory is required during fitting
- predict time speed improvements. About 50x faster for Pandas CategoricalDType, 
  and varying levels of improvements for other data types
- handling of the differential privacy DPOther bin, and non-DP unknowns has been unified by having a universal unknown bin
- bin weights have been changed from per-feature to per-term and are now multi-dimensional
- improved scikit-learn compliance: We now conform to the scikit-learn 1.0 feature names API by using 
  self.feature_names_in_ for the X column names and self.n_features_in_. 
  We use the matching self.feature_types_in_ for feature types, and self.term_names_ for the additive term names.
### Fixed
- merge_ebms now distributes bin weights proportionally according to volume when splitting bins
- DP-EBMs now use sample weights instead of bin counts, which preserves privacy budget
- improved scikit-learn compliance: The following __init__ attributes are no longer overwritten 
  during calls to fit: self.interactions, self.feature_names, self.feature_types
- better handling of floating point overflows when calculating gain and validation metrics
### Breaking Changes
- EBMUtils.merge_models function has been renamed to merge_ebms
- renamed binning type 'quantile_humanized' to 'rounded_quantile'
- feature type 'categorical' has been specialized into separate 'nominal' and 'ordinal' types
- EBM models have changed public attributes:
  - ```
    feature_groups_ -> term_features_
    global_selector -> n_samples_, unique_val_counts_, and zero_val_counts_
    domain_size_ -> min_target_, max_target_
    additive_terms_ -> term_scores_
    bagged_models_ -> BaseCoreEBM has been depricated and the only useful attribute has been moved 
                      into the main EBM class (bagged_models_.model_ -> bagged_scores_)
    feature_importances_ -> has been changed into the function term_importances(), which can now also 
                            generate different types of importances
    preprocessor_ & pair_preprocessor_ -> attributes have been moved into the main EBM model class (details below)
    ```
- EBMPreprocessor attributes have been moved to the main EBM model class
  - ```
    col_names_ -> feature_names_in_
    col_types_ -> feature_types_in_
    col_min_ -> feature_bounds_
    col_max_ -> feature_bounds_
    col_bin_edges_ -> bins_
    col_mapping_ -> bins_
    hist_counts_ -> histogram_counts_
    hist_edges_ -> histogram_edges_
    col_bin_counts_ -> bin_weights_ (and is now a per-term tensor)
    ```

## [v0.2.7] - 2021-09-23
### Added
- Synapse cloud support for visualizations.
### Fixed
- All category names in bar charts now visible for inline rendering (used in cloud environments).
- Joblib preference was previously being overriden. This has been reverted to honor the user's preference.
- Bug in categorical binning for differentially privatized EBMs has been fixed.

## [v0.2.6] - 2021-07-20
### Added
- Differential-privacy augmented EBMs now available as `interpret.privacy.{DPExplainableBoostingClassifier,DPExplainableBoostingRegressor}`.
- Packages `interpret` and `interpret-core` now distributed via docker.
### Changed
- Sampling code including stratification within EBM now performed in native code.
### Fixed
- Computer provider with `joblib` can now support multiple engines with serialization support.
- Labels are now all shown for inline rendering of horizontal bar charts.
- JS dependencies updated.

## [v0.2.5] - 2021-06-21
### Added
- Sample weight support added for EBM.
- Joint `predict_and_contrib` added to EBM where both predictions and feature contributions are generated in one call.
- EBM predictions now substantially faster with categorical featured predictions.
- Preliminary documentation for all of `interpret` now public at https://interpret.ml/docs.
- Decision trees now work in cloud environments (InlineRenderer support).
- Packages `interpret` and `interpret-core` now distributed via sdist.
### Fixed
- EBM uniform binning bug fixed where empty bins can raise exceptions.
- Users can no longer include duplicate interaction terms for EBM.
- CSS adjusted for inline rendering such that it does not interfere with its hosting environment.
- JS dependencies updated.
### Experimental
- Ability to merge multiple EBM models into one. Found in `interpret.glassbox.ebm.utils`.

## [v0.2.4] - 2021-01-19
### Fixed
- Bug fix on global EBM plots.
- Rendering fix for AzureML notebooks.
### Changed
- JavaScript dependencies for inline renderers updated.

## [v0.2.3] - 2021-01-13

**Major** upgrades to EBM in this release. Automatic interaction detection is now
included by default. This will increase accuracy substantially in most cases.
Numerous optimizations to support this, especially around binary classification.
Expect similar or slightly slower training times due to interactions.

### Fixed
- Automated interaction detection uses low-resolution binning
  for both FAST and pairwise training.
### Changed
- EBM argument has been reduced from `outer_bags=16` to `outer_bags=8`.
- EBM now includes interactions by default from `interactions=0` to `interactions=10`.
- Algorithm `treeinterpreter` is now unstable due to upstream dependencies.
- Automated interaction detection now operates from two-pass to one-pass.
- Numeric approximations used in boosting (i.e. approx log / exp).
- Some arguments have been re-ordered for EBM initialization.

## [v0.2.2] - 2020-10-19
### Fixed
- Fixed bug on predicting unknown categories with EBM.
- Fixed bug on max value being placed in its own bin for EBM pre-processing.
- Numerous native fixes and optimizations.
### Added
- Added `max_interaction_bins` as argument to EBM learners for different sized
  bins on interactions, separate to mains.
- New binning method 'quantile_humanized' for EBM.
### Changed
- Interactions in EBM now use their own pre-processing, separate to mains.
- Python 3.5 no longer supported.
- Switched from Python to native code for binning.
- Switched from Python to native code for PRNG in EBM.

## [v0.2.1] - 2020-08-07
### Added
- Python 3.8 support.
### Changed
- Dash based visualizations will always default to listen port 7001 on first attempt;
  if the first attempt fails it will try a random port between 7002-7999.
### Experimental (WIP)
- Further cloud environment support.
- Improvements for multiclass EBM global graphs.

## [v0.2.0] - 2020-07-21
### Breaking Changes
- With warning, EBM classifier adapts internal validation size
  when there are too few instances relative to number of unique classes.
  This ensures that there is at least one instance of each class in the validation set.
- Cloud Jupyter environments now use a CDN to fix major rendering bugs and performance.
  - CDN currently used is https://unpkg.com
  - If you want to specify your own CDN, add the following as the top cell
    ```python
    from interpret import set_visualize_provider
    from interpret.provider import InlineProvider
    from interpret.version import __version__

    # Change this to your custom CDN.
    JS_URL = "https://unpkg.com/@interpretml/interpret-inline@{}/dist/interpret-inline.js".format(__version__)
    set_visualize_provider(InlineProvider(js_url=JS_URL))
    ```
- EBM has changed initialization parameters:
  - ```
    schema -> DROPPED
    n_estimators -> outer_bags
    holdout_size -> validation_size
    scoring -> DROPPED
    holdout_split -> DROPPED
    main_attr -> mains
    data_n_episodes -> max_rounds
    early_stopping_run_length -> early_stopping_rounds
    feature_step_n_inner_bags -> inner_bags
    training_step_epsiodes -> DROPPED
    max_tree_splits -> max_leaves
    min_cases_for_splits -> DROPPED
    min_samples_leaf -> ADDED (Minimum number of samples that are in a leaf)
    binning_strategy -> binning
    max_n_bins -> max_bins
    ```
- EBM has changed public attributes:
  - ```
    n_estimators -> outer_bags
    holdout_size -> validation_size
    scoring -> DROPPED
    holdout_split -> DROPPED
    main_attr -> mains
    data_n_episodes -> max_rounds
    early_stopping_run_length -> early_stopping_rounds
    feature_step_n_inner_bags -> inner_bags
    training_step_epsiodes -> DROPPED
    max_tree_splits -> max_leaves
    min_cases_for_splits -> DROPPED
    min_samples_leaf -> ADDED (Minimum number of samples that are in a leaf)
    binning_strategy -> binning
    max_n_bins -> max_bins

    attribute_sets_ -> feature_groups_
    attribute_set_models_ -> additive_terms_ (Pairs are now transposed)
    model_errors_ -> term_standard_deviations_

    main_episode_idxs_ -> breakpoint_iteration_[0]
    inter_episode_idxs_ -> breakpoint_iteration_[1]

    mean_abs_scores_ -> feature_importances_
    ```
### Fixed
- Internal fixes and refactor for native code.
- Updated dependencies for JavaScript layer.
- Fixed rendering bugs and performance issues around cloud Jupyter notebooks.
- Logging flushing bug fixed.
- Labels that are shaped as nx1 matrices now automatically transform to vectors for training.
### Experimental (WIP)
- Added support for AzureML notebook VM.
- Added local explanation visualizations for multiclass EBM.

## [v0.1.22] - 2020-04-27
### Upcoming Breaking Changes
- EBM initialization arguments and public attributes will change in a near-future release.
- There is a chance Explanation API will change in a near-future release.
### Added
- Docstrings for top-level API including for glassbox and blackbox.
### Fixed
- Minor fix for linear models where class wasn't propagating for logistic.
### Experimental
- For research use, exposed optional_temp_params for EBM's Python / native layer.

## [v0.1.21] - 2020-04-02
### Added
- Module "glassbox.ebm.research" now has purification utilities.
- EBM now exposes "max_n_bins" argument for its preprocessing stage.
### Fixed
- Fix intercept not showing for local EBM binary classification.
- Stack trace information exposed for extension system failures.
- Better handling of sparse to dense conversions for all explainers.
- Internal fixes for native code.
- Better NaN / infinity handling within EBM.
### Changed
- Binning strategy for EBM now defaulted to 'quantile' instead of 'uniform'.

## [v0.1.20] - 2019-12-11
### Fixed
- **Major bug fix** around EBM interactions. If you use interactions, please upgrade immediately.
  Part of the pairwise selection was not operating as expected and has been corrected.
- Fix for handling dataframes when no named columns are specified.
- Various EBM fixes around corner-case datasets.
### Changed
- All top-level methods relating to show's backing web server now use visualize provider directly.
  In theory this shouldn't affect top-level API usage, but please raise an issue in the event of failure.
- Memory footprint heavily reduced for EBM at around 2-3 times.

## [v0.1.19] - 2019-10-25
### Changed
- Changed classification metric exposed between C++/python for EBMs to log loss for future public use.
- Warnings provided when extensions error on load.
### Fixed
- Package joblib added to interpret-core as "required" extra.
- Compiler fixes for Oracle Developer Studio.
- Removed undefined behavior in EBM for several unlikely scenarios.

## [v0.1.18] - 2019-10-09
### Added
- Added "main_attr" argument to EBM models. Can now select a subset of features to train main effects on.
- Added AzureML notebook VM detection for visualizations (switches to inline).
### Fixed
- Missing values now correctly throw exceptions on explainers.
- Major visualization fix for pairwise interaction heatmaps from EBM.
- Corrected inline visualization height in Notebooks.
### Changed
- Various internal C++ fixes.
- New error messages around EBM if the model isn't fitted before calling explain_*.

## [v0.1.17] - 2019-09-24
### Fixed
- Morris sensitivity now works for both predict and predict_proba on scikit models.
- Removal of debug print statements around blackbox explainers.
### Changed
- Dependencies for numpy/scipy/pandas/scikit-learn relaxed to (1.11.1,0.18.1,0.19.2, 0.18.1) respectively.
- Visualization provider defaults set by environment detection (cloud and local use different providers).
### Experimental (WIP)
- Inline visualizations for show(explanation). This allows cloud notebooks, and offline notebook support.
  Dashboard integration still ongoing.

## [v0.1.16] - 2019-09-17
### Added
- Visualize and compute platforms are now refactored and use an extension system. Details on use upcoming in later release.
- Package interpret is now a meta-package using interpret-core.
  This enables partial installs via interpret-core for production environments.
### Fixed
- Updated SHAP dependency to require dill.
### Experimental (WIP)
- Greybox introduced (explainers that only work for specific types of models). Starting with SHAP tree and TreeInterpreter.
- Extension system now works across all explainer types and providers.

## [v0.1.15] - 2019-08-26
### Experimental (WIP)
- Multiclass EBM added. Includes visualization and postprocessing. Currently does not support multiclass pairs.

## [v0.1.14] - 2019-08-20
### Fixed
- Fixed occasional browser crash relating to density graphs.
- Fixed decision trees not displaying in Jupyter notebooks.
### Changed
- Dash components no longer pinned. Upgraded to latest.
- Upgrade from dash-table-experiment to dash-table.
- Numerous renames within native code.
### Experimental (WIP)
- Explanation data methods for PDP, EBM enabled for mli interop.

## [v0.1.13] - 2019-08-14
### Added
- EBM has new parameter 'binning_strategy'. Can now support quantile based binning.
- EBM now gracefully handles many edge cases around data.
- Selenium support added for visual smoke tests.
### Fixed
- Method debug_mode now works in wider environments including WSL.
- Linear models in last version returned the same graphs no matter the selection. Fixed.
### Changed
- Testing requirements now fully separate from default user install.
- Internal EBM class has many renames associated with native codebase. Attribute has been changed to Feature.
- Native codebase has many renames. Diff commits from v0.1.12 to v0.1.13 for more details.
- Dependency gevent lightened to take 1.3.6 or greater. This affects cloud/older Python environments.
- Installation for interpret package should now be 'pip install -U interpret'.
- Removal of skope-rules as a required dependency. User now has to install it manually.
- EBM parameter 'cont_n_bins' renamed to 'max_n_bins'.
### Experimental (WIP)
- Extensions validation method is hardened to ensure blackbox specs are closely met.
- Explanation methods data and visual, require key of form ('mli', key), to access mli interop.

## [v0.1.12] - 2019-08-09
### Fixed
- Fixed EBM bug where 2 features with 1 state are included in the dataset.
- Fixed EBM bug that was causing processing of attributes past an attribute combination with 0 useful attributes to
 fail.

## [v0.1.11] - 2019-08-09
### Added
- C++ testing framework added.
- More granular options for training EBM (not public-facing, added for researchers)
### Fixed
- Improved POSIX compliance for build scripts.
- Failure cases handled better for EBM in both Python/native layer.
- Fixed a bug around dash relating to dependencies.
- Removed dead code around web server for visualization.
### Changed
- For Python setup.py, requirements.txt now used for holding dependencies.
- Directory structure changed for whole repository, in preparation for R support.
- Native code further optimized with compiler flags.
- Consistent scaling for EBM plots across all features.
- For explanation's data method, behavior will be non-standard at key equals -1.
- Testing suite for visual interface added via selenium.
### Experimental (WIP)
- Extension system for blackbox explainers added. Enables other packages to register into interpret.
- Data standardization under way, currently for linear, LIME, SHAP where key equals -1 for data method.

## [v0.1.10] - 2019-07-16
### Fixed
- Fix for duplicated logs.
- EBM now throws exception for multi-class (not supported yet).
- Added requests as dependency.
### Changed
- File requirements.txt renamed to dev-requirements.txt
- Native libraries' names now start with 'lib_' prefix.
- Adjusted return type for debug_mode method to provide logging handler.
- EBM native layer upgraded asserts to use logging.
- EBM native layer hardened for edge case data.
- Adjustments to dev dependencies.
- Method debug_mode defaults log level to INFO.

## [v0.1.9] - 2019-06-14
### Added
- Added method debug_mode in develop module.
- Connected native logging to Python layer.
- Native libraries can now be in release/debug mode.
### Fixed
- Increased system compatibility for C++ code.
### Changed
- Debug related methods expose memory info in human readable form.
- Clean-up of logging levels.
- Various internal C+ fixes.

## [v0.1.8] - 2019-06-07
### Fixed
- Fixed calibration issue with EBM.
- Method show_link fix for anonymous explanation lists.
### Changed
- Method show_link now takes same arguments as show.
- Better error messages with random port allocation.
- More testing for various modules.
- Various internal C+ fixes.

## [v0.1.7] - 2019-06-03
### Added
- Added show_link method. Exposes the URL of show(explanation) as a returned string.
### Fixed
- Fixed shutdown_show_server, can now be called multiple times without failure.
### Changed
- Hardened status_show_server method.
- Testing added for interactive module.
- Removal of extra memory allocation in C++ code for EBM.
- Various internal C++ fixes.

## [v0.1.6] - 2019-05-31
### Added
- Integer indexing for preserve method.
- Public-facing CI build added. Important for pull requests.
### Changed
- Visual-related imports are now loaded when visualize is called for explanations.

## [v0.1.5] - 2019-05-30
### Added
- Added preserve method. Can now save visuals into notebook/file - does not work with decision trees.
- Added status_show_server method. Acts as a check for server reachability.
- Exposed init_show_server method. Can adjust address, base_url, etc.
- Added print_debug_info method in develop module. Important for troubleshooting/bug-reports.
### Fixed
- Various internal C++ fixes.
- Minor clean up on example notebooks.
### Changed
- Additional dependency required: psutil.
- Test refactoring.

## [v0.1.4] - 2019-05-23
### Added
- Root path for show server now has a light monitor page.
- Logging registration can now print to both standard streams and files.
### Fixed
- Error handling for non-existent links fixed for visualization backend.
- In some circumstances, Python process will hang. Resolved with new threading.
### Changed
- Unpinning scipy version, upstream dependencies now compatible with latest.
- Show server is now run by a thread directly, not via executor pools.
- Re-enabled notebook/show tests, new threading resolves hangs.
- Small clean-up of setup.py and Azure pipelines config.

## [v0.1.3] - 2019-05-21
### Added
- Model fit can now support lists of lists as instance data.
- Model fit can now support lists for label data.
### Fixed
- Various internal C++ fixes.
### Changed
- Removed hypothesis as public test dependency.
- C++ logging introduced (no public access).

## [v0.1.2] - 2019-05-17
### Added
- EBM can now disable early stopping with run length set to -1.
- EBM tracking of final episodes per base estimator.
### Fixed
- Pinning scipy, until upstream dependencies are compatible.
### Changed
- Clean-up of EBM logging for training.
- Temporary disable of notebook/show tests until CI environment is fixed.

## [v0.1.1] - 2019-05-16
### Added
- Added server shutdown call for 'show' method.
### Fixed
- Axis titles now included in performance explainer.
- Fixed hang on testing interface.

## [v0.1.0] - 2019-05-14
### Added
- Added port number assignments for 'show' method.
- Includes codebase of v0.0.6.
### Changed
- Native code build scripts hardened.
- Libraries are statically linked where possible.
- Code now conforms to Python Black and its associated flake8.

[v0.2.7]: https://github.com/interpretml/interpret/releases/tag/v0.2.7
[v0.2.6]: https://github.com/interpretml/interpret/releases/tag/v0.2.6
[v0.2.5]: https://github.com/interpretml/interpret/releases/tag/v0.2.5
[v0.2.4]: https://github.com/interpretml/interpret/releases/tag/v0.2.4
[v0.2.3]: https://github.com/interpretml/interpret/releases/tag/v0.2.3
[v0.2.2]: https://github.com/interpretml/interpret/releases/tag/v0.2.2
[v0.2.1]: https://github.com/interpretml/interpret/releases/tag/v0.2.1
[v0.2.0]: https://github.com/interpretml/interpret/releases/tag/v0.2.0
[v0.1.22]: https://github.com/interpretml/interpret/releases/tag/v0.1.22
[v0.1.21]: https://github.com/interpretml/interpret/releases/tag/v0.1.21
[v0.1.20]: https://github.com/interpretml/interpret/releases/tag/v0.1.20
[v0.1.19]: https://github.com/interpretml/interpret/releases/tag/v0.1.19
[v0.1.18]: https://github.com/interpretml/interpret/releases/tag/v0.1.18
[v0.1.17]: https://github.com/interpretml/interpret/releases/tag/v0.1.17
[v0.1.16]: https://github.com/interpretml/interpret/releases/tag/v0.1.16
[v0.1.15]: https://github.com/interpretml/interpret/releases/tag/v0.1.15
[v0.1.14]: https://github.com/interpretml/interpret/releases/tag/v0.1.14
[v0.1.13]: https://github.com/interpretml/interpret/releases/tag/v0.1.13
[v0.1.12]: https://github.com/interpretml/interpret/releases/tag/v0.1.12
[v0.1.11]: https://github.com/interpretml/interpret/releases/tag/v0.1.11
[v0.1.10]: https://github.com/interpretml/interpret/releases/tag/v0.1.10
[v0.1.9]: https://github.com/interpretml/interpret/releases/tag/v0.1.9
[v0.1.8]: https://github.com/interpretml/interpret/releases/tag/v0.1.8
[v0.1.7]: https://github.com/interpretml/interpret/releases/tag/v0.1.7
[v0.1.6]: https://github.com/interpretml/interpret/releases/tag/v0.1.6
[v0.1.5]: https://github.com/interpretml/interpret/releases/tag/v0.1.5
[v0.1.4]: https://github.com/interpretml/interpret/releases/tag/v0.1.4
[v0.1.3]: https://github.com/interpretml/interpret/releases/tag/v0.1.3
[v0.1.2]: https://github.com/interpretml/interpret/releases/tag/v0.1.2
[v0.1.1]: https://github.com/interpretml/interpret/releases/tag/v0.1.1
[v0.1.0]: https://github.com/interpretml/interpret/releases/tag/v0.1.0

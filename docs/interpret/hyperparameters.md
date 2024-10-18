# Hyperparameters

Explainable Boosting Machines (EBMs) often have good performance using the default settings, however hyperparameter tuning can potentially improve model accuracy by a modest amount. The default parameters aim to balance computational efficiency with model accuracy. For some parameters we have a clear understanding of which direction they should be changed in order to improve the model. For these parameters, hyperparameter turning is not recommended and you should set them in accordance with how much time you can afford to fit the model.

The parameters below are ordered by tuning importance, with the most important hyperparameters to tune at the top, and the least important ones to tune at the bottom.


## smoothing_rounds
default: 100

hyperparameters: [0, 50, 100, 200, 500, 1000]

guidance: This is an important hyperparameter to tune. The optimal smoothing_rounds value will vary depending on the dataset's characteristics. Adjust based on the prevalence of smooth feature response curves.

## learning_rate
default: 0.015 (classification), 0.04 (regression)

hyperparameters: [0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]

guidance: This is an important hyperparameter to tune. The conventional wisdom is that a lower learning rate is generally better, but we have found the relationship to be more complex for EBMs. In general, regression seems to prefer a higher learning rate, binary classification seems to prefer a lower learning rate, and multiclass is in-between.

## interactions
default: 0.9

ideal: As many as possible

hyperparameters: [0.0, 0.9, 0.95, 0.99, 100, 250, 1000]

guidance: Introducing more interactions tends to improve model accuracy. Values between 0 and LESS than 1.0 are interpreted as percentages of the number of features. For example, a dataset with 100 features and an interactions value of 0.7 will automatically detect and use 70 interactions. Values of 1 or higher indicate the exact number of interactions to be detected, so for example 1 would create 1 interaction term, and 50 would create 50.

## inner_bags
default: 0

WARNING: Setting this value to 20 will typically increase the fitting time by a factor of 20x.

ideal: 20 (diminishing returns beyond this point)

hyperparameters: [0] OR if you can afford it [0, 20]

guidance: The default inner_bags value of 0 disables inner bagging. Setting this parameter to 1 or other low values will typically make the model worse since model fitting will then only use a subset of the data but not do enough inner bagging to compensate. Increasing the number of inner bags to 20 can improve model accuracy at the cost of significantly longer fitting times. If computation time is not a constraint, we suggest trying both 0 and 20, but not other values in between.

## max_bins
default: 1024

ideal: 1024 (diminishing returns beyond this point)

hyperparameters: [1024]

guidance: Increasing the max_bins value can enhance model accuracy by enabling finer discretization of features. Values above 1024 seem to result in very small changes to model performance, although there might be benefits for very large datasets. Setting max_bins to 1024 often provides a good balance between model performance, memory requirements, and fitting time.

## max_interaction_bins
default: 64

hyperparameters: [64]

guidance: For max_interaction_bins, more is typically better in term of model performance, however fitting times go up significantly above 64 bins for very little benefit. We recommend using 64 as the default for this reason. If your fitting times are acceptable however, setting max_interaction_bins to 256 or even more might improve the model slightly.

## greedy_ratio
default: 12.0

hyperparameters: [0.0, 1.0, 2.0, 5.0, 10.0, 12.0, 20.0]

guidance: Values of greedy_ratio above 5.0 seem to result in similar model performance.

## cyclic_progress
default: 0.0

hyperparameters: [0.0, 1.0]

guidance: Generally, turning off cyclic_progress by setting it to 0.0 is slightly better, although it can take more time to fit if greedy_ratio is close to 1.

## outer_bags
default: 14

ideal: 14 (diminishing returns beyond this point)

hyperparameters: [14]

guidance: Increasing outer bags beyond 14 provides no observable benefit. Reducing outer_bags below 14 might improve fitting time on machines with less than 14 cores. Setting outer_bags to 8 is reasonable on many datasets, and can improve fitting time.

## interaction_smoothing_rounds
default: 100

hyperparameters: [0, 50, 100, 200, 500, 1000]

guidance: interaction_smoothing_rounds appears to have only a minor impact on model accuracy. 100 is a good default choice, but it might be worth trying other values when optimizing a model.

## max_leaves
default: 2

hyperparameters: [2, 3]

guidance: Generally, the default setting of 2 is better, but it's worth checking if changing to 3 can offer better accuracy on your specific data. The max_leaves parameter only applies to main effects.

## min_samples_leaf
default: 4

hyperparameters: [2, 3, 4, 5, 10, 20, 50]

guidance: The default value usually works well, however experimenting with slightly higher values could potentially enhance generalization on certain datasets. For smaller datasets, having a low value might be better. On larger datasets this parameter seems to have little effect.

## min_hessian
default: 0.0

hyperparameters: [0.0, 1e-4]

guidance: Generally 0.0 is close to the best choice for min_hessian, but on some datasets it might be useful to set min_hessian to a small non-zero value.

## max_rounds
default: 25000

ideal: 1000000000 (early stopping should stop long before this point)

hyperparameters: [1000000000]

guidance: The max_rounds parameter serves as a limit to prevent excessive training on datasets where improvements taper off. Set this parameter sufficiently high to avoid premature early stopping provided fitting times are reasonable.

## early_stopping_rounds
default: 100

ideal: 200 (diminishing returns beyond this point)

hyperparameters: [100, 200]

guidance: Having 200 early_stopping_rounds results in a slightly better model than the default of 100, but it requires significantly more time to fit in some cases.  early_stopping_rounds beyond 200 does not seem to improve the model.

## early_stopping_tolerance
default: 0.0

hyperparameters: [0.0]

guidance: Setting early_stopping_tolerance to a small positive value in the range of 1e-4 can help reduce fitting times on some datasets with minimal degradation in model performance. Setting it to a negative value sometimes yields slightly better models. EBMs are a bagged ensemble model, so overfitting each individual bag a little can be beneficial because after the models are averaged together in the ensemble averaging decreases the variance due to overfitting. Using a negative value for early_stopping_tolerance allows the individual models to be overfit.

## validation_size
default: 0.15

hyperparameters: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

guidance: The ideal amount of data to be used as validation is dataset dependent, and should be tuned when possible.

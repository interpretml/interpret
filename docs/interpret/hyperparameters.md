# Hyperparameters

Explainable Boosting Machines (EBMs) are often robust with default settings, however hyperparameter tuning can potentially improve model accuracy by a modest amount. The default parameters aim to balance computational efficiency with model accuracy. For some parameters we have a clear understanding of which direction they should be changed in order to improve the model. For these parameters, hyperparameter turning is not recommended and you should set them in accordance with how much time you can afford to fit the model.

The parameters below are ordered by tuning importance, with the most important hyperparameters to tune at the top, and the least important ones to tune at the bottom.


## smoothing_rounds
default: 100

hyperparameters: [0, 50, 100, 200, 500, 1000]

guidance: This is an important hyperparameter to tune.  The optimal smoothing_rounds value will vary depending on the dataset's characteristics. Adjust based on the prevalence of smooth feature response curves.

## learning_rate
default: 0.01 (classification), 0.05 (regression)

hyperparameters: [0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025]

guidance: This is an important hyperparameter to tune.  The conventional wisdom is that a lower learning rate is generally better, but we have found the relationship to be more complex. In general, regression seems to prefer a higher learning rate, binary classification seems to prefer a lower learning rate, and multiclass is in-between.

## interactions
default: 0.9

hyperparameters: [0, 0.5, 0.75, 0.9, 0.95, 5, 10, 25, 50, 100, 250]

guidance: Introducing more interactions tends to improve model accuracy. Values between 0 and LESS than 1.0 are interpreted as percentages of the number of features. For example, a dataset with 100 features and an interactions value of 0.75 will automatically detect and use 75 interactions. Values of 1 or higher indicate the exact number of interactions to be detected, so for example 1 would create 1 interaction, and 50 would create 50.

## inner_bags
default: 0

WARNING: Setting this value to 50 will typically increase the fitting time by a factor of 50x.

ideal: 50 (diminishing returns beyond this point)

hyperparameters: [0] OR if you can afford it [0, 50]

guidance: The default inner_bags value of 0 disables inner bagging. Setting this parameter to 1 or other low values will typically make the model worse since model fitting will then only use a subset of the data but not do enough inner bagging to compensate. Increasing the number of inner bags to 50 can improve model accuracy at the cost of significantly longer training times. If computation time is not a constraint, we suggest trying both 0 and 50, but not other values in between.

## max_bins
default: 1024

hyperparameters: [256, 512, 1024, 4096, 16384, 65536]

guidance: Higher max_bins values can improve model accuracy by allowing more granular discretization of features. While the default minimizes memory consumption and speeds up training, we suggest testing larger values if resources permit.

## max_interaction_bins
default: 32

hyperparameters: [8, 16, 32, 64, 128, 256]

guidance: For max_interaction_bins, more is not necessarily better, unlike with max_bins. A good value on many datasets seems to be 32, but it's worth trying higher and lower values.

## greedy_ratio
default: 12.0

hyperparameters: [0.0, 1.0, 2.0, 5.0, 12.0, 20.0]

guidance: greedy_ratio is a good candidate for hyperparameter tuning as the best value is dataset dependent.

## cyclic_progress
default: 0.0

hyperparameters: [0.0, 1.0]

guidance: Try both.

## outer_bags
default: 14

ideal: 14 (diminishing returns beyond this point)

hyperparameters: [14]

guidance: Increasing outer bags beyond 14 provides no benefit.  Reducing outer_bags below 14 might improve fitting time on machines with less than 14 cores.

## interaction_smoothing_rounds
default: 50

hyperparameters: [0, 50, 100, 500]

guidance: interaction_smoothing_rounds appears to have only a minor impact on model accuracy. 0 is often the best choice.  0 is often the most accurate choice, but the interaction shape plots will be smoother and easier to interpret with more interaction_smoothing_rounds.

## max_leaves
default: 2

hyperparameters: [2, 3]

guidance: Generally, the default setting is effective, but it's worth checking if changing to 3 can offer better accuracy on your specific data. The max_leaves parameter only applies to main effects.

## min_samples_leaf
default: 4

hyperparameters: [2, 3, 4, 5, 6]

guidance: The default value usually works well, however experimenting with slightly higher values could potentially enhance generalization on certain datasets.

## min_hessian
default: 0.0

hyperparameters: [1e-4, 0.0]

guidance: Generally 0.0 is the best choice for min_hessian, but on some datasets it might be useful to set min_hessian.

## max_rounds
default: 25000

ideal: 1000000000 (early stopping should stop long before this point)

hyperparameters: [1000000000]

guidance: The max_rounds parameter serves as a limit to prevent excessive training on datasets where improvements taper off. Set this parameter sufficiently high to avoid premature early stopping. Consider increasing it if small yet consistent gains are observed in longer trainings.

## early_stopping_rounds
default: 200

guidance: We typically do not advise changing early_stopping_rounds. The default is appropriate for most cases, adequately capturing the optimal model without incurring unnecessary computational costs.

## early_stopping_tolerance
default: 0.0

hyperparameters: [0.0]

guidance: early_stopping_tolerance is set to 0.0 by default, however setting it to a negative value sometimes yields slightly higher accuracy. EBMs are a bagged ensemble model, so overfitting each individual bag a little can be beneficial because after the models are averaged together in the ensemble averaging decreases the variance due to overfitting. Using a negative value for early_stopping_tolerance allows the individual models to be overfit.

## validation_size
default: 0.15

hyperparameters: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

guidance: The ideal amount of data to be used as validation is dataset dependent, and should be tuned when possible.

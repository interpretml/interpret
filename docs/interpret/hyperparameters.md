# Hyperparameters

Explainable Boosting Machines (EBMs) are often robust with default settings, however hyperparameter tuning can potentially improve model accuracy by a modest amount. The default parameters aim to balance computational efficiency with model accuracy. For some parameters we have a clear understanding of which direction they should be changed in order to improve the model. For these parameters, hyperparameter turning is not recommended and you should set them in accordance with how much time you can afford to fit the model.

## max_bins
default: 1024

hyperparameters: [1024, 4096, 16384, 65536]

guidance: Higher max_bins values can improve model accuracy by allowing more granular discretization of features. While the default minimizes memory consumption and speeds up training, we suggest testing larger values if resources permit.

## max_interaction_bins
default: 32

hyperparameters: [8, 16, 32, 64, 128, 256]

guidance: For max_interaction_bins, more is not necessarily better, unlike with max_bins. A good value on many datasets seems to be 32, but it's worth trying higher and lower values.

## interactions
default: 0.95

hyperparameters: [0, 0.25, 0.5, 0.75, 0.95]

guidance: Introducing more interactions tends to improve model accuracy. Values between 0 and LESS than 1.0 are interpreted as percentages of the number of features. For example, a dataset with 100 features and an interactions value of 0.75 will automatically detect and use 75 interactions. Values of 1 or higher indicate the exact number of interactions to be detected, so for example 1 would create 1 interaction, and 50 would create 50.

## validation_size
default: 0.15

hyperparameters: [0.1, 0.15, 0.2]

guidance: The ideal amount of data to be used as validation is dataset dependent, and should be tuned when possible.

## outer_bags
default: 14

ideal: 50 (diminishing returns beyond this point)

guidance: We suggest increasing the number of outer bags if computational resources permit, ideally up to 50 outer bags where improvements plateau.

## inner_bags
default: 0

ideal: 50 (diminishing returns beyond this point)

hyperparameters: [0, 50]

WARNING: Setting this value to 50 will typically increase the fitting time by a factor of 50x.

guidance: The default inner_bags value of 0 disables inner bagging. Setting this parameter to 1 or other low values will typically make the model worse since model fitting will then only use a subset of the data. Increasing the number of inner bags to 50 can improve model accuracy at the cost of significantly longer training times. If computation time is not a constraint, we suggest trying both 0 and 50, but not other values in between.

## learning_rate
default: 0.01
hyperparameters: [0.02, 0.01, 0.005, 0.0025]

guidance: A smaller learning_rate promotes finer model adjustments during fitting, but may require more iterations. Generally, we believe a smaller learning_rate should improve the model, but sometimes hyperparameter tuning seems to be needed to select the best value.

## greedy_ratio
default: 1.5

hyperparameters: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 4.0]

guidance: greedy_ratio is a good candidate for hyperparameter tuning as the best value is dataset dependent.

## cyclic_progress
default: 1.0

hyperparameters: [0.0, 0.25, 0.5, 1.0]

guidance: cyclic_progress is a good candidate for hyperparameter tuning as the best value is dataset dependent.

## smoothing_rounds
default: 200

hyperparameters: [0, 50, 100, 200, 500, 1000, 2000, 4000]

guidance: The optimal smoothing_rounds value will vary depending on the dataset's characteristics. Adjust based on the prevalence of smooth feature response curves.

## interaction_smoothing_rounds
default: 50

hyperparameters: [0, 50, 100, 500]

guidance: interaction_smoothing_rounds appears to have only a minor impact on model accuracy. 0 is often the best choice.

## max_rounds
default: 25000

ideal: 1000000000 (early stopping should stop before this point)

guidance: The max_rounds parameter serves as a limit to prevent excessive training on datasets where improvements taper off. Set this parameter sufficiently high to avoid curtailing the early stopping mechanism. Consider increasing it if small yet consistent gains are observed in longer trainings.

## early_stopping_rounds
default: 50

guidance: We typically do not advise changing early_stopping_rounds. The default is appropriate for most cases, adequately capturing the optimal model without incurring unnecessary computational costs.

## early_stopping_tolerance
default: 0.0

guidance: Altering early_stopping_tolerance is not generally recommended. Increasing it may reduce fitting time at the expense of some accuracy.

## min_samples_leaf
default: 2

hyperparameters: [2, 3, 4]

guidance: The default value usually works well, however experimenting with slightly higher values could potentially enhance generalization on certain datasets.

## min_hessian
default: 0.0001

hyperparameters: [1.0, 0.01, 0.0001, 0.000001]

guidance: The default min_hessian is a solid starting point. While 1.0 may not be optimal in most scenarios, testing different thresholds could yield improvements depending on dataset characteristics.

## max_leaves
default: 3

hyperparameters: [3, 4]

guidance: Generally, the default setting is effective, but it's worth checking if an increment to 4 can offer better accuracy on your specific data.

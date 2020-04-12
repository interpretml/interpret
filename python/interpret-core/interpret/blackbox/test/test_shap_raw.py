import pandas as pd
import zipfile

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from interpret.blackbox import ShapKernel

from interpret.experimental import DataMapper
# TODO Need test for interpret explanation output is valid
# TODO move to interpret.experimental after resolving interpret outputs


def test_raw_features():
    outdirname = 'dataset.6.21.19'
    zipfilename = outdirname + '.zip'
    urlretrieve('https://publictestdatasets.blob.core.windows.net/data/' + zipfilename, zipfilename)
    with zipfile.ZipFile(zipfilename, 'r') as unzip:
        unzip.extractall('.')
    attritionData = pd.read_csv('./WA_Fn-UseC_-HR-Employee-Attrition.csv')

    # Dropping Employee count as all values are 1 and hence attrition is independent of this feature
    attritionData = attritionData.drop(['EmployeeCount'], axis=1)
    # Dropping Employee Number since it is merely an identifier
    attritionData = attritionData.drop(['EmployeeNumber'], axis=1)

    attritionData = attritionData.drop(['Over18'], axis=1)

    # Since all values are 80
    attritionData = attritionData.drop(['StandardHours'], axis=1)

    # Converting target variables from string to numerical values
    target_map = {'Yes': 1, 'No': 0}
    attritionData["Attrition_numerical"] = attritionData["Attrition"].apply(lambda x: target_map[x])
    target = attritionData["Attrition_numerical"]

    attritionXData = attritionData.drop(['Attrition_numerical', 'Attrition'], axis=1)

    # Split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(attritionXData,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        stratify=target)
    # Creating dummy columns for each categorical feature
    categorical = []
    for col, value in attritionXData.iteritems():
        if value.dtype == 'object':
            categorical.append(col)

    # Store the numerical columns in a list numerical
    numerical = attritionXData.columns.difference(categorical)

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical),
            ('cat', categorical_transformer, categorical)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', transformations),
                          ('classifier', SVC(C=1.0, probability=True, gamma='auto'))])
    clf.fit(x_train, y_train)

    data_mapper = DataMapper(transformations, allow_all_transformations=True)
    featurized_x_train = data_mapper.transform(x_train)
    explainer = ShapKernel(clf.steps[-1][1].predict_proba,
                           featurized_x_train)

    explanation = explainer.explain_local(featurized_x_train[:1], y_train[:1])

    raw_feature_explanation = data_mapper.keep_raw_features(explanation)

    assert raw_feature_explanation is not None
    assert raw_feature_explanation != explanation

    assert len(raw_feature_explanation.data(-1)["specific"][0]["mli"][0]["value"]["scores"][0]) == len(x_train.values[0])
    assert len(explanation.data(-1)["specific"][0]["names"]) == len(featurized_x_train[0])

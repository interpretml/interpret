import numpy as np
import pandas as pd
import urllib
from interpret.glassbox import (ExplainableBoostingRegressor,
                                ExplainableBoostingClassifier)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from copy import deepcopy
from json import load, loads, dump

from .. import get_model_data, get_sample_data, get_edited_model

SEED = 5122021

def train_ebm_classifier():
    
    df_house = pd.read_csv(
        'https://gist.githubusercontent.com/xiaohk/35c249349a02862fe7987c9ad9afbba7/raw/4ce9f96f7a0084dd413cc528b4e34c07b151395b/Iowa-house.csv')
    df_house.head()

    gamut_features = ['GarageArea', 'GrLivArea', 'BsmtFinSF1', 'WoodDeckSF', 'OpenPorchSF',
                    'PoolArea', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
                    'TotalBsmtSF', 'FirstFlrSF', 'SecondFlrSF', 'GarageCars', 'ThreeSsnPorch',
                    'KitchenAbvGr', 'EnclosedPorch', 'MSSubClass', 'YearRemodAdd',
                    'TotRmsAbvGrd', 'MasVnrArea', 'BsmtFinSF1', 'MiscVal',
                    'BsmtUnfSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath',
                    'FullBath', 'HalfBath', 'BedroomAbvGr', 'Fireplaces', 'ScreenPorch',
                    'GarageYrBlt', 'MoSold', 'YrSold', 'SalePrice']

    # Rename some features
    house_rename_dict = {
        '1stFlrSF': 'FirstFlrSF',
        '2ndFlrSF': 'SecondFlrSF',
        '3SsnPorch': 'ThreeSsnPorch'
    }

    new_columns = [house_rename_dict[l]
                if l in house_rename_dict else l for l in list(df_house.columns)]
    df_house.columns = new_columns

    features = df_house.columns

    # Set the feature type mapping

    numeric_feature_set = {
        'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'FirstFlrSF', 'SecondFlrSF',
        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
        'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', 'ThreeSsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
        'MoSold', 'YrSold', 'SalePrice'
    }

    feature_types = [
        'continuous' if f in numeric_feature_set else 'categorical' for f in features]

    # Missing value handling

    # for f in df_house.columns:
    #     if df_house[f].isnull().sum() > 0:
    #         print(f, f in numeric_feature_set, df_house[f].isnull().sum())


    def fillna(col):

        # Replace the na value to none if it is categorical data
        if col.name not in numeric_feature_set:
            col.fillna('None', inplace=True)

        # Replace the na value as 0 if it it s numerical data
        else:
            col.fillna(0, inplace=True)

        return col


    df_house = df_house.apply(lambda col: fillna(col))

    # Convert the price into a binary variable
    price_median = np.median(df_house.iloc[:, -1])
    df_house.iloc[:, -1] = list(map(int, df_house.iloc[:, -1] >= price_median))

    # Split the train/test dataset
    x_all = df_house.iloc[:, 1:-1].to_numpy()
    y_all = df_house.iloc[:, -1].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.3,
        random_state=SEED
    )


    # Train an EBM model

    top_10_interaction = [[10, 52], [42, 76], [17, 22], [25, 45], [
        18, 69], [17, 55], [28, 69], [19, 61], [36, 77], [3, 78]]
    my_interactions = top_10_interaction + [[45, 12], [45, 4]]

    ebm = ExplainableBoostingClassifier(
        feature_names=features[1: -1],
        feature_types=feature_types[1: -1],
        random_state=SEED,
        interactions=my_interactions,
        n_jobs=-1)

    ebm.fit(x_train, y_train)

    return ebm, x_train, x_test, y_train, y_test


def test_generate_model_data():
    ebm, _, _, _, _ = train_ebm_classifier()
    data = get_model_data(ebm)

    assert(len(data['features']) == 91)
    assert(len(data['labelEncoder']) == 46)
    assert(data['isClassifier'])


def test_generate_sample_data():
    ebm, _, x_test, _, y_test = train_ebm_classifier()
    data = get_sample_data(ebm, x_test, y_test)

    assert(np.array(data['samples']).shape[1] == 79)
    assert(len(data['labels']) == 438)
    assert(len(data['featureNames']) == 79)
    assert(len(data['featureTypes']) == 79)


def test_get_edited_model():
    ebm, _, x_test, _, y_test = train_ebm_classifier()

    y_pred = ebm.predict(x_test)
    org_accuracy = metrics.accuracy_score(y_test, y_pred)

    file_path = 'https://gist.githubusercontent.com/xiaohk/7931349b05c8d724371155e3477661d7/raw/b55761b0dec609897b0b2d2929c84513fddc0a8f/edit-7-26-2021.gamchanger'

    with urllib.request.urlopen(file_path) as url:
        gamchanger_export = loads(url.read().decode())
        gc_org_accuracy = gamchanger_export['historyList'][0]['metrics']['barData']['accuracy'][2]

        # The initial accuracy should match
        assert(np.abs(org_accuracy - gc_org_accuracy) <= 0.01)

        edited_model = get_edited_model(ebm, gamchanger_export)

        # The accuracy after model editing should match
        y_pred = edited_model.predict(x_test)
        new_accuracy = metrics.accuracy_score(y_test, y_pred)
        gc_new_accuracy = gamchanger_export['historyList'][-1]['metrics']['barData']['accuracy'][2]
        assert(np.abs(new_accuracy - gc_new_accuracy) <= 0.01)

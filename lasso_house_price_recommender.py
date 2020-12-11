# fastscore.schema.0: input_schema.avsc
# fastscore.schema.1: output_schema.avsc


import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings


# modelop.init
def begin():

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    global lasso_model, train_encoded_columns
    lasso_model = pickle.load(open("lasso_model.pickle", "rb"))
    train_encoded_columns = pickle.load(open("train_encoded_columns.pickle", "rb"))

    print("\ntype(lasso_model): ", type(lasso_model), flush=True)
    print("\ntype(train_encoded_columns): ", type(train_encoded_columns), flush=True)

    

# modelop.score
def action(data):
    
    print("\ntype(data): ", type(data), flush=True)

    data = pd.DataFrame(data)

    data_ID = data['Id']  # Saving the Id column then dropping it
    data.drop("Id", axis=1, inplace=True)

    # Imputations
    data['MasVnrType'] = data['MasVnrType'].fillna('None')
    data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
    data['Electrical'] = data['Electrical'].fillna('SBrkr')
    data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['YearBuilt'])

    data['MSZoning'] = data['MSZoning'].fillna('RL')
    data['Functional'] = data['Functional'].fillna('Typ')
    data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(0)
    data['BsmtFullBath'] = data['BsmtFullBath'].fillna(0)
    data['Utilities'] = data['Utilities'].fillna('AllPub')
    data['SaleType'] = data['SaleType'].fillna('WD')
    data['GarageArea'] = data['GarageArea'].fillna(0)
    data['GarageCars'] = data['GarageCars'].fillna(2)
    data['KitchenQual'] = data['KitchenQual'].fillna('TA')
    data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(0)
    data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(0)
    data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(0)
    data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(0)
    data['Exterior2nd'] = data['Exterior2nd'].fillna('VinylSd')
    data['Exterior1st'] = data['Exterior1st'].fillna('VinylSd')

    # Transform some features into catgeoricals
    data['MSSubClass']  = pd.Categorical(data.MSSubClass)
    data['YrSold']  = pd.Categorical(data.YrSold)
    data['MoSold']  = pd.Categorical(data.MoSold)
    
    print("\nshape: ", data.shape, flush=True)

    #  Computing total square-footage as a new feature
    data['TotalSF'] = data['TotalBsmtSF'] + data['firstFlrSF'] + data['secondFlrSF']

    #  Computing total 'porch' square-footage as a new feature
    data['Total_porch_sf'] = (
        data['OpenPorchSF'] + data['threeSsnPorch'] + data['EnclosedPorch'] 
        + data['ScreenPorch'] + data['WoodDeckSF']
    )

    #  Computing total bathrooms as a new feature
    data['Total_Bathrooms'] = (
        data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] 
        + (0.5 * data['BsmtHalfBath'])
    )

    # Engineering some features into Booleans
    f = lambda x: bool(1) if x > 0 else bool(0)

    data['has_pool'] = data['PoolArea'].apply(f)
    data['has_garage'] = data['GarageArea'].apply(f)
    data['has_bsmt'] = data['TotalBsmtSF'].apply(f)
    data['has_fireplace'] = data['Fireplaces'].apply(f)

    # Drop some unnecassary features
    data = data.drop(['threeSsnPorch', 'PoolArea', 'LowQualFinSF'], axis=1)
    
    print("\nshape before get dummies: ", data.shape, flush=True)

    cat_cols = [
        'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 
        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 
        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 
        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
        'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 
        'MiscFeature', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'has_pool', 
        'has_garage', 'has_bsmt', 'has_fireplace'
    ]

    encoded_features = pd.get_dummies(data, columns = cat_cols)
    
    print("\nshape after get dummies: ", encoded_features.shape)

    # Matching dummy variables from training set to current dummy variables
    missing_cols = set(train_encoded_columns) - set(encoded_features.columns)

    print("\nlen of Missing Cols: ", len(missing_cols), flush=True)
    for c in missing_cols:
        encoded_features[c] = 0

    # Matching order of variables to those used in training
    encoded_features = encoded_features[train_encoded_columns]
    
    print("\nshape of encoded_features: ", encoded_features.shape)

    # Computing predictions for each record
    log_predictions = lasso_model.predict(encoded_features)

    # Model was trained on log(SalePrice)
    adjusted_predictions = {}
    
    adjusted_predictions['Id'] = data_ID.astype(int)
    
    # actual predictions = exp(log_predictions)
    adjusted_predictions['predicted_sale_price'] = np.round(np.expm1(log_predictions), 2)

    # return an array of dictionaries such as
    # [{'Id': 1461, 'predicted_sale_price': 120429.01}, 
    #  {'Id': 1462, 'predicted_sale_price': 123970.31}]
    
    print("\noutput: ", pd.DataFrame(adjusted_predictions).to_dict(orient='records'), flush=True)

    yield pd.DataFrame(adjusted_predictions).to_dict(orient='records')


# modelop.metrics
def metrics(datum):

    print()
    print(type(datum), flush=True)
    print()
    print(datum.shape, flush=True)
    print()
    
    datum = pd.DataFrame([np.array(datum)[0][0]])

    adjusted_predictions = datum['predicted_sale_price']
    actuals = datum['actual_sale_price']

    # Mean squared error
    MSE = mean_squared_error(adjusted_predictions, actuals)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(adjusted_predictions, actuals)

    yield {
        "RMSE": np.round(RMSE, 2),
        "MAE": np.round(MAE, 2)
    }

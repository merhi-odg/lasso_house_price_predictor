# fastscore.schema.0: lasso_schema_in
# fastscore.slot.1: in-use

import pandas as pd
import pickle
import numpy as np

# modelop.init
def begin():
    global lasso_model, train_encoded_columns
    lasso_model = pickle.load(open("lasso_model.pickle", "rb"))
    train_encoded_columns = pickle.load(open("train_encoded_columns.pickle", "rb"))

# modelop.score
def action(data):
    
    print(type(data))
    data = pd.DataFrame([data])
    
    if 'SalePrice' in data.columns:  # Checking to see if data is labeled
        labeled=True
        actuals = data['SalePrice']  # Saving actuals (Sale prices)
        data.drop('SalePrice', axis=1, inplace=True)
    else:
        labeled=False
        
    print("Labeled: ", labeled)

    data_ID = data['Id']  # Saving the Id column
    data.drop("Id", axis=1, inplace=True)

    data['MasVnrType'] = data['MasVnrType'].fillna('None')
    data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
    data['Electrical'] = data['Electrical'].fillna('SBrkr')
    data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
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

    data['MSSubClass']  = pd.Categorical(data.MSSubClass)
    data['YrSold']  = pd.Categorical(data.YrSold)
    data['MoSold']  = pd.Categorical(data.MoSold)
    
    print("shape: ", data.shape)

    #  Computing total square-footage as a new feature
    data['TotalSF'] = data['TotalBsmtSF'] + data['firstFlrSF'] + data['secondFlrSF']

    #  Computing total 'porch' square-footage as a new feature
    data['Total_porch_sf'] = (data['OpenPorchSF'] + data['threeSsnPorch'] + data['EnclosedPorch'] 
                             + data['ScreenPorch'] + data['WoodDeckSF'])

    #  Computing total bathrooms as a new feature
    data['Total_Bathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) +
                                data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))


    # Engineering some features into Booleans
    f = lambda x: bool(1) if x > 0 else bool(0)

    data['has_pool'] = data['PoolArea'].apply(f)
    data['has_garage'] = data['GarageArea'].apply(f)
    data['has_bsmt'] = data['TotalBsmtSF'].apply(f)
    data['has_fireplace'] = data['Fireplaces'].apply(f)

    data = data.drop(['threeSsnPorch', 'PoolArea', 'LowQualFinSF'], axis=1)
    
    print("shape before get dummies: ", data.shape)
    cat_cols = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 
                'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
                'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 
                'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 
                'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 
                'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
                'MoSold', 'YrSold', 'SaleType','SaleCondition', 'has_pool', 'has_garage', 'has_bsmt', 'has_fireplace']

    encoded_features = pd.get_dummies(data, columns = cat_cols)
    
    print("shape after get dummies: ", encoded_features.shape)

    # Matching dummy variables from training set to current dummy variables
    missing_cols = set(train_encoded_columns) - set(encoded_features.columns)

    print("len of Missing Cols: ", len(missing_cols))
    for c in missing_cols:
        encoded_features[c] = 0

    # Matching order of variables to those used in training
    encoded_features = encoded_features[train_encoded_columns]
    
    print("shape Encoded Features: ", encoded_features.shape)

    models = {'Lasso': lasso_model}

    log_predictions = {}  # Model was trained on log(SalePrice)
    RMSEs = {}  # Root mean square error

    for name, model in models.items():
        log_predictions[name] = model.predict(encoded_features)  # Computing predictions for each model and each record

    adjusted_predictions = {}
    
    adjusted_predictions['ID'] = int(data_ID)
    
    for name, model in models.items():
        # actual predictions = exp(log_predictions)
        adjusted_predictions[name] = np.expm1(log_predictions[name])
        if labeled:
            #  Computing RMSE if actual data is available
            RMSEs[name] = np.sqrt(mean_squared_error(adjusted_predictions[name], actuals))

    # Use below with schema array_doubles
    out = np.round(np.array(pd.DataFrame(adjusted_predictions)),2).tolist() 

    # Use below with schema three_RMSEs
    #out = RMSEs
    yield out

# modelop.metrics
def metrics(datum):
    yield {"foo": 1}
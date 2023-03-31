import base64
import io

import pandas as pd
import streamlit as st
import requests
import os
import json
import datetime
import pickle


def read_css_file(file_path):
    """Read the contents of a CSS file."""
    with open(file_path) as css_file:
        return css_file.read()


def convert_image_to_base64(image):
    """Convert an image file to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def display_image_in_sidebar(image_base64, width, margin_bottom):
    """Display an image in the Streamlit sidebar with custom dimensions."""
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{image_base64}" style="width:{width}px; margin-bottom:{margin_bottom}px" />',
        unsafe_allow_html=True,
    )


import pickle

import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def load_model(model_name):
    """
    Load model and stats from the file saved
    """
    X_train, X_test, y_train, y_test = pickle.load(open('assets/X_train.pkl', 'rb')), pickle.load(
        open('assets/X_test.pkl', 'rb')), pickle.load(open('assets/y_train.pkl', 'rb')), pickle.load(
        open('assets/y_test.pkl', 'rb'))

    model_path = {
        'xgboost': 'assets/xgboost_model.pkl',
        'linear_regression': 'assets/linear_regression_model.pkl',
        'svr': 'assets/svr_model.pkl',
        'random_forest': 'assets/random_forest_model.pkl'
    }

    # Load model from file
    model = pickle.load(open(model_path[model_name], 'rb'))

    # Check if the model is an XGBoost model to retrieve feature names
    if model_name == 'xgboost':
        features = model.get_booster().feature_names
    # Check if the model is a linear model to retrieve feature names
    elif model_name == 'linear_regression':
        features = list(X_train.columns)
    # Check if the model is a tree-based model to retrieve feature names
    elif model_name in ['svr', 'random_forest']:
        features = list(X_train.columns[model.feature_importances_ != 0])
    else:
        features = None

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return model, features, mae, rmse, r2, mape, X_train, X_test, y_train, y_test


def estimate_house_price(model, data):
    price = model.predict(data)
    return price



def featureTransformation(feature):
    try:
        dict = {
            'MSSubClass': [
                ['20', '1-STORY 1946 & NEWER ALL STYLES'],
                ['30', '1-STORY 1945 & OLDER'],
                ['40', '1-STORY W/FINISHED ATTIC ALL AGES'],
                ['45', '1-1/2 STORY - UNFINISHED ALL AGES'],
                ['50', '1-1/2 STORY FINISHED ALL AGES'],
                ['60', '2-STORY 1946 & NEWER'],
                ['70', '2-STORY 1945 & OLDER'],
                ['75', '2-1/2 STORY ALL AGES'],
                ['80', 'SPLIT OR MULTI-LEVEL'],
                ['85', 'SPLIT FOYER'],
                ['90', 'DUPLEX - ALL STYLES AND AGES'],
                ['120', '1-STORY PUD (Planned Unit Development) - 1946 & NEWER'],
                ['150', '1-1/2 STORY PUD - ALL AGES'],
                ['160', '2-STORY PUD - 1946 & NEWER'],
                ['180', 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER'],
                ['190', '2 FAMILY CONVERSION - ALL STYLES AND AGES']],
            'MSZoning': [['A', 'Agriculture'],
                         ['C', 'Commercial'],
                         ['FV', 'Floating Village Residential'],
                         ['I', 'Industrial'],
                         ['RH', 'Residential High Density'],
                         ['RL', 'Residential Low Density'],
                         ['RP', 'Residential Low Density Park'],
                         ['RM', 'Residential Medium Density']
                         ],
            'Street': [['Grvl', 'Gravel'],
                       ['Pave', 'Paved']
                       ],
            'Alley': [['Grvl', 'Gravel'],
                      ['Pave', 'Paved'],
                      ['NA', 'No alley access']
                      ],
            'LotShape': [['Reg', 'Regular'],
                         ['IR1', 'Slightly irregular'],
                         ['IR2', 'Moderately Irregular'],
                         ['IR3', 'Irregular']
                         ],
            'LandContour': [['Lvl', 'Near Flat/Level'],
                            ['Bnk', 'Banked - Quick and significant rise from street grade to building'],
                            ['HLS', 'Hillside - Significant slope from side to side'],
                            ['Low', 'Depression']
                            ],
            'Utilities': [['AllPub', 'All public Utilities (E,G,W,& S)'],
                          ['NoSewr', 'Electricity, Gas, and Water (Septic Tank)'],
                          ['NoSeWa', 'Electricity and Gas Only'],
                          ['ELO', 'Electricity only']
                          ],
            'LotConfig': [['Inside', 'Inside lot'],
                          ['Corner', 'Corner lot'],
                          ['CulDSac', 'Cul-de-sac'],
                          ['FR2', 'Frontage on 2 sides of property'],
                          ['FR3', 'Frontage on 3 sides of property']
                          ],
            'LandSlope': [['Gtl', 'Gentle slope'],
                          ['Mod', 'Moderate Slope'],
                          ['Sev', 'Severe Slope']
                          ],
            'Neighborhood': [
                ['Blmngtn', 'Bloomington Heights'],
                ['Blueste', 'Bluestem'],
                ['BrDale', 'Briardale'],
                ['BrkSide', 'Brookside'],
                ['ClearCr', 'Clear Creek'],
                ['CollgCr', ' College Creek'],
                ['Crawfor', 'Crawford'],
                ['Edwards', 'Edwards'],
                ['Gilbert', 'Gilbert'],
                ['IDOTRR', 'Iowa DOT and Rail Road'],
                ['MeadowV', 'Meadow Village'],
                ['Mitchel', ' Mitchell'],
                ['Names', 'North Ames'],
                ['NoRidge', 'Northridge'],
                ['NPkVill', 'Northpark Villa'],
                ['NridgHt', 'Northridge Heights'],
                ['NWAmes', 'Northwest Ames'],
                ['OldTown', 'Old Town'],
                ['SWISU', 'South & West of Iowa State University'],
                ['Sawyer', 'Sawyer'],
                ['SawyerW', 'Sawyer West'],
                ['Somerst', 'Somerset'],
                ['StoneBr', 'Stone Brook'],
                ['Timber', 'Timberland'],
                ['Veenker', 'Veenker']
            ],
            "Condition1": [
                ["Artery", "Adjacent to arterial street"],
                ["Feedr", "Adjacent to feeder street"],
                ["Norm", "Normal"],
                ["RRNn", "Within 200' of North-South Railroad"],
                ["RRAn", "Adjacent to North-South Railroad"],
                ["PosN", "Near positive off-site feature--park, greenbelt, etc."],
                ["PosA", "Adjacent to postive off-site feature"],
                ["RRNe", "Within 200' of East-West Railroad"],
                ["RRAe", "Adjacent to East-West Railroad"]
            ],
            "Condition2": [
                ["Artery", "Adjacent to arterial street"],
                ["Feedr", "Adjacent to feeder street"],
                ["Norm", "Normal"],
                ["RRNn", "Within 200' of North-South Railroad"],
                ["RRAn", "Adjacent to North-South Railroad"],
                ["PosN", "Near positive off-site feature--park, greenbelt, etc."],
                ["PosA", "Adjacent to postive off-site feature"],
                ["RRNe", "Within 200' of East-West Railroad"],
                ["RRAe", "Adjacent to East-West Railroad"]
            ],
            "BldgType": [
                ["1Fam", "Single-family Detached"],
                ["2FmCon", "Two-family Conversion; originally built as one-family dwelling"],
                ["Duplx", "Duplex"],
                ["TwnhsE", "Townhouse End Unit"],
                ["TwnhsI", "Townhouse Inside Unit"]
            ],
            "HouseStyle": [
                ["1Story", "One story"],
                ["1.5Fin", "One and one-half story: 2nd level finished"],
                ["1.5Unf", "One and one-half story: 2nd level unfinished"],
                ["2Story", "Two story"],
                ["2.5Fin", "Two and one-half story: 2nd level finished"],
                ["2.5Unf", "Two and one-half story: 2nd level unfinished"],
                ["SFoyer", "Split Foyer"],
                ["SLvl", "Split Level"]
            ],
            "OverallQual": [
                ["10", "Very Excellent"],
                ["9", "Excellent"],
                ["8", "Very Good"],
                ["7", "Good"],
                ["6", "Above Average"],
                ["5", "Average"],
                ["4", "Below Average"],
                ["3", "Fair"],
                ["2", "Poor"],
                ["1", "Very Poor"]
            ],
            "OverallCond": [
                ["10", "Very Excellent"],
                ["9", "Excellent"],
                ["8", "Very Good"],
                ["7", "Good"],
                ["6", "Above Average"],
                ["5", "Average"],
                ["4", "Below Average"],
                ["3", "Fair"],
                ["2", "Poor"],
                ["1", "Very Poor"]
            ],
            "RoofStyle": [
                ["Flat", "Flat"],
                ["Gable", "Gable"],
                ["Gambrel", "Gabrel (Barn)"],
                ["Hip", "Hip"],
                ["Mansard", "Mansard"],
                ["Shed", "Shed"]
            ],
            "RoofMatl": [
                ["ClyTile", "Clay or Tile"],
                ["CompShg", "Standard (Composite) Shingle"],
                ["Membran", "Membrane"],
                ["Metal", "Metal"],
                ["Roll", "Roll"],
                ["Tar&Grv", "Gravel & Tar"],
                ["WdShake", "Wood Shakes"],
                ["WdShngl", "Wood Shingles"]
            ],
            "Exterior1st": [
                ["AsbShng", "Asbestos Shingles"],
                ["AsphShn", "Asphalt Shingles"],
                ["BrkComm", "Brick Common"],
                ["BrkFace", "Brick Face"],
                ["CBlock", "Cinder Block"],
                ["CemntBd", "Cement Board"],
                ["HdBoard", "Hard Board"],
                ["ImStucc", "Imitation Stucco"],
                ["MetalSd", "Metal Siding"],
                ["Other", "Other"],
                ["Plywood", "Plywood"],
                ["PreCast", "PreCast"],
                ["Stone", "Stone"],
                ["Stucco", "Stucco"],
                ["VinylSd", "Vinyl Siding"],
                ["Wd Sdng", "Wood Siding"],
                ["WdShing", "Wood Shingles"]
            ],
            "Exterior2nd": [
                ["AsbShng", "Asbestos Shingles"],
                ["AsphShn", "Asphalt Shingles"],
                ["BrkComm", "Brick Common"],
                ["BrkFace", "Brick Face"],
                ["CBlock", "Cinder Block"],
                ["CemntBd", "Cement Board"],
                ["HdBoard", "Hard Board"],
                ["ImStucc", "Imitation Stucco"],
                ["MetalSd", "Metal Siding"],
                ["Other", "Other"],
                ["Plywood", "Plywood"],
                ["PreCast", "PreCast"],
                ["Stone", "Stone"],
                ["Stucco", "Stucco"],
                ["VinylSd", "Vinyl Siding"],
                ["Wd Sdng", "Wood Siding"],
                ["WdShing", "Wood Shingles"]
            ],
            "MasVnrType": [
                ["BrkCmn", "Brick Common"],
                ["BrkFace", "Brick Face"],
                ["CBlock", "Cinder Block"],
                ["None", "None"],
                ["Stone", "Stone"]
            ],
            "ExterQual": [
                ["Ex", "Excellent"],
                ["Gd", "Good"],
                ["TA", "Average/Typical"],
                ["Fa", "Fair"],
                ["Po", "Poor"]
            ],
            "ExterCond": [
                ["Ex", "Excellent"],
                ["Gd", "Good"],
                ["TA", "Average/Typical"],
                ["Fa", "Fair"],
                ["Po", "Poor"]
            ],
            "Foundation": [
                ["BrkTil", "Brick & Tile"],
                ["CBlock", "Cinder Block"],
                ["PConc", "Poured Contrete"],
                ["Slab", "Slab"],
                ["Stone", "Stone"],
                ["Wood", "Wood"]
            ],
            "BsmtQual": [
                ["Ex", "Excellent (100+ inches)"],
                ["Gd", "Good (90-99 inches)"],
                ["TA", "Typical (80-89 inches)"],
                ["Fa", "Fair (70-79 inches)"],
                ["Po", "Poor (<70 inches)"],
                ["NA", "No Basement"]
            ],
            "BsmtCond": [
                ["Ex", "Excellent"],
                ["Gd", "Good"],
                ["TA", "Typical - slight dampness allowed"],
                ["Fa", "Fair - dampness or some cracking or settling"],
                ["Po", "Poor - Severe cracking, settling, or wetness"],
                ["NA", "No Basement"]
            ],
            "BsmtExposure": [
                ["Gd", "Good Exposure"],
                ["Av", "Average Exposure (split levels or foyers typically score average or above)"],
                ["Mn", "Mimimum Exposure"],
                ["No", "No Exposure"],
                ["NA", "No Basement"]
            ],
            "BsmtFinType1": [
                ["GLQ", "Good Living Quarters"],
                ["ALQ", "Average Living Quarters"],
                ["BLQ", "Below Average Living Quarters"],
                ["Rec", "Average Rec Room"],
                ["LwQ", "Low Quality"],
                ["Unf", "Unfinshed"],
                ["NA", "No Basement"]
            ],
            "BsmtFinType2": [
                ["GLQ", "Good Living Quarters"],
                ["ALQ", "Average Living Quarters"],
                ["BLQ", "Below Average Living Quarters"],
                ["Rec", "Average Rec Room"],
                ["LwQ", "Low Quality"],
                ["Unf", "Unfinshed"],
                ["NA", "No Basement"]
            ],
            "Heating": [
                ["Floor", "Floor Furnace"],
                ["GasA", "Gas forced warm air furnace"],
                ["GasW", "Gas hot water or steam heat"],
                ["Grav", "Gravity furnace"],
                ["OthW", "Hot water or steam heat other than gas"],
                ["Wall", "Wall furnace"]
            ],
            "HeatingQC": [
                ["Ex", "Excellent"],
                ["Gd", "Good"],
                ["TA", "Average/Typical"],
                ["Fa", "Fair"],
                ["Po", "Poor"]
            ],
            "CentralAir": [
                ["N", "No"],
                ["Y", "Yes"]
            ],
            "Electrical": [
                ["SBrkr", "Standard Circuit Breakers & Romex"],
                ["FuseA", "Fuse Box over 60 AMP and all Romex wiring (Average)"],
                ["FuseF", "60 AMP Fuse Box and mostly Romex wiring (Fair)"],
                ["FuseP", "60 AMP Fuse Box and mostly knob & tube wiring (poor)"],
                ["Mix", "Mixed"]
            ],
            "KitchenQual": [
                ["Ex", "Excellent"],
                ["Gd", "Good"],
                ["TA", "Typical/Average"],
                ["Fa", "Fair"],
                ["Po", "Poor"]
            ],
            'Functional': [
                ['Typ', 'Typical Functionality'],
                ['Min1', 'Minor Deductions 1'],
                ['Min2', 'Minor Deductions 2'],
                ['Mod', 'Moderate Deductions'],
                ['Maj1', 'Major Deductions 1'],
                ['Maj2', 'Major Deductions 2'],
                ['Sev', 'Severely Damaged'],
                ['Sal', 'Salvage only']
            ],
            'FireplaceQu': [
                ['Ex', 'Excellent - Exceptional Masonry Fireplace'],
                ['Gd', 'Good - Masonry Fireplace in main level'],
                ['TA', 'Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement'],
                ['Fa', 'Fair - Prefabricated Fireplace in basement'],
                ['Po', 'Poor - Ben Franklin Stove'],
                ['NA', 'No Fireplace']
            ],
            'GarageType': [
                ['2Types', 'More than one type of garage'],
                ['Attchd', 'Attached to home'],
                ['Basment', 'Basement Garage'],
                ['BuiltIn', 'Built-In (Garage part of house - typically has room above garage)'],
                ['CarPort', 'Car Port'],
                ['Detchd', 'Detached from home'],
                ['NA', 'No Garage']
            ],
            'GarageFinish': [
                ['Fin', 'Finished'],
                ['RFn', 'Rough Finished'],
                ['Unf', 'Unfinished'],
                ['NA', 'No Garage']
            ],
            'GarageQual': [
                ['Ex', 'Excellent'],
                ['Gd', 'Good'],
                ['TA', 'Typical/Average'],
                ['Fa', 'Fair'],
                ['Po', 'Poor'],
                ['NA', 'No Garage']
            ],
            'GarageCond': [
                ['Ex', 'Excellent'],
                ['Gd', 'Good'],
                ['TA', 'Typical/Average'],
                ['Fa', 'Fair'],
                ['Po', 'Poor'],
                ['NA', 'No Garage']
            ],
            'PavedDrive': [
                ['Y', 'Paved'],
                ['P', 'Partial Pavement'],
                ['N', 'Dirt/Gravel']
            ],
            'PoolQC': [
                ['Ex', 'Excellent'],
                ['Gd', 'Good'],
                ['TA', 'Average/Typical'],
                ['Fa', 'Fair'],
                ['NA', 'No Pool']
            ],
            'Fence': [
                ['GdPrv', 'Good Privacy'],
                ['MnPrv', 'Minimum Privacy'],
                ['GdWo', 'Good Wood'],
                ['MnWw', 'Minimum Wood/Wire'],
                ['NA', 'No Fence']
            ],
            'MiscFeature': [
                ['Elev', 'Elevator'],
                ['Gar2', '2nd Garage (if not described in garage section)'],
                ['Othr', 'Other'],
                ['Shed', 'Shed (over 100 SF)'],
                ['TenC', 'Tennis Court'],
                ['NA', 'None']
            ],
            'SaleType': [
                ['WD', 'Warranty Deed - Conventional'],
                ['CWD', 'Warranty Deed - Cash'],
                ['VWD', 'Warranty Deed - VA Loan'],
                ['New', 'Home just constructed and sold'],
                ['COD', 'Court Officer Deed/Estate'],
                ['Con', 'Contract 15% Down payment regular terms'],
                ['ConLw', 'Contract Low Down payment and low interest'],
                ['ConLI', 'Contract Low Interest'],
                ['ConLD', 'Contract Low Down'],
                ['Oth', 'Other']
            ],
            'SaleCondition': [
                ['Normal', 'Normal Sale'],
                ['Abnorml', 'Abnormal Sale - trade, foreclosure, short sale'],
                ['AdjLand', 'Adjoining Land Purchase'],
                ['Alloca',
                 'Allocation - two linked properties with separate deeds, typically condo with a garage unit'],
                ['Family', 'Sale between family members'],
                ['Partial', 'Home was not completed when last assessed (associated with New Homes)']
            ]
        }
        model_element_list = []
        full_element_list = []
        for i in range(0, len(dict[feature])):
            model_element_list.append(dict[feature][i][0])
            full_element_list.append(dict[feature][i][1])
        return "categorical", model_element_list, full_element_list
    except BaseException:
        return "numerical", [], []


def getGoodName(feature):
    dict = {
        "OverallQual": "Overall Quality",
        "GrLivArea": "Living Area",
        "ExterQual": "Exterior Quality",
        "GarageCars": "Garage Cars Capacity",
        "BsmtQual": "Basement Height Quality",
        "GarageArea": "Garage Size in Square Feet",
        "TotalBsmtSF": "Total Basement Surface Area ",
        "1stFlrSF": "First Floor Surface Area ",
        "ExterQual_TA": "Average Exterior Quality",
        "KitchenQual": "Kitchen Quality",
        "FullBath": "Number of Full Bathrooms",
        "AgeOfHouse": "Age of House at Time of Sale",
        "YearBuilt": "Original Construction Date",
        "TotRmsAbvGrd": "Total Rooms",
        "KitchenQual_TA": "Average Kitchen Quality",
        "AgeOfRemodel": "Age of Remodel",
        "YearRemodAdd": "Remodel Date ",
        "Foundation_PConc": "Poured Concrete Foundation",
        "FireplaceQu_None": "No Fireplace",
        "Fireplaces": "Number of Fireplaces",
        "FireplaceQu": "Fireplace Quality",
        "ExterQual_Gd": "Good Exterior Quality",
        "BsmtQual_TA": "Average Basement Height Quality",
        "MasVnrArea": "Masonry Veneer Area ",
        "Neighborhood_NridgHt": "Northridge Heights Neighborhood",
        "BsmtFinType1_GLQ": "Good Living Quarters Basement Finish Type 1",
        "GarageFinish_Unf": "Unfinished Garage Interior",
        "HeatingQC": "Heating Quality and Condition",
        "BsmtFinSF1": "Type 1 Finished Square Feet",
        "FireplaceQu_Gd": "Good Fireplace Quality",
        "SaleType_New": "Type of sale",
        "SaleCondition_Partial": "Sale Cndition",
        "GarageType_Detchd": "Detached Garage Type"
    }
    return dict[feature]


def input_to_dataframe(input_dict):
    return pd.DataFrame(input_dict, index=[0])



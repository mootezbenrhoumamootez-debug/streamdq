import pandas as pd
import numpy as np
import joblib

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

SMALL_CAT_COLS = [
    'Broker_Agency_Type', 'Payment_Schedule', 'Employment_Status',
    'Deductible_Tier', 'Acquisition_Channel'
]

HIGH_CARD_COL = 'Region_Code'
REFERENCE_DATE = pd.to_datetime('2020-08-01')


def preprocess(df):
    df = df.copy()

    df['Policy_Start_Month_Num'] = df['Policy_Start_Month'].map(MONTH_MAP)

    for col in ['Policy_Start_Day', 'Policy_Start_Month_Num', 'Policy_Start_Year']:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['Policy_Start_Date'] = pd.to_datetime(
        df['Policy_Start_Year'].astype(int).astype(str) + '-' +
        df['Policy_Start_Month_Num'].astype(int).astype(str) + '-' +
        df['Policy_Start_Day'].astype(int).astype(str),
        errors='coerce'
    )

    df['Policy_Start_DayOfMonth'] = df['Policy_Start_Date'].dt.day
    df['Policy_Start_Month']      = df['Policy_Start_Date'].dt.month
    df['Policy_Start_Weekday']    = df['Policy_Start_Date'].dt.weekday
    df['Policy_Age_Days']         = (REFERENCE_DATE - df['Policy_Start_Date']).dt.days
    df['Days_Since_Policy_Start'] = df['Policy_Age_Days']

    df.drop(columns=['Policy_Start_Date', 'Policy_Start_Month_Num'], inplace=True, errors='ignore')

    df['Child_Dependents'] = df['Child_Dependents'].fillna(df['Child_Dependents'].median())
    df[HIGH_CARD_COL] = df[HIGH_CARD_COL].fillna(-1)

    df = pd.get_dummies(df, columns=SMALL_CAT_COLS, prefix=SMALL_CAT_COLS)
    df[HIGH_CARD_COL] = df[HIGH_CARD_COL].astype('category')

    return df


def load_model():
    model = None
    # ------------------ MODEL LOADING LOGIC ------------------
    model = joblib.load('model.pkl')
    # ------------------ END MODEL LOADING LOGIC ------------------
    return model


def predict(df, model):
    predictions = None
    # ------------------ PREDICTION LOGIC ------------------
    user_ids = df['User_ID']
    X = df.drop(columns=['User_ID'], errors='ignore')

    expected_cols = model.feature_names_
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_cols]

    if HIGH_CARD_COL in X.columns:
        X[HIGH_CARD_COL] = X[HIGH_CARD_COL].astype('category')

    preds = model.predict(X).flatten().astype(int)

    predictions = pd.DataFrame({
        'User_ID': user_ids.values,
        'Purchased_Coverage_Bundle': preds
    })
    # ------------------ END PREDICTION LOGIC ------------------
    return predictions
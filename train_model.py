#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from xgboost import XGBRegressor
from category_encoders import TargetEncoder

from custom_transformers import (
    RollingStatefulTransformer, DateFeatureTransformer, InteractionTransformer,
    ensure_categorical,drop_unnecessary_columns
)



# 1. Load Data
df = pd.read_csv('TRAIN.csv').sort_values('Date')
df = df[df['Sales'] != 0]
df = df.reset_index(drop=True)

# Convert Store_id to string from the start
df['Store_id'] = df['Store_id'].astype(str)


# Keep Sales in X initially for the Rolling Transformer
X = df.drop(columns=['#Order', 'ID'])
y = df['Sales']

# 2. Preprocessing
categorical_cols = ['Store_Type', 'Location_Type', 'Region_Code',
                    'Store_Location_Type', 'Holiday_Discount']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore',
         sparse_output=False), categorical_cols),
        ('target_store', TargetEncoder(cols=['Store_id']), ['Store_id'])
    ],
    remainder='passthrough'
)

# 3. Model
xgb = XGBRegressor(objective='reg:absoluteerror',
                   tree_method='hist', random_state=42)
model_wrapper = TransformedTargetRegressor(
    regressor=xgb, func=np.log1p, inverse_func=np.expm1)

# 4. Pipeline
pipeline = Pipeline([
    ('ensure_cat', FunctionTransformer(ensure_categorical)),
    ('interactions', InteractionTransformer()),
    ('rolling', RollingStatefulTransformer(window_sizes=[7, 30])),
    ('date', DateFeatureTransformer()),
    ('drop_cols', FunctionTransformer(drop_unnecessary_columns)),  # Sales dropped here
    ('preprocessor', preprocessor),
    ('regressor', model_wrapper)
])

# 5. Search
param_dist = {
    'regressor__regressor__n_estimators': [500, 1000],
    'regressor__regressor__max_depth': [3, 6, 10],
    'regressor__regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__regressor__subsample': [0.7, 0.8],
    'regressor__regressor__colsample_bytree': [0.7, 0.8]
}

tscv = TimeSeriesSplit(n_splits=5)
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=5,
    cv=tscv,
    scoring='neg_mean_absolute_percentage_error',
    n_jobs=-1,
    verbose=1,
    error_score='raise'
)

print("Training optimized model...")
search.fit(X, y)

# 6. Save and Inspect
best_pipeline = search.best_estimator_
joblib.dump(best_pipeline, 'sales_model.joblib')
print(f"Best MAPE: {-search.best_score_ * 100:.2f}%")

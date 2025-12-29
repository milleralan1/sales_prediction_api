#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import joblib
from custom_transformers import (
    RollingStatefulTransformer, DateFeatureTransformer, InteractionTransformer,
    ensure_categorical, drop_unnecessary_columns
)


# Load test data
test_df = pd.read_csv('TEST_FINAL.csv')

# Convert Store_id to string (must match training)
test_df['Store_id'] = test_df['Store_id'].astype(str)

# Load trained model
print("Loading model...")
model = joblib.load('sales_model.joblib')

# Make predictions
print("Making predictions...")
preds = model.predict(test_df)

# Save results
test_df['Sales_Predicted'] = preds

test_df.to_csv('submission.csv', index=False)
print(f"Predictions saved to submission.csv ({len(test_df)} rows)")

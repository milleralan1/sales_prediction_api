import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def drop_unnecessary_columns(X):
    cols_to_drop = ['Date', 'ID', '#Order', 'Sales']
    return X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')


# Ensure Store_id is treated as categorical
def ensure_categorical(X):
    X = X.copy()
    X['Store_id'] = X['Store_id'].astype(str)
    return X



class InteractionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        if 'Discount' in X.columns and X['Discount'].dtype == 'object':
            X['Discount'] = X['Discount'].map({'Yes': 1, 'No': 0}).fillna(0)
        X['discount_and_holiday'] = (X['Discount'] & X['Holiday']).astype(int)
        X['Store_Location_Type'] = X['Store_Type'].astype(
            str) + "_" + X['Location_Type'].astype(str)
        X['Holiday_Discount'] = X['Holiday'].astype(
            str) + "_" + X['Discount'].astype(str)
        return X


class RollingStatefulTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_sizes=[7, 30], group_col='Store_id', target_cols=['Sales']):
        self.window_sizes = window_sizes
        self.group_col = group_col
        self.target_cols = target_cols
        self.train_tails_ = {}

    def fit(self, X, y=None):
        temp_df = X.copy()
        # If y is provided separately (standard sklearn fit), put it into temp_df
        if y is not None:
            for col in self.target_cols:
                # Use .values to ensure alignment if X has been sliced/reindexed
                temp_df[col] = y.values if hasattr(y, 'values') else y

        max_w = max(self.window_sizes) + 14
        for store_id, group in temp_df.groupby(self.group_col):
            self.train_tails_[store_id] = group.tail(max_w)
        return self

    def transform(self, X):
        X_out = X.copy()
        # Keep track of the original index to return later
        original_index = X.index

        all_results = []
        for store_id, group in X_out.groupby(self.group_col):
            # Preserve the original index of this group
            group_index = group.index

            history = self.train_tails_.get(store_id)
            if history is not None:
                # Reset index on history to avoid duplicate indices
                history_reset = history.reset_index(drop=True)
                group_reset = group.reset_index(drop=True)
                
                combined = pd.concat([history_reset, group_reset], axis=0).sort_values('Date')
                
                for col in self.target_cols:
                    if col in combined.columns:
                        combined[f'{col}_lag_7'] = combined[col].shift(7)
                        combined[f'{col}_lag_14'] = combined[col].shift(14)
                        for w in self.window_sizes:
                            combined[f'{col}_roll_mean_{w}'] = combined[col].shift(1).rolling(w).mean()

                # Get only the rows that correspond to the current group (last len(group) rows)
                res = combined.tail(len(group)).copy()
                # Restore the original index
                res.index = group_index
                all_results.append(res)
            else:
                for col in self.target_cols:
                    group[f'{col}_lag_7'] = np.nan
                    group[f'{col}_lag_14'] = np.nan
                    for w in self.window_sizes:
                        group[f'{col}_roll_mean_{w}'] = np.nan
                all_results.append(group)

        transformed_df = pd.concat(all_results)
        # Reindex to match the original order
        return transformed_df.reindex(original_index)


class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        dt = pd.to_datetime(X['Date'])
        X['Day'] = dt.dt.day
        X['DayOfWeek'] = dt.dt.dayofweek
        X['Month'] = dt.dt.month
        X['is_month_start'] = (X['Day'] <= 3).astype(int)
        X['is_month_end'] = (X['Day'] >= 28).astype(int)
        X['is_payday'] = ((X['Day'] == 1) | (X['Day'] == 15)).astype(int)
        X['month_sin'] = np.sin(2 * np.pi * X['Month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['Month'] / 12)
        return X

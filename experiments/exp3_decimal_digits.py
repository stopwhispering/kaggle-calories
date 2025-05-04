import time
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error
import numpy as np
from calories.constants import PATH_TRAIN, PATH_TEST, PATH_PREDS_FOR_ENSEMBLES, PATH_CACHE
from calories.preprocessing.dtypes import convert_sex
from trainers.lgbm_trainer import LGBMTrainer
from trainers.metrics.metrics_lgbm import rmsle_lgbm
from trainers.xgb_trainer import XGBTrainer
import optuna

time_start_overall = time.time()

decimal_cols = ['Body_Temp', 'Height', 'Heart_Rate']

df_train = (
    pd.read_csv(PATH_TRAIN, dtype={c: str for c in decimal_cols}).set_index("id").drop("Calories", axis=1)
)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
df_test = pd.read_csv(PATH_TEST, dtype={c: str for c in decimal_cols}).set_index("id")


def count_decimal_digits(s):
    if pd.isna(s):
        return None
    if '.' in s:
        return len(s.split('.')[1])
    return 0


for col in decimal_cols:
    colname = 'decimal_digits_' + col
    df_train[colname] = df_train[col].apply(count_decimal_digits)
    df_test[colname] = df_test[col].apply(count_decimal_digits)

a = 1
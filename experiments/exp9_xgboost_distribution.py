import time
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error
import numpy as np
from sklearn.model_selection import KFold
from xgboost_distribution import XGBDistribution
import os
import pickle
from calories.constants import (
    PATH_TRAIN,
    PATH_TEST,
    PATH_PREDS_FOR_ENSEMBLES,
    PATH_FEATURES,
    PATH_FEATURES_BY_FOLD, PATH_CACHE,
)
from calories.preprocessing.dtypes import convert_sex
from feature_creators.read_features_util import read_features
from my_ml_util.trainers.catboost_trainer import CatBoostTrainer
from my_preprocessing.feature_store.feature_store import FeatureStore
from my_ml_util.trainers.lgbm_trainer import LGBMTrainer
from my_ml_util.trainers.metrics.metrics_lgbm import rmsle_lgbm
from my_ml_util.trainers.xgb_trainer import XGBTrainer

time_start_overall = time.time()
df_train = pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
df_test = pd.read_csv(PATH_TEST).set_index("id")

df_train = convert_sex(df_train)
df_test = convert_sex(df_test)
print(f"Competition Features: {df_train.shape=}, {df_test.shape=}")

print(f"{df_train.shape=}, {df_test.shape=}")
duration_loading_data = str(int(time.time() - time_start_overall))
print(f"{duration_loading_data=}")
time_start_training = time.time()

arr_oof_predictions_mean = np.zeros(len(ser_targets_train))
arr_oof_predictions_std = np.zeros(len(ser_targets_train))
arr_test_predictions_mean = np.zeros(len(df_test))
arr_test_predictions_std = np.zeros(len(df_test))

best_iterations = []
splits = KFold(n_splits=5, shuffle=True, random_state=42).split(df_train)
for fold_no, (train_idx, val_idx) in enumerate(splits):
    df_train_fold = df_train.iloc[train_idx]
    df_val_fold = df_train.iloc[val_idx]
    ser_targets_train_fold: pd.Series = ser_targets_train.iloc[
        train_idx
    ]
    ser_targets_val_fold: pd.Series = ser_targets_train.iloc[
        val_idx
    ]

    model = XGBDistribution(
        distribution="normal",
        # n_estimators=500,
        device="cuda",
        tree_method="hist",
        # objective="reg:squarederror",
        # early_stopping_rounds=10

        random_state=42,
        verbosity=0,
        n_estimators=50_000,
        early_stopping_rounds=50,
    )
    model.fit(df_train_fold, ser_targets_train_fold,
              eval_set=[(df_val_fold, ser_targets_val_fold)],)
    best_iterations.append(model.best_iteration)
    preds_val = model.predict(df_val_fold)
    mean, std = preds_val.loc, preds_val.scale  # noqa
    arr_oof_predictions_mean[val_idx] = mean
    arr_oof_predictions_std[val_idx] = std

    preds_test = model.predict(df_test)
    arr_test_predictions_mean += preds_test.loc  # noqa
    arr_test_predictions_std += preds_test.scale  # noqa

arr_test_predictions_mean /= 5
arr_test_predictions_std /= 5

arr_oof_predictions_mean = np.clip(arr_oof_predictions_mean, 1, 314)

score = root_mean_squared_log_error(
    ser_targets_train,
    arr_oof_predictions_mean,
)

df_oof_predictions = pd.DataFrame(
    {
        # "id": df_train.index,
        "pred": arr_oof_predictions_mean,
        "pred_std": arr_oof_predictions_std,
    },
    index=df_train.index,
)

df_test_predictions_from_oof = pd.DataFrame(
    {
        # "id": df_test.index,
        "pred": arr_test_predictions_mean,
        "pred_std": arr_test_predictions_std,
    },
    index=df_test.index,
)

df_oof_predictions.to_parquet(PATH_CACHE / "exp9_probability_estimates_oof_predictions.parquet")
df_test_predictions_from_oof.to_parquet(PATH_CACHE / "exp9_probability_estimates_test_predictions.parquet")

duration_training = time.time() - time_start_training
print(f"{score=:.5f}, {best_iterations=}, {duration_training=}")


ser_train = df_oof_predictions["pred_std"]
ser_test = df_test_predictions_from_oof["pred_std"]

ser_train.name = "exp9_std"
ser_test.name = "exp9_std"

assert ser_train.name == ser_test.name
filename_prefix = "probability_pred_std"
metadata = {
    "column_name": ser_train.name,
    "filename_prefix": filename_prefix,
    "type": "probability estimates std",
    "subtype": "probability estimates std",
    "description": f"probability estimates std from exp9",
    # "feature_list": feature_list,
    # "agg": "mean",
    # "outer_folds": N_OUTER_FOLDS,
    # "inner_folds": N_INNER_FOLDS,
    # "smooth": SMOOTH,
    # "n_bins": n_bins,
    # "strategy": strategy,
    # "nan_quota": nan_quota,
    "dtype": str(ser_train.dtype),
    "created_at": pd.Timestamp.now(),
    "created_by_script": os.path.basename(__file__),
    "fillna": False,
}

if (PATH_FEATURES / f"{filename_prefix}_metadata.txt").exists():
    print(f"WARNING: {filename_prefix} already exists. Overwriting now.")

# save metadata to flat text file, with line breaks after each key; also pickle it
with open(PATH_FEATURES / f"{filename_prefix}_metadata.txt", "w") as f:
    for key, value in metadata.items():
        f.write(f"{key}: {value}\n")
with open(PATH_FEATURES / f"{filename_prefix}_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)  # noqa

ser_train.to_frame().to_parquet(PATH_FEATURES / f"{filename_prefix}_train.parquet")
ser_test.to_frame().to_parquet(PATH_FEATURES / f"{filename_prefix}_test.parquet")








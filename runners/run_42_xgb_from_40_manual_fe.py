import time
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error
import numpy as np
from calories.constants import (
    PATH_TRAIN,
    PATH_TEST,
    PATH_PREDS_FOR_ENSEMBLES,
    PATH_FEATURES,
    PATH_FEATURES_BY_FOLD,
)
from calories.preprocessing.dtypes import convert_sex
from feature_creators.read_features_util import read_features
from my_preprocessing.feature_store.feature_store import FeatureStore
from trainers.lgbm_trainer import LGBMTrainer
from trainers.metrics.metrics_lgbm import rmsle_lgbm
from trainers.xgb_trainer import XGBTrainer

time_start_overall = time.time()
df_train = pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
df_test = pd.read_csv(PATH_TEST).set_index("id")

df_train = convert_sex(df_train)
df_test = convert_sex(df_test)
print(f"Competition Features: {df_train.shape=}, {df_test.shape=}")


feature_store = FeatureStore(
    path_features=PATH_FEATURES,
    path_features_by_fold=PATH_FEATURES_BY_FOLD,
)
df_train_features, df_test_features = feature_store.read_features(
    column_names=[
        "Combine_Sex_Duration",
        "Multiply_Weight_Duration",
        "Plus_Age_Duration",
        "Multiply_Age_Duration",
        "GroupByThenMean_Age_Height",
        "Divide_Sex_Age",
        "GroupByThenRank_Sex_Weight",
        "Minus_Sex_Heart_Rate",
        "Multiply_Duration_Body_Temp",
        # fs3
        "te_MEAN_Sex_Age_Body_Temp_not_folded_no_fillna",
        "te_MEAN_Age_Height_Duration_Body_Temp_not_folded_no_fillna",
        "te_MEAN_Age_Weight_Duration_Heart_Rate_not_folded_no_fillna",
        "te_MEAN_Age_Height_Weight_Heart_Rate_Body_Temp_not_folded_no_fillna",
        "te_MEAN_Sex_Age_Height_Weight_Heart_Rate_Body_Temp_not_folded_no_fillna",
        "te_MEAN_Age_Height_Weight_Duration_Heart_Rate_not_folded_no_fillna",
    ]
)
df_train = pd.concat([df_train, df_train_features], axis=1)
df_test = pd.concat([df_test, df_test_features], axis=1)
print(f"After adding AutoFE: {df_train.shape=}, {df_test.shape=}")


# https://www.kaggle.com/code/elainedazzio/20250503-pg5
df_train["BMI"] = df_train["Weight"] / ((df_train["Height"] / 100) ** 2 + 1e-6)
df_test["BMI"] = df_test["Weight"] / ((df_test["Height"] / 100) ** 2 + 1e-6)
df_train["Temp_Binary"] = np.where(df_train["Body_Temp"] <= 39.5, 0, 1)
df_test["Temp_Binary"] = np.where(df_test["Body_Temp"] <= 39.5, 0, 1)
df_train["HeartRate_Binary"] = np.where(df_train["Heart_Rate"] <= 99.5, 0, 1)
df_test["HeartRate_Binary"] = np.where(df_test["Heart_Rate"] <= 99.5, 0, 1)
print(f"After adding manual new features: {df_train.shape=}, {df_test.shape=}")

# only adding features
# score=0.05985, best_iterations=[475, 402, 397, 481, 381], duration_training=57.53725028038025 -> no improvement

# also changing hyperparams
# score=0.05967, best_iterations=[2013, 1601, 1862, 1910, 1433], duration_training=206.3833954334259


print(f"{df_train.shape=}, {df_test.shape=}")
duration_loading_data = str(int(time.time() - time_start_overall))
print(f"{duration_loading_data=}")
time_start_training = time.time()

params_xgb = {
    # "eval_metric": 'rmsle',
    "eval_metric": "rmse",
    # https://www.kaggle.com/code/elainedazzio/20250503-pg5
    "objective": "reg:squarederror",  # default anyway
    "tree_method": "auto",
    "learning_rate": 0.021950824560804754,
    "max_depth": 7,
    "subsample": 0.8496893385061571,
    "colsample_bytree": 0.5070454944831791,
    "min_child_weight": 1,
    "gamma": 6.733484974122673e-05,
    "lambda": 5.635181872622184,
    "alpha": 0.1178141157522673,
}

trainer = XGBTrainer(
    params={
        "random_state": 42,
        "verbosity": 0,
        "n_estimators": 5_000,
        "early_stopping_rounds": 100,
    }
    | params_xgb,
    scoring_fn=root_mean_squared_log_error,
    log_transform_targets=True,  # True -> expm1 is applied to preds after predicting before scoring
    early_stop=True,
    # use_gpu=True,  # tree method"!
    log_evaluation=100,
    clip_preds=(1.0, 314.0),
)
(score, best_iterations, df_oof_predictions, df_test_predictions_from_oof, _, _) = (
    trainer.train_and_score(
        df_train=df_train,
        # df_train=pd.concat([df_train, df_train_features_dummy], axis=1),
        # ser_targets_train=np.log1p(ser_targets_train),
        ser_targets_train=ser_targets_train,
        df_test=df_test,
    )
)
duration_training = time.time() - time_start_training
print(f"{score=:.5f}, {best_iterations=}, {duration_training=}")

df_test_predictions_from_full = trainer.train_on_full(
    df_train=df_train,
    # df_train=pd.concat([df_train, df_train_features_dummy], axis=1),
    # ser_targets_train=np.log1p(ser_targets_train),
    ser_targets_train=ser_targets_train,
    df_test=df_test,
)

duration_all = str(int(time.time() - time_start_overall))
filename_prefix = __file__.split("\\")[-1][:-3]  # remove .py
if filename_prefix.startswith("run_"):
    filename_prefix = filename_prefix[4:]
df_oof_predictions["pred"].to_pickle(
    PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_oof.pkl"
)
df_test_predictions_from_oof["pred"].to_pickle(
    PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_test.pkl"
)
df_test_predictions_from_full["pred"].to_pickle(
    PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_test_from_full.pkl"
)
open(
    PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_{score=:.5f}_{duration_all=}", f"a"
).close()

# df_submission = df_test_predictions['pred'].reset_index().rename(
#     columns={"pred": "Calories"}
# )
# df_submission.to_csv(PATH_INTERIM_RESULTS / f"{filename_prefix}_{score=:.5f}_{duration_all=}.csv",
#                      index=False)

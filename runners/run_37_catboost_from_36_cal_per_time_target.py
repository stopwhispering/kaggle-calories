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
from trainers.catboost_trainer import CatBoostTrainer
from trainers.lgbm_trainer import LGBMTrainer
from trainers.metrics.metrics_lgbm import rmsle_lgbm
from trainers.xgb_trainer import XGBTrainer

time_start_overall = time.time()
df_train = pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
df_test = pd.read_csv(PATH_TEST).set_index("id")

df_train = convert_sex(df_train)
df_test = convert_sex(df_test)

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
    ]
)
df_train = pd.concat([df_train, df_train_features], axis=1)
df_test = pd.concat([df_test, df_test_features], axis=1)
print(f"After adding AutoFE: {df_train.shape=}, {df_test.shape=}")

print(f"{df_train.shape=}, {df_test.shape=}")
duration_loading_data = str(int(time.time() - time_start_overall))
print(f"{duration_loading_data=}")
time_start_training = time.time()


ser_targets_calories_per_time = ser_targets_train / df_train["Duration"]
a = 1


params_catboost = {
    "loss_function": "RMSE",
}

trainer = CatBoostTrainer(
    params={
        "random_state": 42,
        "n_estimators": 6000,
        # "verbose": -200,
        "verbose": 100,
        "early_stopping_rounds": 50,
    }
    | params_catboost,
    scoring_fn=root_mean_squared_log_error,
    log_transform_targets=True,  # True -> expm1 is applied to preds after predicting before scoring
    early_stop=True,
    use_gpu=True,
    # log_evaluation=100,  # todo implement for catboost trainer
    # clip_preds=(1.0, 314.0),
)
(score, best_iterations, df_oof_predictions, df_test_predictions_from_oof, _, _) = (
    trainer.train_and_score(
        df_train=df_train,
        # df_train=pd.concat([df_train, df_train_features_dummy], axis=1),
        # ser_targets_train=np.log1p(ser_targets_train),
        ser_targets_train=ser_targets_calories_per_time,
        df_test=df_test,
        # print_score=True,
    )
)
duration_training = time.time() - time_start_training
print(f"OTHER TARGETS {score=:.5f}, {best_iterations=}, {duration_training=}")

df_oof_predictions["pred"] = df_oof_predictions["pred"] * df_train["Duration"]
df_test_predictions_from_oof["pred"] = (
    df_test_predictions_from_oof["pred"] * df_test["Duration"]
)
df_oof_predictions["pred"] = df_oof_predictions["pred"].clip(1.0, 314.0)
df_test_predictions_from_oof["pred"] = df_test_predictions_from_oof["pred"].clip(
    1.0, 314.0
)
score = root_mean_squared_log_error(
    y_true=ser_targets_train,
    y_pred=df_oof_predictions["pred"],
)
print(f"{score=:.5f}, {best_iterations=}, {duration_training=}")


df_test_predictions_from_full = trainer.train_on_full(
    df_train=df_train,
    # df_train=pd.concat([df_train, df_train_features_dummy], axis=1),
    # ser_targets_train=np.log1p(ser_targets_train),
    ser_targets_train=ser_targets_calories_per_time,
    df_test=df_test,
)
df_test_predictions_from_full["pred"] = (
    df_test_predictions_from_full["pred"] * df_test["Duration"]
)
df_test_predictions_from_full["pred"] = df_test_predictions_from_full["pred"].clip(
    1.0, 314.0
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

import time
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error
import numpy as np
from sklearn.model_selection import StratifiedKFold

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


# identify duplicates in train data
df_train_d = df_train.assign(Calories=ser_targets_train).copy()
df_duplicates = df_train_d[df_train_d.duplicated(keep=False)].copy()
df_no_duplicates = df_train_d[~df_train_d.duplicated(keep=False)].copy()

# those that have no duplicates each get a random number between 0 and 5
# df_no_duplicates["unique_no"] = np.arange(0, len(df_no_duplicates))
df_no_duplicates["unique_no"] = np.random.randint(0, 5, len(df_no_duplicates))

# those that have duplicates get the same number
# df_duplicates["unique_no"] = df_duplicates.groupby(list(df_duplicates.columns)).ngroup() + len(df_no_duplicates)
df_duplicates["unique_no"] = (
    df_duplicates.groupby(list(df_duplicates.columns)).ngroup()
    + df_no_duplicates["unique_no"].max()
    + 1
)

# now we can merge the two dataframes
df_train_for_strat = (
    pd.concat([df_no_duplicates, df_duplicates], axis=0)
    .sort_index()
    .drop("Calories", axis=1)
)

# instead of default kfold, we use stratified kfold
# splits = KFold(n_splits=5, shuffle=True, random_state=42).split(df_train)
splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
    df_train_for_strat, df_train_for_strat["unique_no"]
)


print(f"{df_train.shape=}, {df_test.shape=}")
duration_loading_data = str(int(time.time() - time_start_overall))
print(f"{duration_loading_data=}")
time_start_training = time.time()

params_xgb = {
    # "eval_metric": 'rmsle',
    "eval_metric": "rmse",
    # https://www.kaggle.com/code/andrewsokolovsky/catboost-xgboost-lightgbm-rmsle-0-05684
    "max_depth": 10,
    "colsample_bytree": 0.7,
    "subsample": 0.9,
    "learning_rate": 0.02,
    "gamma": 0.01,
    "max_delta_step": 2,
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
    use_gpu=True,
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
        splits=splits,
    )
)
duration_training = time.time() - time_start_training
print(f"{score=:.5f}, {best_iterations=}, {duration_training=}")

# df_test_predictions_from_full = trainer.train_on_full(
#     df_train=df_train,
#     # df_train=pd.concat([df_train, df_train_features_dummy], axis=1),
#     # ser_targets_train=np.log1p(ser_targets_train),
#     ser_targets_train=ser_targets_train,
#     df_test=df_test,
# )  # WOULD BE USELESS HERE

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
# df_test_predictions_from_full["pred"].to_pickle(PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_test_from_full.pkl")
open(
    PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_{score=:.5f}_{duration_all=}", f"a"
).close()

# df_submission = df_test_predictions['pred'].reset_index().rename(
#     columns={"pred": "Calories"}
# )
# df_submission.to_csv(PATH_INTERIM_RESULTS / f"{filename_prefix}_{score=:.5f}_{duration_all=}.csv",
#                      index=False)

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

feature_store = FeatureStore(
    path_features=PATH_FEATURES,
    path_features_by_fold=PATH_FEATURES_BY_FOLD,
)
df_train_features, df_test_features = feature_store.read_features(
    column_names=[
        # "Combine_Sex_Duration",
        "Multiply_Weight_Duration",
        "Plus_Age_Duration",
        "Multiply_Age_Duration",
        # "GroupByThenMean_Age_Height",
        "Divide_Sex_Age",
        # "GroupByThenRank_Sex_Weight",
        "Minus_Sex_Heart_Rate",
        "Multiply_Duration_Body_Temp",

        # fs9
        'Plus_Sex_Age',
        'Max_Age_Duration',
        # 'GroupByThenMean_Age_Duration',
        # 'Combine_Duration_Heart_Rate',

        # fs3
        # "te_MEAN_Sex_Age_Body_Temp_not_folded_no_fillna",
        # "te_MEAN_Age_Height_Duration_Body_Temp_not_folded_no_fillna",
        # "te_MEAN_Age_Weight_Duration_Heart_Rate_not_folded_no_fillna",
        "te_MEAN_Age_Height_Weight_Heart_Rate_Body_Temp_not_folded_no_fillna",
        "te_MEAN_Sex_Age_Height_Weight_Heart_Rate_Body_Temp_not_folded_no_fillna",
        "te_MEAN_Age_Height_Weight_Duration_Heart_Rate_not_folded_no_fillna",

        "poly_Height_Duration2",
        "bins_Duration_15_uniform",
        "bins_Height_15_quantile",
        "bins_Age_17_kmeans",
        "bins_Body_Temp_3_kmeans",

        'map_orig_targets_Height_Weight_Heart_Rate',
        'map_orig_targets_Duration',
        'map_orig_targets_Sex_Height_Weight_Heart_Rate',
        'map_orig_targets_Sex_Age_Height_Weight_Duration_Heart_Rate',

        # fs10
        'map_orig_targets_max_Sex_Height_Weight_Heart_Rate',
        'map_orig_targets_max_Sex_Age_Weight_Heart_Rate_Body_Temp',
        'map_orig_targets_max_Sex_Duration',
        'map_orig_targets_max_Sex_Age_Height_Weight_Duration_Body_Temp',
        'map_orig_targets_max_Age_Weight_Heart_Rate_Body_Temp',
        'map_orig_targets_max_Sex_Age_Height_Weight_Duration_Heart_Rate_Body_Temp',
        'map_orig_targets_max_Age_Height_Duration_Heart_Rate_Body_Temp',

        # fs11
        'map_orig_targets_nunique_Sex_Weight_Heart_Rate_Body_Temp',
        'map_orig_targets_min_Height_Weight_Heart_Rate',
        'map_orig_targets_nunique_Age_Height_Weight_Duration_Heart_Rate',
        'map_orig_targets_nunique_Age_Weight_Heart_Rate_Body_Temp',
        'map_orig_targets_nunique_Age_Height_Weight_Duration_Body_Temp',
        'map_orig_targets_nunique_Sex_Age_Height_Weight_Heart_Rate_Body_Temp',
        'map_orig_targets_nunique_Height_Weight_Heart_Rate_Body_Temp',

        # fs12
        'masked_Heart_Rate_x_Male',
        'masked_Duration_x_Female',
        'masked_Duration_x_Male',
        'masked_Age_x_Female',
        'masked_Height_x_Male'

    ]
)
df_train = pd.concat([df_train, df_train_features], axis=1)
df_test = pd.concat([df_test, df_test_features], axis=1)
print(f"After adding AutoFE: {df_train.shape=}, {df_test.shape=}")

print(f"{df_train.shape=}, {df_test.shape=}")
duration_loading_data = str(int(time.time() - time_start_overall))
print(f"{duration_loading_data=}")
time_start_training = time.time()

params_lgbm = {
    # "metric": 'custom',   # for rmsle
    "metric": "rmse",  # for rmsle
    # 'learning_rate': 0.008,
    # 'max_depth': 10,
    # 'colsample_bytree': 0.7,
    "subsample": 0.9,
    "learning_rate": 0.008,
    "max_depth": 15,
    "colsample_bytree": 0.25,
    "num_leaves": 480,
}


trainer = LGBMTrainer(
    params={"random_state": 42, "verbose": -1, "n_estimators": 75_000} | params_lgbm,
    scoring_fn=root_mean_squared_log_error,
    log_transform_targets=True,  # True -> expm1 is applied to preds after predicting before scoring
    early_stop=True,
    # use_gpu=True,
    log_evaluation=500,
    stopping_rounds=600,
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
trainer.params_lgbm['n_estimators'] = int(np.mean(best_iterations) * 1.2)
print(f'{trainer.params_lgbm['n_estimators']=}')

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
print("Finished")

import time
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error
import numpy as np
from calories.constants import (
    PATH_TRAIN,
    PATH_TEST,
    PATH_PREDS_FOR_ENSEMBLES,
    PATH_FEATURES,
    PATH_FEATURES_BY_FOLD, PATH_ORIGINAL,
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

_df_original = (
    pd.read_csv(PATH_ORIGINAL)
    .rename({"Gender": "Sex"}, axis=1)
    .drop(["User_ID"], axis=1)
)
ser_targets_original = _df_original["Calories"]
df_train_original = _df_original[df_train.columns.tolist()]

# feature_store = FeatureStore(
#     path_features=PATH_FEATURES,
#     path_features_by_fold=PATH_FEATURES_BY_FOLD,
# )
# df_train_features, df_test_features = feature_store.read_features(
#     column_names=[
#         "Combine_Sex_Duration",
#         "Multiply_Weight_Duration",
#         "Plus_Age_Duration",
#         "Multiply_Age_Duration",
#         "GroupByThenMean_Age_Height",
#         "Divide_Sex_Age",
#         "GroupByThenRank_Sex_Weight",
#         "Minus_Sex_Heart_Rate",
#         "Multiply_Duration_Body_Temp",
#
#         # fs9
#         'Plus_Sex_Age',
#         'Max_Age_Duration',
#         'GroupByThenMean_Age_Duration',
#         'Combine_Duration_Heart_Rate',
#
#         # fs3
#         "te_MEAN_Sex_Age_Body_Temp_not_folded_no_fillna",
#         "te_MEAN_Age_Height_Duration_Body_Temp_not_folded_no_fillna",
#         "te_MEAN_Age_Weight_Duration_Heart_Rate_not_folded_no_fillna",
#         "te_MEAN_Age_Height_Weight_Heart_Rate_Body_Temp_not_folded_no_fillna",
#         "te_MEAN_Sex_Age_Height_Weight_Heart_Rate_Body_Temp_not_folded_no_fillna",
#         "te_MEAN_Age_Height_Weight_Duration_Heart_Rate_not_folded_no_fillna",
#
#         "poly_Height_Duration2",
#         "bins_Duration_15_uniform",
#         "bins_Height_15_quantile",
#         "bins_Age_17_kmeans",
#         "bins_Body_Temp_3_kmeans",
#
#         'map_orig_targets_Height_Weight_Heart_Rate',
#         'map_orig_targets_Duration',
#         'map_orig_targets_Sex_Height_Weight_Heart_Rate',
#         'map_orig_targets_Sex_Age_Height_Weight_Duration_Heart_Rate'
#     ]
# )
# df_train = pd.concat([df_train, df_train_features], axis=1)
# df_test = pd.concat([df_test, df_test_features], axis=1)
# print(f"After adding AutoFE: {df_train.shape=}, {df_test.shape=}")

print(f"{df_train.shape=}, {df_test.shape=}")
duration_loading_data = str(int(time.time() - time_start_overall))
print(f"{duration_loading_data=}")
time_start_training = time.time()






def modify_fold_data(df_train, df_val, df_test, ser_targets_train, i_fold: int):
    # selected_columns = nested_features_column_names
    # df_te_train_fold, df_te_val_fold, df_te_test_fold = nested_features_by_outer_fold[
    #     i_fold
    # ]
    #
    # df_te_train_fold_selected = df_te_train_fold[selected_columns]
    # df_te_val_fold_selected = df_te_val_fold[selected_columns]
    # df_te_test_fold_selected = df_te_test_fold[selected_columns]
    #
    # # make sure the columns really are dummy, and make sure we don't forget a dummy...
    # assert (df_train[selected_columns] == 1).all().all()
    # assert set(df_train.columns[(df_train == 1).all()].tolist()) == set(
    #     selected_columns
    # )

    df_train["is_original"] = 0
    df_val["is_original"] = 0
    df_test["is_original"] = 0
    df_train_original["is_original"] = 1

    assert (df_train.columns == df_train_original.columns).all()  # noqa

    df_train = pd.concat([df_train, df_train_original], axis=0, ignore_index=True)
    ser_targets_train = pd.concat(
        [ser_targets_train, ser_targets_original], axis=0, ignore_index=True
    )


    # df_train[selected_columns] = df_te_train_fold_selected
    # df_val[selected_columns] = df_te_val_fold_selected
    # if df_test is not None:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
    #         df_test[selected_columns] = df_te_test_fold_selected.values

    # print(f"{df_train.shape=}")
    return df_train, df_val, df_test, ser_targets_train







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
    clip_preds=(1.0, 314.0),
    fn_modify_fold_data=modify_fold_data,
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

# df_test_predictions_from_full = trainer.train_on_full(
#     df_train=df_train,
#     # df_train=pd.concat([df_train, df_train_features_dummy], axis=1),
#     # ser_targets_train=np.log1p(ser_targets_train),
#     ser_targets_train=ser_targets_train,
#     df_test=df_test,
# ) # todo fix n_estimators

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
# df_test_predictions_from_full["pred"].to_pickle(
#     PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_test_from_full.pkl"
# )
open(
    PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_{score=:.5f}_{duration_all=}", f"a"
).close()

# df_submission = df_test_predictions['pred'].reset_index().rename(
#     columns={"pred": "Calories"}
# )
# df_submission.to_csv(PATH_INTERIM_RESULTS / f"{filename_prefix}_{score=:.5f}_{duration_all=}.csv",
#                      index=False)

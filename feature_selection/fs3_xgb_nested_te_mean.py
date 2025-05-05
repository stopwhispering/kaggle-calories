import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import random

from calories.constants import (
    PATH_TRAIN,
    PATH_ENSEMBLE_SUBMISSIONS,
    PATH_CACHE,
    PATH_FEATURES_BY_FOLD,
)
from calories.preprocessing.dtypes import convert_sex
from feature_creators.read_features_util import read_features
from feature_creators.read_nested_features_util import read_nested_features
from my_feature_selection.ffs import ForwardFeatureSelector
from my_preprocessing.fe import FeatureEngineer
from my_preprocessing.feature_store.feature_store import FeatureStore
from my_preprocessing.target_encoding import CustomTargetEncoder
import pickle
import time
import gc

from trainers.xgb_trainer import XGBTrainer

# continue pt1 with doubled n_estimators
time_start = time.time()
df_train = pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
# df_test = pd.read_csv(PATH_TEST).set_index("id")


df_train = convert_sex(df_train)
# df_test = convert_sex(df_test)

feature_store = FeatureStore(
    path_features_by_fold=PATH_FEATURES_BY_FOLD,
)
nested_features_by_outer_fold = feature_store.read_features_by_fold(
    agg=["mean"],
    shuffle=True,
)

# nested_features_by_outer_fold: list[
#     tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = read_nested_features(
#     # feature_lists=
#     agg=['mean'],
#     shuffle=True,
#     include_original=False,
# )
nested_features_column_names = nested_features_by_outer_fold[0][0].columns.tolist()
print(
    f"{len(nested_features_by_outer_fold)=}. Total duration so far: {time.time() - time_start:.2f} seconds"
)
print(f"{len(nested_features_column_names)=}")

# we need one dataframe with dummy values for the FFS class
df_train_features_dummy = pd.DataFrame(
    data=1,
    index=df_train.index,
    columns=nested_features_column_names,
)
print(f"{df_train_features_dummy.shape=}")


FEATURES_ALREADY_FOUND = [
    "te_MEAN_Sex_Duration_no_fillna",
    "te_MEAN_Height_Weight_Body_Temp_no_fillna",
]
print(f"{len(FEATURES_ALREADY_FOUND)=}")

INITIAL_FEATURES_TO_DISCARD = [
    # already found to discard
    "te_MEAN_Sex_Duration_Heart_Rate_Body_Temp_no_fillna",
    "te_MEAN_Age_Duration_Heart_Rate_no_fillna",
    "te_MEAN_Sex_Age_Duration_Heart_Rate_no_fillna",
    "te_MEAN_Height_Body_Temp_no_fillna",
    "te_MEAN_Weight_Body_Temp_no_fillna",
    "te_MEAN_Height_Weight_Body_Temp_no_fillna",
]
print(f"{len(INITIAL_FEATURES_TO_DISCARD)=}")


def add_target_encoding(df_train, df_val, df_test, ser_targets_train, i_fold: int):
    selected_columns = [
        c for c in nested_features_column_names if c in df_train.columns
    ]
    df_te_train_fold, df_te_val_fold, df_te_test_fold = nested_features_by_outer_fold[
        i_fold
    ]

    df_te_train_fold_selected = df_te_train_fold[selected_columns]
    df_te_val_fold_selected = df_te_val_fold[selected_columns]
    df_te_test_fold_selected = df_te_test_fold[selected_columns]

    df_train[selected_columns] = df_te_train_fold_selected
    df_val[selected_columns] = df_te_val_fold_selected
    if df_test is not None:
        df_test[selected_columns] = df_te_test_fold_selected
    return df_train, df_val, df_test


# FFS
def score_dataset(
    df_train: pd.DataFrame, ser_targets_train: pd.Series = ser_targets_train
):
    params_xgb = {
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
            "early_stopping_rounds": 20,
        }  # expected ~450 iterations
        | params_xgb,
        scoring_fn=root_mean_squared_log_error,
        log_transform_targets=True,  # True -> expm1 is applied to preds after predicting before scoring
        early_stop=True,
        use_gpu=True,
        silent=True,
        fn_add_target_encoding=add_target_encoding,
    )

    (score, best_iterations, df_oof_predictions, df_test_predictions, _, _) = (
        trainer.train_and_score(
            df_train=df_train,
            ser_targets_train=ser_targets_train,
            # splits=KFold(n_splits=12, shuffle=True, random_state=42).split(df_train),
        )
    )
    # print(f"{score=:.5f}, {best_iterations=}")

    return -score


ffs = ForwardFeatureSelector(
    df_train_all_features=pd.concat([df_train, df_train_features_dummy], axis=1),
    model_fn=score_dataset,
    original_features=df_train.columns.tolist(),
    initial_features_to_select=FEATURES_ALREADY_FOUND,
    # initial_features_to_select=
    max_features_to_select=25,
    stop_at_max_score=False,  # Make False next time
    initial_features_to_discard=INITIAL_FEATURES_TO_DISCARD,
    # save_intermediate_results=True,
    discard_worst_n_features_each_round=3,
)
df_results: pd.DataFrame = ffs.start()
df_results.to_excel(PATH_CACHE / "df_results_fs3.xlsx", index=False)

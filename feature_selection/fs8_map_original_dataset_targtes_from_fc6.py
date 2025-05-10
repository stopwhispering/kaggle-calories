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
    PATH_FEATURES,
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

from my_ml_util.trainers.catboost_trainer import (CatBoostTrainer)
from my_ml_util.trainers.xgb_trainer import XGBTrainer

# continue pt1 with doubled n_estimators
time_start = time.time()
df_train = pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
# df_test = pd.read_csv(PATH_TEST).set_index("id")


df_train = convert_sex(df_train)
# df_test = convert_sex(df_test)

feature_store = FeatureStore(
    path_features=PATH_FEATURES,
    # path_features_by_fold=PATH_FEATURES_BY_FOLD,
)
df_train_features, _df_test_features = feature_store.read_features(
    criteria={"type": ("map original targets",)}  # fc6
)
# _cat_cols = df_train_features.select_dtypes("object").columns.tolist()
# df_train_features[_cat_cols] = df_train_features[_cat_cols].astype("category")
print(f"{df_train_features.shape=}")

FEATURES_ALREADY_FOUND = [
]
print(f"{len(FEATURES_ALREADY_FOUND)=}")

INITIAL_FEATURES_TO_DISCARD = [
    # already found to discard
]


# FFS
def score_dataset(
    df_train: pd.DataFrame, ser_targets_train: pd.Series = ser_targets_train
):
    params_catboost = {
        "loss_function": "RMSE",
        "learning_rate": 0.10,  # 0.20 -> ca. 12 secs
    }
    trainer = CatBoostTrainer(
        params={
            "random_state": 42,
            "n_estimators": 6000,
            # "verbose": -200,
            "verbose": 100,
            "early_stopping_rounds": 6,
        }
        | params_catboost,
        scoring_fn=root_mean_squared_log_error,
        log_transform_targets=True,  # True -> expm1 is applied to preds after predicting before scoring
        early_stop=True,
        use_gpu=True,
        silent=True,
        clip_preds=(1.0, 314.0),
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
    df_train_all_features=pd.concat([df_train, df_train_features], axis=1),
    model_fn=score_dataset,
    original_features=df_train.columns.tolist(),
    initial_features_to_select=FEATURES_ALREADY_FOUND,
    # initial_features_to_select=
    max_features_to_select=25,
    stop_at_max_score=False,  # Make False next time
    initial_features_to_discard=INITIAL_FEATURES_TO_DISCARD,
    # save_intermediate_results=True,
    discard_worst_n_features_each_round=0,
)
df_results: pd.DataFrame = ffs.start()
df_results.to_excel(PATH_CACHE / "df_results_fs8.xlsx", index=False)

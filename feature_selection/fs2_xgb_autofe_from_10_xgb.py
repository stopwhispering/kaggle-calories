import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import random

from calories.constants import PATH_TRAIN, PATH_ENSEMBLE_SUBMISSIONS, PATH_CACHE
from calories.preprocessing.dtypes import convert_sex
from feature_creators.read_features_util import read_features
from feature_creators.read_nested_features_util import read_nested_features
from my_feature_selection.ffs import ForwardFeatureSelector
from my_preprocessing.fe import FeatureEngineer
from my_preprocessing.target_encoding import CustomTargetEncoder
import pickle
import time
import gc

from trainers.xgb_trainer import XGBTrainer

# continue pt1 with doubled n_estimators
time_start = time.time()
df_train = (
    pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
# df_test = pd.read_csv(PATH_TEST).set_index("id")


df_train = convert_sex(df_train)
# df_test = convert_sex(df_test)


df_train_features, _df_test_features = read_features(include_original=False,
                                                     # type_="binning",
                                                     shuffle=True)
_cat_cols = df_train_features.select_dtypes("object").columns.tolist()
df_train_features[_cat_cols] = df_train_features[_cat_cols].astype("category")
print(f"{df_train_features.shape=}")

FEATURES_ALREADY_FOUND = [
    'Combine_Sex_Duration', 'Multiply_Weight_Duration', 'Plus_Age_Duration'  # fs1
]
print(f'{len(FEATURES_ALREADY_FOUND)=}')

INITIAL_FEATURES_TO_DISCARD = [
    # already found to discard
]


# FFS
def score_dataset(df_train: pd.DataFrame,
                  ser_targets_train: pd.Series = ser_targets_train):
    params_xgb = {
        "eval_metric": 'rmse',

        # https://www.kaggle.com/code/andrewsokolovsky/catboost-xgboost-lightgbm-rmsle-0-05684
        'max_depth': 10,
        'colsample_bytree': 0.7,
        'subsample': 0.9,
        'learning_rate': 0.02,
        'gamma': 0.01,
        'max_delta_step': 2,
    }
    trainer = XGBTrainer(
        params={"random_state": 42, "verbosity": 0, "n_estimators": 5_000, "early_stopping_rounds": 20}  # expected ~450 iterations
               | params_xgb,  # mae, not rmse
        scoring_fn=root_mean_squared_log_error,
        log_transform_targets=True,  # True -> expm1 is applied to preds after predicting before scoring
        early_stop=True,
        use_gpu=True,
        silent=True,
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
df_results.to_excel(PATH_CACHE / "df_results_fs2.xlsx", index=False)

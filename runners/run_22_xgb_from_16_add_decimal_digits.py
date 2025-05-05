import time
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error
import numpy as np
from calories.constants import PATH_TRAIN, PATH_TEST, PATH_PREDS_FOR_ENSEMBLES
from calories.preprocessing.dtypes import convert_sex
from feature_creators.read_features_util import read_features
from trainers.lgbm_trainer import LGBMTrainer
from trainers.metrics.metrics_lgbm import rmsle_lgbm
from trainers.xgb_trainer import XGBTrainer

time_start_overall = time.time()
df_train = pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
df_test = pd.read_csv(PATH_TEST).set_index("id")

df_train = convert_sex(df_train)
df_test = convert_sex(df_test)

df_train_features, df_test_features = read_features(
    column_names=[
        "Combine_Sex_Duration",
        "Multiply_Weight_Duration",
        "Plus_Age_Duration",
        "Multiply_Age_Duration",
        "GroupByThenMean_Age_Height",
        "Divide_Sex_Age",
    ],
    include_original=False,
    shuffle=True,
)
_cat_cols = df_train_features.select_dtypes("object").columns.tolist()
df_train_features[_cat_cols] = df_train_features[_cat_cols].astype("category")
df_test_features[_cat_cols] = df_test_features[_cat_cols].astype("category")
df_train = pd.concat([df_train, df_train_features], axis=1)
df_test = pd.concat([df_test, df_test_features], axis=1)


decimal_cols = ["Body_Temp", "Height", "Heart_Rate"]

df_train_str = (
    pd.read_csv(PATH_TRAIN, dtype={c: str for c in decimal_cols})
    .set_index("id")
    .drop("Calories", axis=1)
)
df_test_str = pd.read_csv(PATH_TEST, dtype={c: str for c in decimal_cols}).set_index(
    "id"
)


def count_decimal_digits(s):
    if pd.isna(s):
        return None
    if "." in s:
        return len(s.split(".")[1])
    return 0


for col in decimal_cols:
    colname = "decimal_digits_" + col
    df_train[colname] = df_train_str[col].apply(count_decimal_digits)
    df_test[colname] = df_test_str[col].apply(count_decimal_digits)


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
(score, best_iterations, df_oof_predictions, df_test_predictions, _, _) = (
    trainer.train_and_score(
        df_train=df_train,
        # df_train=pd.concat([df_train, df_train_features_dummy], axis=1),
        # ser_targets_train=np.log1p(ser_targets_train),
        ser_targets_train=ser_targets_train,
        df_test=df_test,
    )
)
print(f"{score=:.5f}")

duration_training = time.time() - time_start_training
print(f"{score=:.5f}, {best_iterations=}, {duration_training=}")

duration_all = str(int(time.time() - time_start_overall))
filename_prefix = __file__.split("\\")[-1][:-3]  # remove .py
if filename_prefix.startswith("run_"):
    filename_prefix = filename_prefix[4:]
df_oof_predictions["pred"].to_pickle(
    PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_oof.pkl"
)
df_test_predictions["pred"].to_pickle(
    PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_test.pkl"
)
open(
    PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_{score=:.5f}_{duration_all=}", f"a"
).close()

# df_submission = df_test_predictions['pred'].reset_index().rename(
#     columns={"pred": "Calories"}
# )
# df_submission.to_csv(PATH_INTERIM_RESULTS / f"{filename_prefix}_{score=:.5f}_{duration_all=}.csv",
#                      index=False)

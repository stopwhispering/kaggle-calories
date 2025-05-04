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
df_train = (
    pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
df_test = pd.read_csv(PATH_TEST).set_index("id")

df_train = convert_sex(df_train)
df_test = convert_sex(df_test)

# https://www.kaggle.com/code/yunusemreakca/s5e5-calories-prediction/notebook
df_train["BMI"] = df_train["Weight"] / (df_train["Height"] / 100) ** 2
df_test["BMI"] = df_test["Weight"] / (df_test["Height"] / 100) ** 2
df_train["Effort_Score"] = df_train["Heart_Rate"] * df_train["Duration"]
df_test["Effort_Score"] = df_test["Heart_Rate"] * df_test["Duration"]
df_train["HR_per_min"] = df_train["Heart_Rate"] / df_train["Duration"]
df_test["HR_per_min"] = df_test["Heart_Rate"] / df_test["Duration"]
df_train["HeartRate_per_BodyTemp"] = df_train["Heart_Rate"] / df_train["Body_Temp"]
df_test["HeartRate_per_BodyTemp"] = df_test["Heart_Rate"] / df_test["Body_Temp"]
df_train["HR_age_ratio"] = df_train["Heart_Rate"] / df_train["Age"]
df_test["HR_age_ratio"] = df_test["Heart_Rate"] / df_test["Age"]
df_train["Weight_per_Height"] = df_train["Weight"] / df_train["Height"]
df_test["Weight_per_Height"] = df_test["Weight"] / df_test["Height"]
df_train["Age_BMI"] = df_train["Age"] * df_train["BMI"]
df_test["Age_BMI"] = df_test["Age"] * df_test["BMI"]
df_train["BodyTemp_Duration"] = df_train["Body_Temp"] * df_train["Duration"]
df_test["BodyTemp_Duration"] = df_test["Body_Temp"] * df_test["Duration"]
mean_body_temp_train = df_train["Body_Temp"].mean()
df_train["BodyTemp_Deviation"] = df_train["Body_Temp"] - mean_body_temp_train
mean_body_temp_test = df_test["Body_Temp"].mean()
df_test["BodyTemp_Deviation"] = df_test["Body_Temp"] - mean_body_temp_test
df_train["Max_Heart_Rate"] = 220 - df_train["Age"]
df_test["Max_Heart_Rate"] = 220 - df_test["Age"]
df_train["HeartRate_Max_HR_Ratio"] = df_train["Heart_Rate"] / df_train["Max_Heart_Rate"]
df_test["HeartRate_Max_HR_Ratio"] = df_test["Heart_Rate"] / df_test["Max_Heart_Rate"]
df_train["Effort_Score_per_Duration"] = df_train["Effort_Score"] / df_train["Duration"]
df_test["Effort_Score_per_Duration"] = df_test["Effort_Score"] / df_test["Duration"]
df_train["Weight_Age"] = df_train["Weight"] * df_train["Age"]
df_test["Weight_Age"] = df_test["Weight"] * df_test["Age"]
df_train["Weight_per_BMI"] = df_train["Weight"] / df_train["BMI"]
df_test["Weight_per_BMI"] = df_test["Weight"] / df_test["BMI"]
df_train["Exercise_Duration_Category"] = pd.cut(df_train["Duration"], bins=[0, 30, 60, 180], labels=["Short", "Medium", "Long"])
df_test["Exercise_Duration_Category"] = pd.cut(df_test["Duration"], bins=[0, 30, 60, 180], labels=["Short", "Medium", "Long"])
df_train["Age_Group"] = pd.cut(df_train["Age"], bins=[0, 30, 50, 100], labels=["Young", "Middle-Aged", "Old"])
df_test["Age_Group"] = pd.cut(df_test["Age"], bins=[0, 30, 50, 100], labels=["Young", "Middle-Aged", "Old"])



df_train_features, df_test_features = read_features(column_names=['Combine_Sex_Duration', 'Multiply_Weight_Duration', 'Plus_Age_Duration',
                                                                  'Multiply_Age_Duration', 'GroupByThenMean_Age_Height', 'Divide_Sex_Age'],
                                                    include_original=False,
                                                    shuffle=True)
_cat_cols = df_train_features.select_dtypes('object').columns.tolist()
df_train_features[_cat_cols] = df_train_features[_cat_cols].astype('category')
df_test_features[_cat_cols] = df_test_features[_cat_cols].astype('category')
df_train = pd.concat([df_train, df_train_features], axis=1)
df_test = pd.concat([df_test, df_test_features], axis=1)

print(f'{df_train.shape=}, {df_test.shape=}')
duration_loading_data = str(int(time.time() - time_start_overall))
print(f"{duration_loading_data=}")
time_start_training = time.time()

params_xgb = {
    # "eval_metric": 'rmsle',
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
    params={"random_state": 42, "verbosity": 0, "n_estimators": 5_000, "early_stopping_rounds": 100} | params_xgb,
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
print(f'{score=:.5f}')

duration_training = time.time() - time_start_training
print(f"{score=:.5f}, {best_iterations=}, {duration_training=}")

duration_all = str(int(time.time() - time_start_overall))
filename_prefix = __file__.split('\\')[-1][:-3]  # remove .py
if filename_prefix.startswith("run_"):
    filename_prefix = filename_prefix[4:]
df_oof_predictions["pred"].to_pickle(PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_oof.pkl")
df_test_predictions["pred"].to_pickle(PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_test.pkl")
open(PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_{score=:.5f}_{duration_all=}", f"a").close()

# df_submission = df_test_predictions['pred'].reset_index().rename(
#     columns={"pred": "Calories"}
# )
# df_submission.to_csv(PATH_INTERIM_RESULTS / f"{filename_prefix}_{score=:.5f}_{duration_all=}.csv",
#                      index=False)

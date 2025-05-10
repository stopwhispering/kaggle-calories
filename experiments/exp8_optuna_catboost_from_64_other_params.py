import time
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_log_error
import numpy as np
from calories.constants import (
    PATH_TRAIN,
    PATH_TEST,
    PATH_PREDS_FOR_ENSEMBLES,
    PATH_CACHE, PATH_FEATURES,
)
from calories.preprocessing.dtypes import convert_sex
from my_ml_util.trainers.catboost_trainer import CatBoostTrainer
from my_ml_util.trainers.lgbm_trainer import LGBMTrainer
from my_ml_util.trainers.metrics.metrics_lgbm import rmsle_lgbm
from my_ml_util.trainers.xgb_trainer import XGBTrainer
import optuna

from my_preprocessing.feature_store.feature_store import FeatureStore

time_start_overall = time.time()
df_train = pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
df_test = pd.read_csv(PATH_TEST).set_index("id")

df_train = convert_sex(df_train)
df_test = convert_sex(df_test)
print(f"Competition Features: {df_train.shape=}, {df_test.shape=}")

feature_store = FeatureStore(
    path_features=PATH_FEATURES,
    # path_features_by_fold=PATH_FEATURES_BY_FOLD,
)
df_train_features, df_test_features = feature_store.read_features(
    column_names=[
        # "Combine_Sex_Duration",  # disabled in later runs of exp8
        "Multiply_Weight_Duration",
        "Plus_Age_Duration",
        "Multiply_Age_Duration",
        "GroupByThenMean_Age_Height",
        "Divide_Sex_Age",
        "GroupByThenRank_Sex_Weight",
        "Minus_Sex_Heart_Rate",
        "Multiply_Duration_Body_Temp",

        # fs9
        'Plus_Sex_Age',
        'Max_Age_Duration',
        'GroupByThenMean_Age_Duration',
        # 'Combine_Duration_Heart_Rate',  # disabled in later runs of exp8

        # fs3
        "te_MEAN_Sex_Age_Body_Temp_not_folded_no_fillna",
        "te_MEAN_Age_Height_Duration_Body_Temp_not_folded_no_fillna",
        "te_MEAN_Age_Weight_Duration_Heart_Rate_not_folded_no_fillna",
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

    ]
)
df_train = pd.concat([df_train, df_train_features], axis=1)
df_test = pd.concat([df_test, df_test_features], axis=1)
print(f"After adding saved features: {df_train.shape=}, {df_test.shape=}")

print(f"{df_train.shape=}, {df_test.shape=}")
duration_loading_data = str(int(time.time() - time_start_overall))
print(f"{duration_loading_data=}")
time_start_training = time.time()














results = []

estimator = CatBoostRegressor()


def objective(trial):
    start = time.time()



    params_gbdt = {

        # catboost
        # core parameters
        # 'n_estimators': trial.suggest_int('n_estimators', 8, 50),  # we're using early stopping
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),  # alias: "eta"; default: 0.03
        'max_depth': trial.suggest_int('max_depth', 4, 16),  # default: 6 (16 if grow_policy is "Lossguide"); alias: "depth"
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0, log=True),  # alias: "l2_leaf_reg"; default: 3.0

        # medium to high importance parameters
        # 'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 1.0),  # see below, special case; only relevant if "bootstrap_type" is "Bayesian"
        'max_bin': trial.suggest_int('max_bin', 2, 255),  # alias: "border_count"; default: 254
        # 'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.5, 1.0),  # alias: 'rsm'; default: 1.0; DISABLED WITH ERROR on GPU for most objectives
        'random_strength': trial.suggest_int('random_strength', 0.001, 10.0),  # default: 1.0

        # low to medium importance parameters or specific scenarios
        'feature_border_type': trial.suggest_categorical('feature_border_type', [  # default: "GreedyLogSum"
            "Median", "Uniform", "UniformAndQuantiles", "MaxLogSum", "MinEntropy", "GreedyLogSum"
        ]),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 20),  # default: 1; might improve accuracy for LogLoss/RMSE at the cost of speed
        "leaf_estimation_method": trial.suggest_categorical(
            "leaf_estimation_method", ["Newton", "Gradient"],  # ["Newton", "Gradient", "Exact"]; "Exact" not available for RMSE
        ),  # default: depends on objective
        'grow_policy': trial.suggest_categorical('grow_policy', ["SymmetricTree", "Lossguide", "Depthwise"]),  # default: "SymmetricTree"
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ["Bernoulli", "MVS", "No", "Poisson", "Bayesian"]),  # default: usually "Bayesian"; if "No", don't set "subsample" and "bagging_temperature"


        # very little importance for Catboost
        # 'min_child_samples': trial.suggest_int('min_child_samples', 1, 10),  # alias: "min_data_in_leaf"; default: 1; only relevant if grow_policy is "Lossguide" or "Depthwise"


        # # xgb
        # "max_depth": trial.suggest_int("max_depth", 3, 45),
        # "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
        # "min_child_weight": trial.suggest_int("min_child_weight", 10, 90),
        # "reg_alpha": trial.suggest_int("reg_alpha", 2, 15),
        # # "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 0.1),
        # "subsample": trial.suggest_float("subsample", 0, 1.0),
        # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.15, 1.0),
        # "colsample_bynode": trial.suggest_float(
        #     "colsample_bytree", 0.15, 1.0
        # ),  # todo fix this
        # # "n_estimators": trial.suggest_int("n_estimators", 5, 1500),
        # # "num_leaves": trial.suggest_int("num_leaves", 2, 2048),
        # # "max_bin": trial.suggest_int("max_bin", 32, 2048),
    }

    # special cases catboost
    # bagging_temperature: float or 0 or 1 (special cases), default 1
    if params_gbdt['bootstrap_type'] == "Bayesian":
        bagging_temperature_type = trial.suggest_categorical('bagging_temperature_type', [0, 1, "float"])
        if bagging_temperature_type == "float":
            params_gbdt['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.1, 1.0)
        else:
            params_gbdt['bagging_temperature'] = bagging_temperature_type

    trainer = CatBoostTrainer(
        params={
                   "random_state": 42,
                   "n_estimators": 6000,
                   # "verbose": -200,
                   "verbose": 100,
                   "early_stopping_rounds": 10,
               } | params_gbdt,
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
            # df_test=df_test,
        )
    )

    duration = time.time() - start
    print(f"{score=:.5f}, {best_iterations=}, {params_gbdt=}, {duration=:.2f}")

    results.append(
        {
            "score": score,
            "best_iterations": best_iterations,
            "params": params_gbdt,
            "duration": duration,
        }
    )

    return score


study = optuna.create_study(direction="minimize", study_name='exp8-study', storage='sqlite:///exp8.db', load_if_exists=True)
study.optimize(objective, n_trials=100, timeout=36_000, catch=(Exception,))
print(f"Best params is {study.best_params} with value {study.best_value}")

df_results = pd.DataFrame(results)
df_results.to_csv(
    PATH_CACHE / "results_exp_7_catboost_optuna.csv",
    index=False,
)

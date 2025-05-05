import time
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error
import numpy as np
from calories.constants import (
    PATH_TRAIN,
    PATH_TEST,
    PATH_PREDS_FOR_ENSEMBLES,
    PATH_CACHE,
)
from calories.preprocessing.dtypes import convert_sex
from trainers.lgbm_trainer import LGBMTrainer
from trainers.metrics.metrics_lgbm import rmsle_lgbm
from trainers.xgb_trainer import XGBTrainer
import optuna

time_start_overall = time.time()
df_train = pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
df_test = pd.read_csv(PATH_TEST).set_index("id")

df_train = convert_sex(df_train)
df_test = convert_sex(df_test)

duration_loading_data = str(int(time.time() - time_start_overall))
print(f"{duration_loading_data=}")
time_start_training = time.time()


results = []


def objective(trial):
    start = time.time()

    params_xgb = {
        "max_depth": trial.suggest_int("max_depth", 3, 45),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 90),
        "reg_alpha": trial.suggest_int("reg_alpha", 2, 15),
        # "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 0.1),
        "subsample": trial.suggest_float("subsample", 0, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.15, 1.0),
        "colsample_bynode": trial.suggest_float(
            "colsample_bytree", 0.15, 1.0
        ),  # todo fix this
        # "n_estimators": trial.suggest_int("n_estimators", 5, 1500),
        # "num_leaves": trial.suggest_int("num_leaves", 2, 2048),
        # "max_bin": trial.suggest_int("max_bin", 32, 2048),
    }

    trainer = XGBTrainer(
        params={
            "random_state": 42,
            "verbosity": 0,
            "n_estimators": 20_000,
            # 'objective': 'reg:squaredlogerror',
            "eval_metric": "rmsle",
            # 'reg_lambda': 1,
            "early_stopping_rounds": 100,
        }  #
        | params_xgb,
        scoring_fn=root_mean_squared_log_error,
        early_stop=True,
        use_gpu=True,
        silent=True,
    )

    (score, best_iterations, df_oof_predictions, df_test_predictions, _, _) = (
        trainer.train_and_score(
            df_train=df_train,
            ser_targets_train=ser_targets_train,
            # df_test=df_test,
        )
    )

    duration = time.time() - start
    print(f"{score=:.5f}, {best_iterations=}, {params_xgb=}, {duration=:.2f}")

    results.append(
        {
            "score": score,
            "best_iterations": best_iterations,
            "params": params_xgb,
            "duration": duration,
        }
    )

    return score


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=800, timeout=36_000)
print(f"Best params is {study.best_params} with value {study.best_value}")

df_results = pd.DataFrame(results)
df_results.to_csv(
    PATH_CACHE / "results_exp_2_xgb_optuna.csv",
    index=False,
)

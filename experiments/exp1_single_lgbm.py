import time
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error
import numpy as np
from calories.constants import PATH_TRAIN, PATH_TEST
from calories.preprocessing.dtypes import convert_sex
from trainers.lgbm_trainer import LGBMTrainer
from trainers.metrics.metrics_lgbm import rmsle_lgbm

time_start_overall = time.time()
df_train = pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
df_test = pd.read_csv(PATH_TEST).set_index("id")

df_train = convert_sex(df_train)
df_test = convert_sex(df_test)

duration_loading_data = str(int(time.time() - time_start_overall))
print(f"{duration_loading_data=}")
time_start_training = time.time()

params_lgbm = {
    "metric": "custom",  # for rmsle
}

trainer = LGBMTrainer(
    params={"random_state": 42, "verbose": -1, "n_estimators": 5_000} | params_lgbm,
    scoring_fn=root_mean_squared_log_error,
    early_stop=True,
    # fn_add_target_encoding=add_target_encoding,
    log_evaluation=200,
    stopping_rounds=100,
    eval_metric=rmsle_lgbm,
)
(score, best_iterations, df_oof_predictions, df_test_predictions, _, _) = (
    trainer.train_and_score(
        df_train=df_train,
        # df_train=pd.concat([df_train, df_train_features_dummy], axis=1),
        ser_targets_train=ser_targets_train,
        df_test=df_test,
    )
)
duration_training = time.time() - time_start_training
print(f"{score=:.5f}, {best_iterations=}, {duration_training=}")

# duration_all = str(int(time.time() - time_start_overall))
# filename_prefix = __file__.split('\\')[-1][:-3]  # remove .py
# if filename_prefix.startswith("run_single_"):
#     filename_prefix = filename_prefix[11:]
# df_oof_predictions["pred"].to_pickle(PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_oof.pkl")
# df_test_predictions["pred"].to_pickle(PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_test.pkl")
# open(PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_{score=:.5f}_{duration_all=}", f"a").close()
#
# df_submission = df_test_predictions['pred'].reset_index().rename(
#     columns={"pred": "Calories"}
# )
# df_submission.to_csv(PATH_INTERIM_RESULTS / f"{filename_prefix}_{score=:.5f}_{duration_all=}.csv",
#                      index=False)

# C:\Users\Johannes\PycharmProjects\my-ml-util\trainers\metrics\metrics_lgbm.py:6: RuntimeWarning: invalid value encountered in log1p
#   score = np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))
# score=0.06250, best_iterations=[741, 1307, 1485, 1052, 328], duration_training=43.52465081214905
#
# Process finished with exit code 0

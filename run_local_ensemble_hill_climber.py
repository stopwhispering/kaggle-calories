import pandas as pd
from sklearn.metrics import root_mean_squared_log_error

from calories.constants import (
    PATH_TRAIN,
    PATH_PREDS_FOR_ENSEMBLES,
    PATH_ENSEMBLE_SUBMISSIONS,
)
from my_ml_util.ensembling.hill_climbing import (
    HillClimber,
    score_multiple_rmse,
    score_multiple_rmsle,
)
import numpy as np
import time


duration_start = time.time()
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]


# get list of files that match pattern "_oof.pkl" in PATH_PREDS_FOR_ENSEMBLES
oof_files = [file for file in PATH_PREDS_FOR_ENSEMBLES.glob("*_oof.pkl")]

test_files = [file for file in PATH_PREDS_FOR_ENSEMBLES.glob("*_test.pkl")]
test_files_from_full = [
    file for file in PATH_PREDS_FOR_ENSEMBLES.glob("*_test_from_full.pkl")
]

# for each oof file, get the corresponding test file; load both files
all_oof = []
all_test = []
for oof_file in oof_files:
    test_file = PATH_PREDS_FOR_ENSEMBLES / (
        oof_file.stem.replace("_oof", "_test_from_full") + ".pkl"
    )
    if not test_file.exists():
        test_file = PATH_PREDS_FOR_ENSEMBLES / (
            oof_file.stem.replace("_oof", "_test") + ".pkl"
        )
        assert test_file.exists()

    print(f"Using test file {test_file.name}.")
    # assert test_file in test_files, f"Test file not found for {oof_file.name}"

    # load the files
    ser_oof = pd.read_pickle(oof_file)
    ser_test = pd.read_pickle(test_file)

    # if len(ser_oof) == 794_868:
    if len(ser_oof) == 765000:
        ser_oof = ser_oof.iloc[:750_000]

    name = oof_file.stem.replace("_oof", "")
    ser_oof.name = name
    ser_test.name = name

    all_oof.append(ser_oof)
    all_test.append(ser_test)

df_oof = pd.concat(all_oof, axis=1)
df_test = pd.concat(all_test, axis=1)

df_oof = df_oof.clip(lower=0)
df_test = df_test.clip(lower=0)

weights, df_weights = HillClimber(
    score_multiple_function=score_multiple_rmsle,
    use_negative_weights=True,
    tol=0.00001,  # 0.0001# 0.0001,
    scoring_func=root_mean_squared_log_error,
).run(df_oof, ser_targets_train)
print(weights)
print(df_weights)

ser_ensemble = np.matmul(df_oof, weights).rename("pred")
final_score = root_mean_squared_log_error(ser_targets_train, ser_ensemble)
print(f"{final_score=:.5f}")

ser_test_ensemble = np.matmul(df_test, weights).rename("pred")
print(f"{ser_test_ensemble.shape=}")

models_not_used = [m for m in df_oof.columns if m not in df_weights["model"].values]
print(f"{models_not_used=}")

# get date + time as string
date_time_str = pd.to_datetime("now").strftime("%Y-%m-%d_%H-%M-%S")
filename = f"hc_ensemble_{date_time_str}_{final_score:.5f}.csv"

ser_test_ensemble.name = "Calories"
df_submission = ser_test_ensemble.reset_index().rename(columns={"pred": "Calories"})
df_submission["Calories"] = np.clip(df_submission["Calories"], 1, 314)
df_submission.to_csv(PATH_ENSEMBLE_SUBMISSIONS / filename, index=False)

df_weights.to_csv(
    PATH_ENSEMBLE_SUBMISSIONS / f"hc_weights_{date_time_str}.csv", index=False
)

ser_ensemble.name = "Calories"
df_ensemble_oof = ser_ensemble.reset_index().rename(columns={"pred": "Calories", "index": "id"})
df_ensemble_oof["Calories"] = np.clip(df_ensemble_oof["Calories"], 0, None)
df_ensemble_oof.to_csv(
    PATH_ENSEMBLE_SUBMISSIONS / f"hc_train_ensemble_{date_time_str}.csv", index=False
)

# ser_ensemble.to_csv(PATH_ENSEMBLE_SUBMISSIONS / f"hc_train_ensemble_{date_time_str}.csv", index=False)

duration_total = time.time() - duration_start
print(f"Duration: {duration_total:.2f} seconds")

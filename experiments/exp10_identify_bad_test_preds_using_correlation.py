import pandas as pd
from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error

import cupy as cp
from calories.constants import (
    PATH_TRAIN,
    PATH_PREDS_FOR_ENSEMBLES,
    PATH_ENSEMBLE_SUBMISSIONS, PATH_CACHE,
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

# code from Hill Climbing
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

    # test_file = PATH_PREDS_FOR_ENSEMBLES / (
    #     oof_file.stem.replace("_oof", "_test") + ".pkl"
    # )

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

    if all_test:
        if not ser_test.index.equals(all_test[0].index):
            print(f"Warning: {ser_test.index} != {all_test[0].index} ({ser_test.name=}). Replacing index.")
            ser_test.index = all_test[0].index
        # assert ser_test.index.equals(all_test[0].index)

    all_oof.append(ser_oof)
    all_test.append(ser_test)

df_oof = pd.concat(all_oof, axis=1)
df_test = pd.concat(all_test, axis=1)

df_oof = df_oof.clip(lower=0)
df_test = df_test.clip(lower=0)
print(f'{df_oof.shape=}, {df_test.shape=}')


def compute_correlation(df: pd.DataFrame):
    # for each column in df, compute correlation to first column
    results = []
    for col in df.columns:
        if col == df.columns[0]:
            continue
        corr_pearson = df.iloc[:, 0].corr(df[col])
        corr_spearman = df.iloc[:, 0].corr(df[col], method="spearman")
        # print(f"{corr:.4f} Correlation {df.columns[0]} vs {col}")
        results.append(
            {
                "col": col,
                "pearson (default)": corr_pearson,
                "spearman": corr_spearman,
            }
        )
    return results

# sort by correlation
# results = sorted(compute_correlation(df_test), key=lambda x: x["spearman"], reverse=True)
results = sorted(compute_correlation(df_test), key=lambda x: x["pearson (default)"], reverse=True)
print(pd.DataFrame(results))
# pd.DataFrame(results).to_excel(PATH_CACHE / "correlation_test.xlsx", index=False)

results = sorted(compute_correlation(df_test), key=lambda x: x["spearman"], reverse=True)
print(pd.DataFrame(results))

# # sort by name
# results = sorted(compute_correlation(df_test), key=lambda x: x["col"], reverse=False)
# print(pd.DataFrame(results))


# results = sorted(compute_correlation(df_oof), key=lambda x: x["spearman"], reverse=True)
# print(pd.DataFrame(results))
#
# # sort by name
# results = sorted(compute_correlation(df_oof), key=lambda x: x["col"], reverse=False)
# print(pd.DataFrame(results))

print(f'{df_test.iloc[:, 0].name=}')

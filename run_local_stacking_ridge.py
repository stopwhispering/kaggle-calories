import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_log_error

from calories.constants import (
    PATH_TRAIN,
    PATH_PREDS_FOR_ENSEMBLES,
    PATH_ENSEMBLE_SUBMISSIONS,
    PATH_CACHE,
)
from my_ml_util.ensembling.hill_climbing import (
    HillClimber,
    score_multiple_rmse,
    score_multiple_rmsle,
)
import numpy as np
import time

from trainers.sklearn_trainer import SklearnTrainer


def get_oofs_and_test_preds():
    # get list of files that match pattern "_oof.pkl" in PATH_PREDS_FOR_ENSEMBLES
    oof_files = [file for file in PATH_PREDS_FOR_ENSEMBLES.glob("*_oof.pkl")]

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

        if len(ser_oof) == 794_868:
            ser_oof = ser_oof.iloc[:750_000]

        name = oof_file.stem.replace("_oof", "")
        ser_oof.name = name
        ser_test.name = name

        all_oof.append(ser_oof)
        all_test.append(ser_test)

    df_oof = pd.concat(all_oof, axis=1)
    df_test = pd.concat(all_test, axis=1)

    # df_oof = df_oof.clip(lower=1, upper=314)
    # df_test = df_test.clip(lower=1, upper=314)

    return df_oof, df_test


df_models_oof, df_models_test = get_oofs_and_test_preds()
df_models_oof = df_models_oof.iloc[:750_000]


import time
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error
import numpy as np
from calories.constants import (
    PATH_TRAIN,
    PATH_TEST,
    PATH_PREDS_FOR_ENSEMBLES,
    PATH_FEATURES,
    PATH_ORIGINAL,
)
from calories.preprocessing.dtypes import convert_sex
from feature_creators.read_features_util import read_features
from my_preprocessing.feature_store.feature_store import FeatureStore
from trainers.lgbm_trainer import LGBMTrainer
from trainers.metrics.metrics_lgbm import rmsle_lgbm
from trainers.xgb_trainer import XGBTrainer

time_start_overall = time.time()
df_train = pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
df_test = pd.read_csv(PATH_TEST).set_index("id")

df_train = convert_sex(df_train)
df_test = convert_sex(df_test)

feature_store = FeatureStore(
    path_features=PATH_FEATURES,
)
df_train_features, df_test_features = feature_store.read_features(
    column_names=[
        "Combine_Sex_Duration",
        "Multiply_Weight_Duration",
        "Plus_Age_Duration",
        "Multiply_Age_Duration",
        "GroupByThenMean_Age_Height",
        "Divide_Sex_Age",
    ],
)
df_train = pd.concat([df_train, df_train_features], axis=1)
df_test = pd.concat([df_test, df_test_features], axis=1)


# df_train = pd.concat([df_train, df_models_oof], axis=1)
# df_test = pd.concat([df_test, df_models_test], axis=1)

# for ridge, only preds as features
df_train = df_models_oof
df_test = df_models_test


print(f"{df_train.shape=}, {df_test.shape=}")
duration_loading_data = str(int(time.time() - time_start_overall))
print(f"{duration_loading_data=}")
time_start_training = time.time()

params_ridge = {"alpha": 1.8912865010023905, "tol": 0.0029820687995857588}

trainer = SklearnTrainer(
    params={"random_state": 42} | params_ridge,
    scoring_fn=root_mean_squared_log_error,
    log_transform_targets=True,  # True -> expm1 is applied to preds after predicting before scoring
    early_stop=False,
    clip_preds=(1.0, 314.0),
    model_class=Ridge,
)
(score, best_iterations, df_oof_predictions, df_test_predictions_from_oof, _, _) = (
    trainer.train_and_score(
        # df_train=df_train,
        df_train=np.log1p(df_train),
        # df_train=pd.concat([df_train, df_train_features_dummy], axis=1),
        # ser_targets_train=np.log1p(ser_targets_train),
        ser_targets_train=ser_targets_train,
        df_test=df_test,
    )
)
print(f"{score=:.5f}")
score = root_mean_squared_log_error(
    y_true=ser_targets_train,  # .iloc[:750_000],
    y_pred=df_oof_predictions["pred"],  # .iloc[:750_000],
)

duration_training = time.time() - time_start_training
print(f"{score=:.5f}, {best_iterations=}, {duration_training=}")

df_test_predictions_from_full = trainer.train_on_full(
    df_train=df_train,
    # df_train=pd.concat([df_train, df_train_features_dummy], axis=1),
    # ser_targets_train=np.log1p(ser_targets_train),
    ser_targets_train=ser_targets_train,
    df_test=df_test,
)

duration_all = str(int(time.time() - time_start_overall))
filename_prefix = __file__.split("\\")[-1][:-3]  # remove .py
if filename_prefix.startswith("run_"):
    filename_prefix = filename_prefix[4:]
df_oof_predictions["pred"].to_pickle(PATH_CACHE / f"{filename_prefix}_oof.pkl")
df_test_predictions_from_oof["pred"].to_pickle(
    PATH_CACHE / f"{filename_prefix}_test.pkl"
)
df_test_predictions_from_full["pred"].to_pickle(
    PATH_CACHE / f"{filename_prefix}_test_from_full.pkl"
)
open(PATH_CACHE / f"{filename_prefix}_{score=:.5f}_{duration_all=}", f"a").close()


df_submission = (
    df_test_predictions_from_full["pred"]
    .reset_index()
    .rename(columns={"pred": "Calories"})
)
df_submission["Calories"] = np.clip(df_submission["Calories"], 1, 314)
df_submission.to_csv(
    PATH_ENSEMBLE_SUBMISSIONS
    / f"stacking_ridge_{filename_prefix}_{score=:.5f}_{duration_all=}.csv",
    index=False,
)

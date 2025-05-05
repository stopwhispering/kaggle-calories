import time
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error
import numpy as np
from calories.constants import PATH_TRAIN, PATH_TEST, PATH_PREDS_FOR_ENSEMBLES
from calories.preprocessing.dtypes import convert_sex
from feature_creators.read_features_util import read_features
from feature_creators.read_nested_features_util import read_nested_features
from trainers.lgbm_trainer import LGBMTrainer
from trainers.metrics.metrics_lgbm import rmsle_lgbm
from trainers.xgb_trainer import XGBTrainer
import warnings

time_start_overall = time.time()
df_train = pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
df_test = pd.read_csv(PATH_TEST).set_index("id")

df_train = convert_sex(df_train)
df_test = convert_sex(df_test)


# df_train_features, df_test_features = read_features(column_names=None,  # all
#                                                     include_original=False,
#                                                     shuffle=True)
# _cat_cols = df_train_features.select_dtypes('object').columns.tolist()
# df_train_features[_cat_cols] = df_train_features[_cat_cols].astype('category')
# df_test_features[_cat_cols] = df_test_features[_cat_cols].astype('category')
# df_train = pd.concat([df_train, df_train_features], axis=1)
# df_test = pd.concat([df_test, df_test_features], axis=1)

nested_features_by_outer_fold: list[
    # column_names=column_names,
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
] = read_nested_features(
    agg="mean",
    # column_names=column_names,
    include_original=False,
)
# assert len(column_names) == len(nested_features_by_outer_fold[0][0].columns), f"{len(column_names)=} != {len(nested_features_by_outer_fold[0][0].columns)=}"

nested_features_column_names = nested_features_by_outer_fold[0][0].columns.tolist()
print(f"{len(nested_features_by_outer_fold)=}")
print(f"{len(nested_features_column_names)=}")

# we need one dataframe with dummy values for the FFS class
df_train_features_dummy = pd.DataFrame(
    data=1,
    index=df_train.index,
    columns=nested_features_column_names,
)


def add_target_encoding(df_train, df_val, df_test, ser_targets_train, i_fold: int):
    # todo separate module
    selected_columns = nested_features_column_names
    df_te_train_fold, df_te_val_fold, df_te_test_fold = nested_features_by_outer_fold[
        i_fold
    ]

    df_te_train_fold_selected = df_te_train_fold[selected_columns]
    df_te_val_fold_selected = df_te_val_fold[selected_columns]
    df_te_test_fold_selected = df_te_test_fold[selected_columns]

    # make sure the columns really are dummy, and make sure we don't forget a dummy...
    assert (df_train[selected_columns] == 1).all().all()
    assert set(df_train.columns[(df_train == 1).all()].tolist()) == set(
        selected_columns
    )

    df_train[selected_columns] = df_te_train_fold_selected
    df_val[selected_columns] = df_te_val_fold_selected
    if df_test is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            df_test[selected_columns] = df_te_test_fold_selected.values

    print(f"{df_train.shape=}")
    return df_train, df_val, df_test


print(f"{df_train.shape=}, {df_test.shape=}")


duration_loading_data = str(int(time.time() - time_start_overall))
print(f"{duration_loading_data=}")
time_start_training = time.time()

params_xgb = {
    "eval_metric": "rmsle",
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
    early_stop=True,
    use_gpu=True,
    fn_add_target_encoding=add_target_encoding,
    log_evaluation=100,
)
print(f"{pd.concat([df_train, df_train_features_dummy], axis=1).shape=}")
(score, best_iterations, df_oof_predictions, df_test_predictions, _, _) = (
    trainer.train_and_score(
        # df_train=df_train,
        df_train=pd.concat([df_train, df_train_features_dummy], axis=1),
        ser_targets_train=ser_targets_train,
        df_test=df_test,
    )
)
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

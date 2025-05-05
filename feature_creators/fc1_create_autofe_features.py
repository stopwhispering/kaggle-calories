import pandas as pd
from time import time

from calories.constants import PATH_TRAIN, PATH_TEST, PATH_FEATURES
from calories.preprocessing.dtypes import convert_sex
from my_preprocessing.fe import FeatureEngineer
import os
import pickle

time_start = time()
df_train = pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
df_test = pd.read_csv(PATH_TEST).set_index("id")

df_train = convert_sex(df_train)
df_test = convert_sex(df_test)

fe = FeatureEngineer()
potential_features = fe.find_potential_features(
    df_train,
    include_log=False,
    include_abs=False,
    max_nunique_if_numeric=200_000,
)

fe = fe.add_features(potential_features).fit(df_train)
df_train_fe = fe.transform(
    df_train, identify_highly_correlated_features=True, max_corr=0.9999999
)
print(f"{df_train_fe.shape=}")

df_test_fe = fe.transform(df_test)
print(f"{df_test_fe.shape=}")


def save(ser_train, ser_test, colname):
    filename_prefix = f"autofe_{colname}"
    metadata = {
        "column_name": colname,
        "filename_prefix": filename_prefix,
        "type": "autofe",
        "subtype": f"autofe {colname.split('_')[0]}",
        "description": f"AUTOFE feature.",
        "feature_list": [],  # maybe todo
        # "agg": "mean",
        # "kfold_splits": N_SPLITS,
        # "smooth": SMOOTH,
        "created_at": pd.Timestamp.now(),
        "created_by_script": os.path.basename(__file__),
    }

    if (PATH_FEATURES / f"{filename_prefix}_train.pkl").exists():
        print(f"WARNING: {filename_prefix} already exists. Overwriting now.")

    # todo somewhere convert from float64 to float32

    # save metadata to flat text file, with line breaks after each key; also pickle it
    with open(PATH_FEATURES / f"{filename_prefix}_metadata.txt", "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    with open(PATH_FEATURES / f"{filename_prefix}_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # save train/test
    ser_train.to_frame().to_parquet(PATH_FEATURES / f"{filename_prefix}_train.parquet")
    ser_test.to_frame().to_parquet(PATH_FEATURES / f"{filename_prefix}_test.parquet")
    # ser_train.to_pickle(PATH_FEATURES / f"{filename_prefix}_train.pkl")
    # ser_test.to_pickle(PATH_FEATURES / f"{filename_prefix}_test.pkl")


for colname in df_train_fe.columns:
    if colname not in df_train.columns:
        print(f"Saving {colname}")
        save(
            df_train_fe[colname],
            df_test_fe[colname],
            colname,
        )

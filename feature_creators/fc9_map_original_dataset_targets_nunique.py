

import pickle
import os
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from tqdm import tqdm

from calories.constants import PATH_TRAIN, PATH_TEST, PATH_FEATURES, PATH_ORIGINAL
import itertools

def save(
    ser_train: pd.Series,
    ser_test: pd.Series,
    feature_list: list[str],
    nan_quota: float,
    # n_bins: int,
    # strategy: str,
):
    assert ser_train.name == ser_test.name
    filename_prefix = ser_train.name
    metadata = {
        "column_name": ser_train.name,
        "filename_prefix": filename_prefix,
        "type": "map original targets nunique",
        "subtype": "map original targets nunique",
        "description": f"map original targets nunique",
        "feature_list": feature_list,
        # "agg": "mean",
        # "outer_folds": N_OUTER_FOLDS,
        # "inner_folds": N_INNER_FOLDS,
        # "smooth": SMOOTH,
        # "n_bins": n_bins,
        # "strategy": strategy,
        "nan_quota": nan_quota,
        "dtype": str(ser_train.dtype),
        "created_at": pd.Timestamp.now(),
        "created_by_script": os.path.basename(__file__),
        "fillna": False,
    }

    if (PATH_FEATURES / f"{filename_prefix}_metadata.txt").exists():
        print(f"WARNING: {filename_prefix} already exists. Overwriting now.")

    # save metadata to flat text file, with line breaks after each key; also pickle it
    with open(PATH_FEATURES / f"{filename_prefix}_metadata.txt", "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    with open(PATH_FEATURES / f"{filename_prefix}_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)  # noqa

    ser_train.to_frame().to_parquet(PATH_FEATURES / f"{filename_prefix}_train.parquet")
    ser_test.to_frame().to_parquet(PATH_FEATURES / f"{filename_prefix}_test.parquet")


if __name__ == "__main__":
    df_train = pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
    ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
    df_test = pd.read_csv(PATH_TEST).set_index("id")
    df_original = (
        pd.read_csv(PATH_ORIGINAL)
        .rename({"Gender": "Sex"}, axis=1)
        .drop(["User_ID"], axis=1)
    )

    # features = df_train.select_dtypes("number").columns.tolist()
    features = df_train.columns.tolist()

    potential_combinations = []
    for size in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        potential_combinations += list(
            itertools.combinations(features, size)
        )

    for feature_combination in tqdm(potential_combinations[::-1]):
        feature_combination = list(feature_combination)
        colname = f"map_orig_targets_nunique_{'_'.join(feature_combination)}"

        ser_original_concat = df_original[feature_combination].astype(str).agg("_".join, axis=1)
        ser_original_nunique = df_original.groupby(ser_original_concat)["Calories"].nunique()

        ser_train_concat = df_train[feature_combination].astype(str).agg("_".join, axis=1)
        ser_test_concat = df_test[feature_combination].astype(str).agg("_".join, axis=1)

        ser_map_orig_train = ser_train_concat.map(ser_original_nunique).rename(colname).astype("float32")
        ser_map_orig_test = ser_test_concat.map(ser_original_nunique).rename(colname).astype("float32")

        # checking nan quota
        ser_combined = pd.concat([ser_map_orig_train, ser_map_orig_test])
        nan_quota = ser_combined.isna().sum() / len(ser_combined)
        print(f"{colname=:<80}, {nan_quota=:f} (train/test)")

        save(
            ser_map_orig_train,
            ser_map_orig_test,
            feature_list=feature_combination,
            nan_quota=nan_quota,
        )
        #         break
        #     break
        # break

    # # results = []
    # for col in tqdm(df_train_poly.columns.tolist()):
    #     colname = f'poly_{col}'
    #     print(f"{col=} -> {colname=}")
    #
    #     ser_te_train = df_train_poly[col]
    #     ser_te_test = df_test_poly[col]
    #
    #     ser_te_train.name = colname
    #     ser_te_test.name = colname
    #

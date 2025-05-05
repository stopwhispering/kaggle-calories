# SEE https://www.kaggle.com/competitions/playground-series-s4e9/discussion/533961
# SEE https://www.kaggle.com/code/cdeotte/rapids-cuml-lasso-lb-0-72500-cv-0-72800

import pickle
import os
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from tqdm import tqdm

from calories.constants import PATH_TRAIN, PATH_TEST, PATH_FEATURES


def compute_binned(
    df_train: pd.DataFrame, df_test: pd.DataFrame, n_bins: int, strategy: str, col: str
) -> tuple[pd.Series, pd.Series]:
    column_name = f"bins_{col}_{n_bins}_{strategy}"
    # kbins can't handle nan and expects 2D array
    df_train_nona = df_train[[col]].dropna()
    df_test_nona = df_test[[col]].dropna()

    kbins = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy=strategy,
    )

    arr_binned_train = kbins.fit_transform(df_train_nona)
    arr_binned_test = kbins.transform(df_test_nona)

    # re-add nan values
    ser_binned_train = df_train[col].rename(column_name).copy()
    ser_binned_train.loc[df_train_nona.index] = arr_binned_train.flatten()

    ser_binned_test = df_test[col].rename(column_name).copy()
    ser_binned_test.loc[df_test_nona.index] = arr_binned_test.flatten()

    return ser_binned_train, ser_binned_test


def save(
    ser_train: pd.Series,
    ser_test: pd.Series,
    feature_list: list[str],
    n_bins: int,
    strategy: str,
):
    assert ser_train.name == ser_test.name
    filename_prefix = ser_train.name
    metadata = {
        "column_name": ser_train.name,
        "filename_prefix": filename_prefix,
        "type": "binning",
        "subtype": "binning with {n_bins} bins and {strategy} strategy",
        "description": f"binning with sklearn's KBinsDiscretizer",
        "feature_list": feature_list,
        # "agg": "mean",
        # "outer_folds": N_OUTER_FOLDS,
        # "inner_folds": N_INNER_FOLDS,
        # "smooth": SMOOTH,
        "n_bins": n_bins,
        "strategy": strategy,
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

    df_train = df_train.drop("Sex", axis=1)
    df_test = df_test.drop("Sex", axis=1)

    binning_columns = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

    n_total = 0
    for n_bins in tqdm(range(3, 21)):
        for strategy in ["uniform", "quantile", "kmeans"]:
            for col in binning_columns:
                ser_binned_train, ser_binned_test = compute_binned(
                    df_train,
                    df_test,
                    n_bins,
                    strategy,
                    col,
                )

                print(f"Saving {ser_binned_train.name}")
                save(
                    ser_binned_train,
                    ser_binned_test,
                    feature_list=[col],
                    n_bins=n_bins,
                    strategy=strategy,
                )
                n_total += 1
        #         break
        #     break
        # break

    print(f"{n_total=}")
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

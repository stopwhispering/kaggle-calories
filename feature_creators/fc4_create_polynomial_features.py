# SEE https://www.kaggle.com/competitions/playground-series-s4e9/discussion/533961
# SEE https://www.kaggle.com/code/cdeotte/rapids-cuml-lasso-lb-0-72500-cv-0-72800

import pickle
import os
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from calories.constants import PATH_TRAIN, PATH_TEST, PATH_FEATURES


if __name__ == "__main__":
    df_train = pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
    ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Calories"]
    df_test = pd.read_csv(PATH_TEST).set_index("id")

    df_train = df_train.drop("Sex", axis=1)
    df_test = df_test.drop("Sex", axis=1)

    def create_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
        poly = PolynomialFeatures(degree=3, include_bias=False)
        arr_poly = poly.fit_transform(df)

        df_poly = pd.DataFrame(
            arr_poly,
            columns=poly.get_feature_names_out(input_features=df.columns),
            index=df.index,
        )

        # we only want features like A*B^2, B*A^2 etc.
        # remove original features
        df_poly = df_poly.drop(columns=df.columns)

        # keep only features with at least one 2nd degree input feature -> remove a*b, a^3, etc.
        # df_poly = df_poly.loc[:, df_poly.columns.str.contains(r"^.*_2$")]
        # df_rmv = df_poly.loc[:, ~df_poly.columns.str.contains("2")]
        df_poly = df_poly.loc[:, df_poly.columns.str.contains("2")]

        # remove feature without interaction, i.e. without space in the name -> remove a^2
        # df_rmv = df_poly.loc[:, ~df_poly.columns.str.contains(" ")]
        df_poly = df_poly.loc[:, df_poly.columns.str.contains(" ")]

        # cleanup colnames: replace space with _ and remove ^
        df_poly.columns = df_poly.columns.str.replace(" ", "_").str.replace("^", "")
        return df_poly

    DTYPE = "float32"

    df_train_poly = create_polynomial_features(df_train).astype(DTYPE)
    df_test_poly = create_polynomial_features(df_test).astype(DTYPE)
    assert df_train_poly.shape[1] == df_test_poly.shape[1]

    # results = []
    for col in tqdm(df_train_poly.columns.tolist()):
        colname = f"poly_{col}"
        print(f"{col=} -> {colname=}")

        ser_te_train = df_train_poly[col]
        ser_te_test = df_test_poly[col]

        ser_te_train.name = colname
        ser_te_test.name = colname

        filename_prefix = colname
        metadata = {
            "column_name": colname,
            "filename_prefix": filename_prefix,
            "type": "polynomial",
            "subtype": "polynomial",
            "description": f"Polynomial with sklearn's PolynomialFeatures",
            # "feature_list": [],
            # "agg": "mean",
            # "outer_folds": N_OUTER_FOLDS,
            # "inner_folds": N_INNER_FOLDS,
            # "smooth": SMOOTH,
            "dtype": DTYPE,
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

        ser_te_train.to_frame().to_parquet(
            PATH_FEATURES / f"{filename_prefix}_train.parquet"
        )
        ser_te_test.to_frame().to_parquet(
            PATH_FEATURES / f"{filename_prefix}_test.parquet"
        )

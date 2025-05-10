# https://www.kaggle.com/competitions/playground-series-s5e5/discussion/578215

import pickle
import os
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from tqdm import tqdm

from calories.constants import PATH_TRAIN, PATH_TEST, PATH_FEATURES, PATH_ORIGINAL
import itertools

from calories.preprocessing.dtypes import convert_sex


def save(
    ser_train: pd.Series,
    ser_test: pd.Series,
    feature_list: list[str],
    # nan_quota: float,
    # n_bins: int,
    # strategy: str,
):
    assert ser_train.name == ser_test.name
    filename_prefix = ser_train.name
    metadata = {
        "column_name": ser_train.name,
        "filename_prefix": filename_prefix,
        "type": "masked feature",
        "subtype": "masked feature by sex",
        "description": f"masked feature by sex",
        "feature_list": feature_list,
        # "agg": "mean",
        # "outer_folds": N_OUTER_FOLDS,
        # "inner_folds": N_INNER_FOLDS,
        # "smooth": SMOOTH,
        # "n_bins": n_bins,
        # "strategy": strategy,
        # "nan_quota": nan_quota,
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

    df_train = convert_sex(df_train)
    df_test = convert_sex(df_test)
    gender_col = 'Sex'


    def get_interactions_onehot(df, feature, ) -> tuple[pd.Series, pd.Series]:

        # Create one-hot columns (no need for get_dummies - we know it's binary)
        df['Female'] = df[gender_col]  # 1 if female, 0 otherwise
        df['Male'] = 1 - df[gender_col]  # Inverse

        # Create interactions
        ser_male = (df[feature] * df['Male']).rename(f'masked_{feature}_x_Male')
        ser_female = (df[feature] * df['Female']).rename(f'masked_{feature}_x_Female')
        # df[f'{feature}_x_Male'] = df[feature] * df['Male']
        # df[f'{feature}_x_Female'] = df[feature] * df['Female']

        # Drop temporary one-hot columns (optional)
        # df.drop(['Male', 'Female'], axis=1, inplace=True)

        return ser_male, ser_female


    # Usage
    for feature in [c for c in df_train.columns if c != gender_col]:
        ser_male_train, ser_female_train = get_interactions_onehot(df_train, feature)
        ser_male_test, ser_female_test = get_interactions_onehot(df_test, feature)

        save(
            ser_male_train,
            ser_male_test,
            feature_list=[feature, gender_col],
        )

        save(
            ser_female_train,
            ser_female_test,
            feature_list=[feature, gender_col],
        )
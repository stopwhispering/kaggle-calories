# SEE https://www.kaggle.com/competitions/playground-series-s4e9/discussion/533961
# SEE https://www.kaggle.com/code/cdeotte/rapids-cuml-lasso-lb-0-72500-cv-0-72800

import itertools
import pickle
import os
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from calories.constants import PATH_TRAIN, PATH_TEST, PATH_FEATURES_BY_FOLD
from calories.preprocessing.dtypes import convert_sex
from my_preprocessing.target_encoding import CustomTargetEncoder


if __name__ == "__main__":
    df_train = (
        pd.read_csv(PATH_TRAIN).set_index("id").drop("Calories", axis=1)
    )
    ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")[
        "Calories"
    ]
    df_test = pd.read_csv(PATH_TEST).set_index("id")

    df_train = convert_sex(df_train)
    df_test = convert_sex(df_test)

    def get_useful_combinations_for_te(
        df_train,
        max_nunique=600_000,
        min_nunique=5,
        feature_sizes: tuple[int] = (1, 2, 3, 4, 5, 6, 7),
    ) -> list[list[str]]:
        ser_nunique_by_feature = df_train.nunique()

        potential_combinations = []
        for size in feature_sizes:
            potential_combinations += list(
                itertools.combinations(df_train.columns.tolist(), size)
            )

        potential_combinations = [list(c) for c in potential_combinations]
        print(f'{len(potential_combinations)=}')
        combinations = []
        for combination in potential_combinations:
            nuniques = int(ser_nunique_by_feature.loc[combination].product())
            if max_nunique >= nuniques >= min_nunique:
                combinations.append(combination)

        print(f"{combinations=}")
        print(f"{len(combinations)=}")

        return combinations

    combinations = get_useful_combinations_for_te(df_train)

    # N_SPLITS = 5  # default: 5

    N_OUTER_FOLDS = 5  # for training
    N_INNER_FOLDS = 4 # for TE
    SMOOTH = 0

    # results = []
    for feature_list in tqdm(combinations):
        print(f"{feature_list=}")
        df_train_current = df_train.copy()
        df_test_current = df_test.copy()

        train_folds = []
        val_folds = []
        test_folds = []
        for outer_fold_no, (train_idx, val_idx) in enumerate(KFold(shuffle=True, random_state=42, n_splits=N_OUTER_FOLDS).split(df_train)):
            df_train_current_fold = df_train_current.iloc[train_idx].copy()
            df_val_current_fold = df_train_current.iloc[val_idx].copy()

            te = CustomTargetEncoder(
                cols=feature_list,
                agg="mean",
                kfold_splits=KFold(shuffle=True, random_state=42, n_splits=N_INNER_FOLDS).split(df_train_current_fold),
                smooth=SMOOTH,
                fillna=False,
                )
            te.fit(df_train_current_fold, ser_targets_train)
            ser_te_train = te.transform(df_train_current_fold)
            ser_te_val = te.transform(df_val_current_fold)
            ser_te_test = te.transform(df_test_current)

            ser_te_train.name = f"{ser_te_train.name}_no_fillna"
            ser_te_val.name = f"{ser_te_val.name}_no_fillna"
            ser_te_test.name = f"{ser_te_test.name}_no_fillna"

            train_folds.append(ser_te_train)
            val_folds.append(ser_te_val)
            test_folds.append(ser_te_test)

        filename_prefix = f"{ser_te_train.name}_{N_OUTER_FOLDS}_outer_{N_INNER_FOLDS}_inner_folds"  # noqa
        metadata = {
            "column_name": ser_te_train.name,
            "filename_prefix": filename_prefix,
            "type": "te",
            "subtype": "te_mean without fillna",
            "description": f"Target encoding of {feature_list} with mean aggregation, {N_OUTER_FOLDS=} / {N_INNER_FOLDS=} without fillna",
            "feature_list": feature_list,
            "agg": "mean",
            "outer_folds": N_OUTER_FOLDS,
            "inner_folds": N_INNER_FOLDS,
            "smooth": SMOOTH,
            "created_at": pd.Timestamp.now(),
            "created_by_script": os.path.basename(__file__),

            "fillna": False,
        }

        if (PATH_FEATURES_BY_FOLD / f"{filename_prefix}_metadata.txt").exists():
            print(f"WARNING: {filename_prefix} already exists. Overwriting now.")

        # save metadata to flat text file, with line breaks after each key; also pickle it
        with open(PATH_FEATURES_BY_FOLD / f"{filename_prefix}_metadata.txt", "w") as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        with open(PATH_FEATURES_BY_FOLD / f"{filename_prefix}_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)  # noqa

        # save train/val/test
        assert len(train_folds) == len(val_folds) == len(val_folds) == N_OUTER_FOLDS
        for i in range(N_OUTER_FOLDS):
            ser_te_train = train_folds[i]
            ser_te_val = val_folds[i]
            ser_te_test = test_folds[i]
            ser_te_train.to_frame().to_parquet(PATH_FEATURES_BY_FOLD / f"{filename_prefix}_train_{i}.parquet")
            ser_te_val.to_frame().to_parquet(PATH_FEATURES_BY_FOLD / f"{filename_prefix}_val_{i}.parquet")
            ser_te_test.to_frame().to_parquet(PATH_FEATURES_BY_FOLD / f"{filename_prefix}_test_{i}.parquet")
            # ser_te_train.to_pickle(PATH_FEATURES_BY_FOLD / f"{filename_prefix}_train_{i}.pkl")
            # ser_te_val.to_pickle(PATH_FEATURES_BY_FOLD / f"{filename_prefix}_val_{i}.pkl")
            # ser_te_test.to_pickle(PATH_FEATURES_BY_FOLD / f"{filename_prefix}_test_{i}.pkl")

        # score = score_fn(
        #     df_train_current,
        #     ser_targets_train,
        #     df_test_current,
        # )
        # results.append((colname, feature_list, score))

    # df_results = pd.DataFrame(results)
    # print(df_results.sort_values(2, ascending=True))

    # df_results.to_pickle(PATH_INTERIM_RESULTS / "df_te_mean_results_45_features.pkl")
    # df_results.to_excel(PATH_INTERIM_RESULTS / "df_te_mean_results_45_features.xlsx")

    #
    # fe = FeatureEngineer()
    # potential_features = fe.find_potential_features(
    #     df_train,
    #     include_log=False,
    #     include_abs=False,
    #     max_nunique_if_numeric=8,
    # )
    # fe = fe.add_features(potential_features).fit(df_train)
    # df_train_fe = fe.transform(
    #     df_train, identify_highly_correlated_features=True, max_corr=0.99
    # )
    # print(f"{df_train_fe.shape=}")
    #
    # df_test_fe = fe.transform(df_test)
    # print(f"{df_test_fe.shape=}")

import gc
import logging

import pandas as pd
from tqdm import tqdm

from calories.constants import PATH_FEATURES_BY_FOLD, PATH_FEATURES_BY_FOLD_WITH_ORIGINAL
from my_preprocessing.dtype import DtypeOptimizer


def _read_all_metadata(
        column_names: list[str] | None = None,  # default: all
        feature_lists: list[list[str]] | None = None,  # default: all
        agg: str = "mean",
        include_original=False,
) -> list[dict]:
    assert agg=='mean', 'Not implemented, yet'
    # get all feature files' metadata (pattern ...metadata.pkl)
    if not include_original:
        file_paths = [
            f for f in PATH_FEATURES_BY_FOLD.glob("*metadata.pkl")
        ]
    else:
        file_paths = [
            f for f in PATH_FEATURES_BY_FOLD_WITH_ORIGINAL.glob("*metadata.pkl")
        ]

    # read all metadata files
    metadata = []
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            metadata.append(pd.read_pickle(f))

    if column_names is not None:
        metadata = [m for m in metadata if m["column_name"] in column_names]

    if feature_lists is not None:
        feature_lists = [set(f) for f in feature_lists]
        # metadata = [m for m in metadata if m["feature_list"] in feature_lists]
        metadata = [m for m in metadata if set(m["feature_list"]) in feature_lists]
        # any missing?  # {'Host_Popularity_percentage', 'Episode_Num'}
        # missing_feature_lists = [f for f in feature_lists if f not in [m["feature_list"] for m in metadata]]
        missing_feature_lists = [f for f in feature_lists if f not in [set(m["feature_list"]) for m in metadata]]
        assert not missing_feature_lists, f"Missing feature lists: {missing_feature_lists}"

        metadata = [m for m in metadata if m["agg"] == agg]  # todo other aggs
    # assert same outer_fold
    assert all(
        m["outer_folds"] == metadata[0]["outer_folds"] for m in metadata
    ), "All metadata must have the same outer folds"

    return metadata


def read_nested_features(
        column_names: list[str] | None = None,  # default: all
        feature_lists: list[list[str]] | None = None,  # default: all
        agg: str | list[str] | None = None,  # e.g. ['mean', 'nunique']  # noqa
        newest_first=True,
        shuffle=False,
        optimize_dtypes=False,
        from_parquet=True,
        include_original=False,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:

    logging.warning("USE CLASS IMPLEMENTATION")
    raise NotImplementedError("USE CLASS IMPLEMENTATION")

    metadata: list[dict] = _read_all_metadata(column_names=column_names,
                                              feature_lists=feature_lists,
                                              include_original=include_original,)

    # if we have duplicates (same column name), print warning and discard all but latest
    duplicates = {}
    for m in metadata:
        if m["column_name"] not in duplicates:
            duplicates[m["column_name"]] = []
        duplicates[m["column_name"]].append(m)

    for colname, items in duplicates.items():
        if len(items) > 1:
            print(f"WARNING: {colname} has {len(items)} duplicates. Discarding all but latest.")
            items.sort(key=lambda x: x["created_at"], reverse=True)
            for item in items[1:]:
                metadata.remove(item)

    if agg is not None:
        if not isinstance(agg, list):
            agg = [agg]
        metadata = [m for m in metadata if m.get("agg") in agg]
        # metadata = [m for m in metadata if m.get("agg") == agg]

    # sort by data, newest first
    if newest_first:
        metadata.sort(key=lambda x: x["created_at"], reverse=True)

    print(f'{len(metadata)=}')

    # logging.warning('WARN')
    # metadata = metadata[:500]
    # print(f'{len(metadata)=}')

    if shuffle:
        import random
        random.shuffle(metadata)

    path = PATH_FEATURES_BY_FOLD_WITH_ORIGINAL if include_original else PATH_FEATURES_BY_FOLD

    # df_train_features, df_test_features = None, None
    results_by_fold = []
    for i_outer_fold in tqdm(range(m["outer_folds"])):
        train_series, val_series, test_series = [], [], []
        for m in metadata:

            if from_parquet and (path / f"{m['filename_prefix']}_train_{i_outer_fold}.parquet").exists():
                ser_train: pd.Series = pd.read_parquet(path / f"{m['filename_prefix']}_train_{i_outer_fold}.parquet").iloc[:, 0]
                ser_val: pd.Series = pd.read_parquet(path / f"{m['filename_prefix']}_val_{i_outer_fold}.parquet").iloc[:, 0]
                try:
                    ser_test: pd.Series = pd.read_parquet(path / f"{m['filename_prefix']}_test_{i_outer_fold}.parquet").iloc[:, 0]
                except:
                    print(f"WARNING: {m['filename_prefix']}_test_{i_outer_fold}.parquet not found")
                    raise
            else:
                ser_train: pd.Series = pd.read_pickle(path / f"{m['filename_prefix']}_train_{i_outer_fold}.pkl")
                ser_val: pd.Series = pd.read_pickle(path / f"{m['filename_prefix']}_val_{i_outer_fold}.pkl")
                ser_test: pd.Series = pd.read_pickle(path / f"{m['filename_prefix']}_test_{i_outer_fold}.pkl")

            if optimize_dtypes:
                dtype = DtypeOptimizer.get_optimized_float_type(ser_train)
                ser_train = ser_train.astype(dtype)
                ser_val = ser_val.astype(dtype)
                ser_test = ser_test.astype(dtype)

            if not (path / f"{m['filename_prefix']}_train_{i_outer_fold}.parquet").exists():
                ser_train.to_frame().to_parquet(path / f"{m['filename_prefix']}_train_{i_outer_fold}.parquet")
                ser_val.to_frame().to_parquet(path / f"{m['filename_prefix']}_val_{i_outer_fold}.parquet")
                ser_test.to_frame().to_parquet(path / f"{m['filename_prefix']}_test_{i_outer_fold}.parquet")
                logging.warning(f"WARNING: Writing parquet: {m['filename_prefix']}")

            # mean = ser_train.mean()
            # ser_train = ser_train.fillna(mean)
            # ser_val = ser_val.fillna(mean)
            # ser_test = ser_test.fillna(mean)
            # logging.warn(f"WARNING: Filling NaN with mean")

            train_series.append(ser_train)
            val_series.append(ser_val)
            test_series.append(ser_test)

        assert all(
            (ser_train.index == train_series[0].index).all() for ser_train in train_series
        ), "Train series have different indices"
        assert all(
            (ser_test.index == test_series[0].index).all() for ser_test in test_series
        ), "Test series have different indices"

        df_train_features = pd.concat(train_series,  axis=1)
        df_train_features.index.name = "id"
        del train_series
        df_val_features = pd.concat(val_series, axis=1)
        df_val_features.index.name = "id"
        del val_series
        df_test_features = pd.concat(test_series, axis=1)
        df_test_features.index.name = "id"
        del test_series
        gc.collect()
        # make sure we don't have duplicate column names
        assert df_train_features.columns.is_unique, "Train features have duplicate column names"

        results_by_fold.append((df_train_features, df_val_features, df_test_features))

    return results_by_fold


if __name__ == '__main__':
    # # df_train_features, df_test_features = read_features_by_column_name(['ep_len_minutes_sin', 'te_MEAN_Host_Popularity_percentage_Number_of_Ads'])
    # df_train_features, df_test_features = read_features(['ep_len_minutes_sin', 'te_mean_ORIG_Episode_Length_minutes_Episode_Sentiment'])
    # # df_train_features, df_test_features = read_features_by_column_name()
    # print(df_train_features.shape)
    # print(df_test_features.shape)
    # print(df_test_features.columns)
    a = 1
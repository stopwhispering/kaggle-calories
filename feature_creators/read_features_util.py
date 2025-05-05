import pandas as pd
import logging

from calories.constants import PATH_FEATURES, PATH_FEATURES_WITH_ORIGINAL


def _read_all_metadata(
    column_names: list[str] | None = None,  # default: all
    type_=None,  # default: all
    include_original=False,
) -> list[dict]:
    # get all feature files' metadata (pattern ...metadata.pkl)

    path = PATH_FEATURES_WITH_ORIGINAL if include_original else PATH_FEATURES
    file_paths = [f for f in path.glob("*metadata.pkl")]

    # read all metadata files
    metadata = []
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            metadata.append(pd.read_pickle(f))

    if column_names is not None:
        metadata = [m for m in metadata if m["column_name"] in column_names]
        assert len(metadata) == len(column_names), (
            f"Missing metadata for columns: {set(column_names) - {m['column_name'] for m in metadata}}"
        )

    if type_ is not None:
        metadata = [m for m in metadata if m["type"] == type_]
        assert len(metadata) > 0, f"No metadata found for type: {type_}"

    return metadata


def read_features(
    column_names: list[str] | None = None,  # default: all
    newest_first=False,
    include_original=False,
    type_=None,  # default: all
    shuffle=False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.warning("USE CLASS IMPLEMENTATION")
    # raise NotImplementedError("USE CLASS IMPLEMENTATION")

    metadata: list[dict] = _read_all_metadata(
        column_names=column_names, include_original=include_original, type_=type_
    )

    # if we have a duplicates (same column name), print warning and discard all but latest
    duplicates = {}
    for m in metadata:
        if m["column_name"] not in duplicates:
            duplicates[m["column_name"]] = []
        duplicates[m["column_name"]].append(m)

    for colname, items in duplicates.items():
        if len(items) > 1:
            print(
                f"WARNING: {colname} has {len(items)} duplicates. Discarding all but latest."
            )
            items.sort(key=lambda x: x["created_at"], reverse=True)
            for item in items[1:]:
                metadata.remove(item)

    # sort by data, newest first
    if newest_first:
        metadata.sort(key=lambda x: x["created_at"], reverse=True)

    if shuffle:
        import random

        random.shuffle(metadata)

    path = PATH_FEATURES_WITH_ORIGINAL if include_original else PATH_FEATURES
    # df_train_features, df_test_features = None, None
    train_series, test_series = [], []
    for m in metadata:
        # print(f"{m['column_name']}: {m['filename_prefix']}")
        if (path / f"{m['filename_prefix']}_train.parquet").exists():
            # read parquet files
            ser_train = pd.read_parquet(
                path / f"{m['filename_prefix']}_train.parquet"
            ).iloc[:, 0]
            ser_test = pd.read_parquet(
                path / f"{m['filename_prefix']}_test.parquet"
            ).iloc[:, 0]
        else:
            # read pickle files
            ser_train = pd.read_pickle(path / f"{m['filename_prefix']}_train.pkl")
            ser_test = pd.read_pickle(path / f"{m['filename_prefix']}_test.pkl")

            # and save them as parquets
            ser_train.to_frame().to_parquet(
                path / f"{m['filename_prefix']}_train.parquet"
            )
            ser_test.to_frame().to_parquet(
                path / f"{m['filename_prefix']}_test.parquet"
            )
            logging.warning(
                f"Saved {m['filename_prefix']}_train.parquet and {m['filename_prefix']}_test.parquet"
            )

        train_series.append(ser_train)
        test_series.append(ser_test)

    assert all(
        (ser_train.index == train_series[0].index).all() for ser_train in train_series
    ), "Train series have different indices"
    assert all(
        (ser_test.index == test_series[0].index).all() for ser_test in test_series
    ), "Test series have different indices"

    df_train_features = pd.concat(train_series, axis=1).copy(deep=True)
    df_train_features.index.name = "id"

    df_test_features = pd.concat(test_series, axis=1).copy(deep=True)
    df_test_features.index.name = "id"

    # make sure we don't have duplicate column names
    assert df_train_features.columns.is_unique, (
        "Train features have duplicate column names"
    )

    return df_train_features, df_test_features


if __name__ == "__main__":
    # df_train_features, df_test_features = read_features_by_column_name(['ep_len_minutes_sin', 'te_MEAN_Host_Popularity_percentage_Number_of_Ads'])
    df_train_features, df_test_features = read_features(
        ["ep_len_minutes_sin", "te_mean_ORIG_Episode_Length_minutes_Episode_Sentiment"]
    )
    # df_train_features, df_test_features = read_features_by_column_name()
    print(df_train_features.shape)
    print(df_test_features.shape)
    print(df_test_features.columns)
    a = 1

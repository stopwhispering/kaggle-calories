from calories.constants import PATH_FEATURES, PATH_FEATURES_BY_FOLD
from my_preprocessing.feature_store.feature_store import FeatureStore

feature_store = FeatureStore(
    path_features=PATH_FEATURES,
    path_features_by_fold=PATH_FEATURES_BY_FOLD,
)

df_metadata, df_metadata_by_folds = feature_store.overview()
print(df_metadata)

df_train_features, df_test_features = feature_store.read_features(
    criteria={"type": ("autofe",), "subtype": ("autofe Minus",)}
)
print(f"{df_train_features.shape=}, {df_test_features.shape=}")


df_train_features, df_test_features = feature_store.read_features(
    criteria={"type": ("autofe",)}
)
print(f"{df_train_features.shape=}, {df_test_features.shape=}")

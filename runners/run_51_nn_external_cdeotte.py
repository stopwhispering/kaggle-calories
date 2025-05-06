# https://www.kaggle.com/code/cdeotte/nn-mlp-starter-cv-0-0608/output
from calories.constants import PATH_CACHE, PATH_PREDS_FOR_ENSEMBLES
import pandas as pd
import numpy as np

ser_train_sample = pd.read_pickle(PATH_PREDS_FOR_ENSEMBLES / '48_lgbm_from_47_sweep1_stilted-sweep-13_oof.pkl')

df_test_submission = pd.read_csv(PATH_CACHE / "cdeotte_51_nn_submission_v1.csv").rename({'Calories': 'pred'}, axis=1)
ser_test_submission = df_test_submission["pred"]
del df_test_submission

# tempdelme = pd.read_pickle(PATH_PREDS_FOR_ENSEMBLES / '46_catboost_from_44_add_binning_test.pkl')


arr_oof = np.load(PATH_CACHE / "cdeotte_51_nn_oof_v1.npy")
arr_oof = np.expm1(arr_oof)
ser_oof = pd.Series(arr_oof, name="pred", index=ser_train_sample.index)

filename_prefix = __file__.split("\\")[-1][:-3]  # remove .py
if filename_prefix.startswith("run_"):
    filename_prefix = filename_prefix[4:]
ser_oof.to_pickle(
    PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_oof.pkl"
)
df_test_submission["pred"].to_pickle(
    PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_test.pkl"
)
# open(
#     PATH_PREDS_FOR_ENSEMBLES / f"{filename_prefix}_{score=:.5f}_{duration_all=}", f"a"
# ).close()

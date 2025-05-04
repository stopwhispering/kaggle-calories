from pathlib import Path

PATH_BASE = Path(
    "/Users/Johannes/PycharmProjects/kaggle-calories_2025-5"
)

PATH_TRAIN = PATH_BASE / "competition_data/train.csv"
PATH_TEST = PATH_BASE / "competition_data/test.csv"
PATH_SAMPLE_SUBMISSION = PATH_BASE / "competition_data/sample_submission.csv"
PATH_ORIGINAL = PATH_BASE / "competition_data/calories.csv"

PATH_CACHE = PATH_BASE / "cache/"
PATH_PREDS_FOR_ENSEMBLES = PATH_BASE / "preds_for_ensembles/"
PATH_ENSEMBLE_SUBMISSIONS = PATH_BASE / "ensemble_submissions/"
PATH_FEATURES = PATH_BASE / "features/"
PATH_FEATURES_WITH_ORIGINAL = PATH_BASE / "features_with_original/"
PATH_FEATURES_BY_FOLD = PATH_BASE / "features_by_fold/"
PATH_FEATURES_BY_FOLD_WITH_ORIGINAL = PATH_BASE / "features_by_fold_with_original/"
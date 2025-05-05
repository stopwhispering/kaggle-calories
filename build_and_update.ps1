# ruff format .
poetry build

copy "./dist/calories-0.1.0-py3-none-any.whl" "./kaggle-dataset"

copy "./calories" "./kaggle-dataset/calories" -Recurse -Force

# required other projects
cd ../PreProcessing
./build_and_update.ps1
cd ../kaggle-calories_2025-5
copy "../PreProcessing/app/my_preprocessing-0.1.0-py3-none-any.whl" "./kaggle-dataset"

# cd ../FeatureSelection
# ./build_and_update.ps1
# cd ../kaggle-calories_2025-5
# copy "../FeatureSelection/app/my_feature_selection-0.1.0-py3-none-any.whl" "./kaggle-dataset"

cd ../my-ml-util
./build_and_update.ps1
cd ../kaggle-calories_2025-5
copy "../my-ml-util/app/my_ml_util-0.1.0-py3-none-any.whl" "./kaggle-dataset"


cd ./kaggle-dataset

# kaggle datasets metadata -d stopwhispering/calories-app
kaggle datasets version --dir-mode zip -p ./ -m "API Upload Wheel + src"

cd ..



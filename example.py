from taptap.taptap import Taptap
from taptap.exp_utils import lightgbm_hpo
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import r2_score
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def get_score(train_data, test_data, target_col, best_params):
    train_x = train_data.drop(columns=target_col).copy()
    test_x = test_data.drop(columns=target_col).copy()
    train_y = train_data[[target_col]]
    test_y = test_data[[target_col]]
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
    gbm = lgb.LGBMRegressor(**best_params)
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = pd.DataFrame(gbm.predict(test_x), index=test_x.index)
    score = r2_score(test_y, pred)
    return score, gbm


if __name__ == '__main__':
    data = fetch_california_housing(as_frame=True).frame
    target_col = 'MedHouseVal'
    task = 'regression'
    train_data, test_data, _, _ = train_test_split(
        data, data[[target_col]], test_size=0.25, random_state=42
    )
    best_params = lightgbm_hpo(
        data=train_data, target_col=target_col, task=task, n_trials=10, n_jobs=16
    )
    original_score, gbm = get_score(
        train_data, test_data, target_col=target_col, best_params=best_params
    )
    print("The score training by the original data is", original_score)

    model = Taptap(llm='ztphs980/taptap-distill',
                   experiment_dir='./experiment_taptap/',
                   steps=1000,
                   batch_size=8,
                   numerical_modeling='split',
                   gradient_accumulation_steps=2)

    # Fine-tuning
    model.fit(train_data, target_col=target_col, task=task)

    # Sampling
    synthetic_data = model.sample(n_samples=2 * train_data.shape[0],
                                  data=train_data,
                                  task=task,
                                  max_length=1024)

    # Label generation
    synthetic_data[target_col] = gbm.predict(synthetic_data.drop(columns=[target_col]))

    # Training using synthetic data
    new_score, _ = get_score(
        synthetic_data, test_data, target_col=target_col, best_params=best_params
    )
    print("The score training by the synthetic data is", new_score)








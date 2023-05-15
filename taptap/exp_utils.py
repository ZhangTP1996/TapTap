import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def lightgbm_hpo(data, target_col, task, n_trials, n_jobs):
    X = data.copy()
    y = data[[target_col]]
    del X[target_col]
    train_x, val_x, train_y, val_y = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    def objective(trial):
        param_grid = {
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "seed": 2022,
            "verbosity": -1,
            "n_jobs": n_jobs
        }

        if task == 'classification' and data[target_col].nunique() > 2:
            param_grid['num_class'] = data[target_col].nunique()
        if task == 'regression':
            gbm = lgb.LGBMRegressor(**param_grid)
        else:
            gbm = lgb.LGBMClassifier(**param_grid)
        gbm.fit(train_x, train_y,
                eval_set=[(val_x, val_y)],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        return gbm.best_score_['valid_0'].popitem(last=False)[-1]

    study = optuna.create_study(direction="minimize", study_name="LightGBM")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
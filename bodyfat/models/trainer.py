from typing import Tuple
 
from dvclive import Live
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor
 
from bodyfat.config import Config, load_config
from bodyfat.data import Dataset
 
config = load_config()
 
 
def evaluate(model: XGBRegressor, dataset: Dataset) -> Tuple[float, float]:
    predictions = model.predict(dataset.features)
    rmse = root_mean_squared_error(dataset.labels, predictions)
    r2 = r2_score(dataset.labels, predictions)
 
    return rmse, r2
 
 
def train_model(train_dataset: Dataset, val_dataset: Dataset):
    with Live(Config.Path.EXPERIMENTS_DIR) as live:
        model = XGBRegressor(
            n_estimators=config.model.n_estimators, max_depth=config.model.max_depth
        )
 
        model.fit(train_dataset.features, train_dataset.labels)
 
        train_rmse, train_r2 = evaluate(model, train_dataset)
        live.log_metric("train/rmse", train_rmse)
        live.log_metric("train/r2", train_r2)
 
        val_rmse, val_r2 = evaluate(model, val_dataset)
        live.log_metric("val/rmse", val_rmse)
        live.log_metric("val/r2", val_r2)
 
        model.save_model(Config.Path.MODELS_DIR / "model.json")
import re
import sqlite3
from pathlib import Path
 
import pandas as pd
from sklearn.model_selection import train_test_split
 
from bodyfat import load_config
 
 
def to_snake_case(s: str) -> str:
    s = re.sub(r"\W+", "_", s)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", s).lower()
    s = re.sub(r"_+", "_", s)
    return s.strip("_")
 
 
class DatasetBuilder:
    def __init__(self):
        self.config = load_config()
 
    def build(self, database_path: Path, output_dir: Path):
        df = self.read_sql(database_path)
        df = self.preprocess_dataframe(df)
        self.split_and_save(df, output_dir)
 
    @staticmethod
    def read_sql(data_path: Path) -> pd.DataFrame:
        with sqlite3.connect(data_path) as connection:
            return pd.read_sql_query("SELECT * FROM bodyfat", connection)
 
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop("Original", axis=1)
        df.columns = [to_snake_case(col) for col in df.columns]
        return df
 
    def split_and_save(self, df: pd.DataFrame, output_dir: Path):
        X = df.drop("body_fat", axis=1)
        y = df["body_fat"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.data.test_size, stratify=df["sex"]
        )
 
        self.save_subset(X_train, y_train, "train", output_dir)
        self.save_subset(X_test, y_test, "test", output_dir)
 
    def save_subset(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        subset: str,
        output_dir: Path,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output_dir / f"X_{subset}.parquet")
        labels.to_frame().to_parquet(output_dir / f"labels_{subset}.parquet")
 
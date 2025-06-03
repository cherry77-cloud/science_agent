import pandas as pd
from pathlib import Path
from sklearn.datasets import (
    load_diabetes,
    load_digits,
    load_wine,
    load_breast_cancer,
    load_linnerud,
)
from config import GENERATED_DATA_DIR
from agno.utils.log import logger


def load_and_save_toy_dataset(dataset_name_str: str) -> Path:
    loader_func_map = {
        "load_diabetes": load_diabetes,
        "load_digits": load_digits,
        "load_breast_cancer": load_breast_cancer,
        "load_linnerud": load_linnerud,
        "load_wine": load_wine,
    }
    if dataset_name_str not in loader_func_map:
        raise ValueError(f"Unknown toy dataset: {dataset_name_str}")

    data_bunch = loader_func_map[dataset_name_str]()
    # ... (DataFrame创建逻辑保持不变) ...
    df = pd.DataFrame(data_bunch.data, columns=data_bunch.feature_names)
    if hasattr(data_bunch, "target"):
        if data_bunch.target.ndim == 1:
            df["target"] = data_bunch.target
        elif data_bunch.target.ndim > 1 and hasattr(data_bunch, "target_names"):
            for i, name in enumerate(data_bunch.target_names):
                df[name] = data_bunch.target[:, i]

    # 修改保存路径
    # 使用 GENERATED_DATA_DIR
    file_name = f"{dataset_name_str.replace('load_', '')}.csv"
    save_path = GENERATED_DATA_DIR / file_name
    GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(save_path, index=False)
    logger.info(f"Toy dataset '{dataset_name_str}' saved to: {save_path}")
    return save_path

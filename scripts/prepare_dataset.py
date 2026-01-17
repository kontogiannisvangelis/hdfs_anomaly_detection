import yaml
from datasets import DatasetDict

from src.dataset.dataset_loader import load_csv, load_npz
from src.dataset.dataset_builder import create_hg_train_val_test
from src.dataset.tokenize import create_tokenizer, tokenize_dataset

def prepare_dataset(config_path: str = "../config/base.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_type = config["TYPE_OF_DATA"]
    model_name = config["MODEL_NAME"]
    tokenized_dataset_path = config["TOKENIZED_DATASET_PATH"]
    drop_duplicates = config["DROP_DUPLICATES"]

    if data_type == 0:
        df = load_csv(config["HDFS_FILE_PATH"], config["ANOMALY_LABEL_FILE_PATH"])
        splits = create_hg_train_val_test(df)

    elif data_type == 1:
        df = load_npz(
            path_npz=config["HDFS_FILE_PATH"],
            drop_duplicates=drop_duplicates
        )
        splits = create_hg_train_val_test(df)

    elif data_type == 2:
        df = load_npz(
            path_npz=config["HDFS_FILE_PATH"],
            path_event_templates=config["HDFS_TEMPLATES_FILE_PATH"],
            drop_duplicates=drop_duplicates
        )
        splits = create_hg_train_val_test(df)

    else:
        raise ValueError("Invalid TYPE_OF_DATA value")

    tokenizer = create_tokenizer(model_name)

    tokenized_splits = {
        name: tokenize_dataset(ds, tokenizer)
        for name, ds in splits.items()
    }

    DatasetDict(tokenized_splits).save_to_disk(tokenized_dataset_path)


if __name__ == "__main__":
    prepare_dataset()

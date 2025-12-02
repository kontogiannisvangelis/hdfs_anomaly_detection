from datasets import Dataset, ClassLabel

def create_hg_train_val_test(df, seed=42):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column(
        "label",
        ClassLabel(num_classes=2, names=["0", "1"])
    )

    train_temp = dataset.train_test_split(
        test_size=0.2,
        stratify_by_column="label",
        seed=seed
    )

    val_test = train_temp["test"].train_test_split(
        test_size=0.5,
        stratify_by_column="label",
        seed=seed
    )

    return {
        "train": train_temp["train"],
        "val": val_test["train"],
        "test": val_test["test"]
    }
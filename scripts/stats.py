import json
import os
import matplotlib.pyplot as plt

METRIC = "eval_f1"

DISTINGUISHING_HPARAMS = [
    "NUM_TRAIN_EPOCHS",
    "TRAIN_BATCH_SIZE",
    "EVAL_BATCH_SIZE",
    "LEARNING_RATE",
    "LORA_RANK",
    "LORA_ALPHA",
    "LORA_DROPOUT",
    "CLASS_WEIGHTS_ENABLED",
]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_run_label(hparams):
    parts = []
    for key in DISTINGUISHING_HPARAMS:
        if key in hparams:
            parts.append(f"{key}={hparams[key]}")
    return ", ".join(parts)


def extract_metric_over_epochs(train_history, metric):
    epochs = []
    values = []

    for entry in train_history:
        if metric in entry and "epoch" in entry:
            epochs.append(entry["epoch"])
            values.append(entry[metric])

    return epochs, values


def plot_from_folder(folder, metric=METRIC):
    plt.figure(figsize=(12, 7))
    plt.title(f"Metric Over Epochs: {metric}", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(metric, fontsize=14)

    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            path = os.path.join(folder, filename)
            data = load_json(path)

            hparams = data["hyperparameters"]
            train_history = data["train_history"]

            label = extract_run_label(hparams)

            epochs, metric_values = extract_metric_over_epochs(train_history, metric)

            if len(epochs) == 0:
                print(f"Metric '{metric}' not found in {filename}")
                continue

            plt.plot(epochs, metric_values, marker="o", label=label)

    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    FOLDER = "checkpoints/trainer"
    plot_from_folder(FOLDER)

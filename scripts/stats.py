import json
import os
import matplotlib.pyplot as plt

METRIC = "eval_f1"

DISTINGUISHING_HPARAMS = [
    "TRAIN_BATCH_SIZE",
    "LEARNING_RATE",
    "LORA_RANK",
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

def get_label_epoch_metric_values(folder, metric=METRIC):
    res = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            path = os.path.join(folder, filename)
            data = load_json(path)
            hparams = data["hyperparameters"]
            train_history = data["train_history"]
            label = extract_run_label(hparams)
            epoch, metric_values = extract_metric_over_epochs(train_history, metric)
            res.append((label, epoch, metric_values))
    return res

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
            plt.figure(figsize=(12, 10))
            plt.ylim(0.8, 0.95)
            plt.plot(epochs, metric_values, marker="o", label=label)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def find_max_for_each_label(folder, metric=METRIC):
    total = get_label_epoch_metric_values(folder, metric)
    p = []
    for t in total:
        p.append((max(t[2]), t[0]))
    p.sort(key=lambda x: x[0], reverse=True)
    for e in p:
        print(e)
def find_max(folder, metric=METRIC):
    total = get_label_epoch_metric_values(folder, metric)
    label = ""
    max_val = 0
    for t in total:
        max_local = max(t[2])
        if max_local > max_val:
            max_val = max_local
            label = t[0]
    print(label)
    print(max_val)




if __name__ == "__main__":
    FOLDER = "../checkpoints/trainer_16_2_26"
    find_max_for_each_label(FOLDER)

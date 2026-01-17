import os
import json
import numpy as np
import torch
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
import evaluate

from src.config.config_parger import get_arg_parser, load_yaml_config, merge_configs


class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, focal=False, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal = focal
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce_loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight, reduction="none")

        if self.focal:
            pt = torch.exp(-ce_loss)
            loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        else:
            loss = ce_loss.mean()

        return (loss, outputs) if return_outputs else loss


accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(predictions=preds, references=labels, average="binary")["precision"],
        "recall": recall.compute(predictions=preds, references=labels, average="binary")["recall"],
        "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
    }


def save_training_log(output_dir, cfg, trainer):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(output_dir, f"training_log_{timestamp}.json")

    log_data = {
        "hyperparameters": cfg,
        "train_history": trainer.state.log_history,  # contains train loss, eval loss, metrics
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)

    print(f"Training log saved to: {log_path}")

def print_conf(cfg):
    print("--------------------")
    print("ACTIVE CONFIGURATION")
    print("--------------------")
    for key, val in cfg.items():
        print(f"* {key}: {val}")

def main():
    parser = get_arg_parser()
    cli_args = parser.parse_args()

    yaml_cfg = load_yaml_config("../config/base.yaml")
    cfg = merge_configs(yaml_cfg, cli_args)

    print("--------------------")
    print("ACTIVE CONFIGURATION")
    print("--------------------")
    for key, val in cfg.items():
        print(f"* {key}: {val}")

    if cfg.get("DRY_RUN"):
        return

    ds = load_from_disk(cfg["TOKENIZED_DATASET_PATH"])
    train_ds, eval_ds = ds["train"], ds["val"]

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["MODEL_NAME"],
        num_labels=2,
        id2label={0: "Normal", 1: "Anomaly"},
        label2id={"Normal": 0, "Anomaly": 1},
    )

    lora_cfg = LoraConfig(
        r=cfg["LORA_RANK"],
        lora_alpha=cfg["LORA_ALPHA"],
        target_modules=["query", "value"],
        lora_dropout=cfg["LORA_DROPOUT"],
        bias="none",
        task_type="SEQ_CLS",
    )

    model = get_peft_model(model, lora_cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL_NAME"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if cfg.get("CLASS_WEIGHTS_ENABLED"):
        labels = np.array([int(x) for x in train_ds["label"]])
        weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        class_weights = torch.tensor(weights, dtype=torch.float)
    else:
        class_weights = None

    args = TrainingArguments(
        output_dir=cfg["TRAINER_CHECKPOINTS_PATH"],
        num_train_epochs=cfg["NUM_TRAIN_EPOCHS"],
        per_device_train_batch_size=cfg["TRAIN_BATCH_SIZE"],
        per_device_eval_batch_size=cfg["EVAL_BATCH_SIZE"],
        learning_rate=cfg["LEARNING_RATE"],
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=True
    )

    trainer = CustomTrainer(
        model=model.to("cuda" if torch.cuda.is_available() else "cpu"),
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        focal=cfg.get("FOCAL_ENABLED", False),
        gamma=cfg.get("GAMMA", 2.0)
    )

    print("----------------")
    print("FINE TUNE STARTS")
    print("----------------")
    trainer.train()
    save_training_log(cfg["TRAINER_CHECKPOINTS_PATH"], cfg, trainer) # TODO check this


if __name__ == "__main__":
    main()

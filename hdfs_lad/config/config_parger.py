import argparse
import yaml

def load_yaml_config(path):
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(yaml_cfg, cli_args):
    cfg = yaml_cfg.copy()
    for key, value in vars(cli_args).items():
        if value is not None:
            cfg[key] = value
    return cfg

def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--DRY_RUN", action="store_true")

    parser.add_argument("--CONFIG_FILE_PATH", type=str)


    # File paths
    parser.add_argument("--TOKENIZED_DATASET_PATH", type=str)

    # Training arguments
    parser.add_argument("--NUM_TRAIN_EPOCHS", type=int)
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int)
    parser.add_argument("--EVAL_BATCH_SIZE", type=int)
    parser.add_argument("--LEARNING_RATE", type=float)
    parser.add_argument("--TRAINER_CHECKPOINTS_PATH", type=str)

    # LoRA configuration
    parser.add_argument("--LORA_RANK", type=int)
    parser.add_argument("--LORA_ALPHA", type=int)
    parser.add_argument("--LORA_DROPOUT", type=float)

    # Weighted trainer options
    parser.add_argument("--CLASS_WEIGHTS_ENABLED", action="store_true")
    parser.add_argument("--FOCAL_ENABLED", action="store_true")
    parser.add_argument("--GAMMA", type=float)

    # Model
    parser.add_argument("--MODEL_NAME", type=str)

    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    cli_args = parser.parse_args()

    yaml_cfg = load_yaml_config("../../config/base.yaml")
    cfg = merge_configs(yaml_cfg, cli_args)
    print("--------------------")
    print("ACTIVE CONFIGURATION")
    print("--------------------")
    for key, val in cfg.items():
        print(f"* {key}: {val}")
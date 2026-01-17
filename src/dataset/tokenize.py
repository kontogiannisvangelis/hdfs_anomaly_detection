from transformers import AutoTokenizer

def create_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_dataset(split_dataset, tokenizer):
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    tokenized = split_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )

    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )
    return tokenized
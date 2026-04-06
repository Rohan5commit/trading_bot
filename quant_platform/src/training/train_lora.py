from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _as_int(value, default: int | None = None) -> int | None:
    if value is None:
        return default
    return int(value)


def _as_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    train_file = data_cfg.get("train_text_path") or data_cfg["corpus_text_path"]
    val_file = data_cfg.get("val_text_path")

    if val_file:
        dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})
        train_split = "train"
        eval_split = "validation"
    else:
        dataset = load_dataset("json", data_files=train_file)
        dataset = dataset["train"].train_test_split(test_size=0.05, seed=train_cfg.get("seed", 42))
        train_split = "train"
        eval_split = "test"

    max_train_samples = train_cfg.get("max_train_samples")
    if max_train_samples:
        train_count = min(int(max_train_samples), len(dataset[train_split]))
        dataset[train_split] = dataset[train_split].shuffle(seed=train_cfg.get("seed", 42)).select(range(train_count))

    max_eval_samples = train_cfg.get("max_eval_samples")
    if max_eval_samples:
        eval_count = min(int(max_eval_samples), len(dataset[eval_split]))
        dataset[eval_split] = dataset[eval_split].shuffle(seed=train_cfg.get("seed", 42) + 1).select(range(eval_count))

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if model_cfg.get("use_4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        dtype=torch.bfloat16 if train_cfg.get("bf16") else torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    peft_config = LoraConfig(
        r=model_cfg["lora_rank"],
        lora_alpha=model_cfg["lora_alpha"],
        lora_dropout=model_cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=model_cfg["target_modules"],
    )

    output_dir = train_cfg["output_dir"]
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=_as_int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=_as_int(
            train_cfg.get("per_device_eval_batch_size", train_cfg["per_device_train_batch_size"])
        ),
        gradient_accumulation_steps=_as_int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=_as_float(train_cfg["learning_rate"]),
        warmup_ratio=_as_float(train_cfg["warmup_ratio"]),
        num_train_epochs=_as_float(train_cfg["epochs"]),
        logging_steps=_as_int(train_cfg["logging_steps"]),
        eval_steps=_as_int(train_cfg["eval_steps"]),
        save_steps=_as_int(train_cfg["save_steps"]),
        max_steps=_as_int(train_cfg.get("max_steps", -1), -1),
        bf16=train_cfg.get("bf16", False),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        lr_scheduler_type=train_cfg.get("scheduler", "cosine"),
        report_to=["wandb"] if os.getenv("WANDB_API_KEY") else [],
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        dataset_text_field="text",
        max_length=model_cfg["max_seq_length"],
        packing=train_cfg.get("packing", False),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset[train_split],
        eval_dataset=dataset[eval_split],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": model_cfg["base_model"],
        "lora_rank": model_cfg["lora_rank"],
        "lora_alpha": model_cfg["lora_alpha"],
        "epochs": train_cfg["epochs"],
        "max_steps": train_cfg.get("max_steps", -1),
        "train_samples": len(dataset[train_split]),
        "val_samples": len(dataset[eval_split]),
    }
    with open(Path(output_dir) / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()

import logging
import os
import shutil
import sys
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, HfArgumentParser,
                          Trainer, TrainingArguments)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_to_train: str = field(
        default="/home/rickzhou/function_calling/models/starcoder2-7b"
    )
    seq_len: int = field(default=2048)
    attention_type: str = field(default="flash_attention_2")

    # lora configs
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = field(default=0.05)


def prepare_dataset(tokenizer, max_length, split="train"):
    """Load and prepare the dataset with chat template."""
    dataset = load_dataset(
        "riczhou/hermes-function-calling-v1-glaive-split", split=split
    )

    def format_conversation(example):
        """Format conversation using ChatML template."""
        try:
            conversation = ""
            for message in example:
                if (
                    not isinstance(message, dict)
                    or "from" not in message
                    or "value" not in message
                ):
                    logger.warning(f"Skipping malformed message: {message}")
                    continue

                role = message["from"]
                if role == "system":
                    conversation += (
                        f"<|im_start|>system\n{message['value']}<|im_end|>\n"
                    )
                elif role == "human":
                    conversation += f"<|im_start|>user\n{message['value']}<|im_end|>\n"
                elif role == "gpt":
                    conversation += (
                        f"<|im_start|>assistant\n{message['value']}<|im_end|>\n"
                    )
            return conversation
        except Exception as e:
            logger.error(f"Error formatting conversation: {e}")
            return ""

    def tokenize_function(examples):
        conversations = [format_conversation(ex) for ex in examples["conversations"]]
        # Filter out empty conversations
        conversations = [conv for conv in conversations if conv]
        return tokenizer(
            conversations,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset


def main():
    if len(sys.argv) == 1:
        raise ValueError("Missing configuration file.")

    torch.cuda.empty_cache()

    config_file = sys.argv[1] if len(sys.argv) == 2 else sys.argv[2]
    parser = HfArgumentParser((ModelConfig, TrainingArguments))
    model_config, training_args = parser.parse_json_file(json_file=config_file)

    logger.info(f"Base model: {model_config.model_to_train}")
    logger.info(f"Saving to: {training_args.output_dir}")

    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Try flash attention first, fall back to default if not available
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_to_train,
            attn_implementation=model_config.attention_type,
            max_memory={0: "22GB", 1: "22GB"},
            **model_kwargs,
        )
    except Exception as e:
        logger.warning(
            f"Flash attention failed: {e}. Falling back to default attention."
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_to_train,
            **model_kwargs,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_to_train, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=model_config.lora_target_modules,
        lora_dropout=model_config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    train_dataset = prepare_dataset(tokenizer, model_config.seq_len, split="train")
    eval_dataset = prepare_dataset(tokenizer, model_config.seq_len, split="validation")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    # Copy custom modeling file if it exists
    source_file = os.path.join(model_config.model_to_train, "modeling_custom.py")
    if os.path.exists(source_file):
        destination_file = os.path.join(training_args.output_dir, "modeling_custom.py")
        try:
            shutil.copy2(source_file, destination_file)
        except Exception as e:
            logger.error(f"Error copying modeling_custom.py: {e}")


if __name__ == "__main__":
    main()

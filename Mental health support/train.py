import argparse
from typing import Dict, List

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

SUPPORTIVE_SYSTEM_PROMPT = (
    "You are a compassionate mental wellness support assistant. "
    "Respond gently, validate feelings, suggest simple coping steps, and avoid judgment. "
    "Do not provide medical diagnosis or emergency guarantees."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a supportive chatbot on EmpatheticDialogues")
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    parser.add_argument("--output_dir", type=str, default="./mental_health_model")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=float, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    return parser.parse_args()


def build_examples(split_dataset, split_name: str) -> List[str]:
    """Create user->assistant training examples from consecutive utterances in the same conversation."""
    conversations: Dict[int, List[Dict]] = {}

    for row in split_dataset:
        conv_id = int(row["conv_id"])
        conversations.setdefault(conv_id, []).append(
            {
                "utterance_idx": int(row["utterance_idx"]),
                "speaker_idx": int(row["speaker_idx"]),
                "utterance": row["utterance"].strip(),
            }
        )

    texts: List[str] = []
    for _, turns in conversations.items():
        turns = sorted(turns, key=lambda x: x["utterance_idx"])
        for i in range(len(turns) - 1):
            current_turn = turns[i]
            next_turn = turns[i + 1]
            if current_turn["speaker_idx"] == next_turn["speaker_idx"]:
                continue

            user_text = current_turn["utterance"]
            assistant_text = next_turn["utterance"]

            if not user_text or not assistant_text:
                continue

            sample = (
                f"System: {SUPPORTIVE_SYSTEM_PROMPT}\n"
                f"User: {user_text}\n"
                f"Assistant: {assistant_text}"
            )
            texts.append(sample)

    print(f"Built {len(texts)} samples for split: {split_name}")
    return texts


def tokenize_examples(texts: List[str], tokenizer, max_length: int):
    dataset = Dataset.from_dict({"text": texts})

    def _tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])
    return tokenized


def main():
    args = parse_args()

    raw = load_dataset("empathetic_dialogues")

    train_texts = build_examples(raw["train"], "train")
    valid_texts = build_examples(raw["validation"], "validation")

    if args.max_train_samples:
        train_texts = train_texts[: args.max_train_samples]
    if args.max_eval_samples:
        valid_texts = valid_texts[: args.max_eval_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = tokenize_examples(train_texts, tokenizer, args.max_length)
    eval_dataset = tokenize_examples(valid_texts, tokenizer, args.max_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Model and tokenizer saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

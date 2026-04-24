import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a kind and supportive mental wellness assistant. "
    "Use calm and validating language. Keep responses practical and gentle. "
    "Do not diagnose conditions or provide emergency guarantees."
)


def build_prompt(user_text: str) -> str:
    return f"System: {SYSTEM_PROMPT}\nUser: {user_text}\nAssistant:"


def generate_reply(model, tokenizer, user_text: str, max_new_tokens: int = 120) -> str:
    prompt = build_prompt(user_text)
    inputs = tokenizer(prompt, return_tensors="pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    reply = generated.split("Assistant:")[-1].strip()
    return reply


def main():
    parser = argparse.ArgumentParser(description="Mental Health Support Chatbot CLI")
    parser.add_argument("--model_path", type=str, default="./mental_health_model")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    print("Mental Health Support Chatbot")
    print("Type 'exit' to quit.\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Take care. You are not alone.")
            break

        reply = generate_reply(model, tokenizer, user_text)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()

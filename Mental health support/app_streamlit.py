import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a kind and supportive mental wellness assistant. "
    "Use calm and validating language. Keep responses practical and gentle. "
    "Do not diagnose conditions or provide emergency guarantees."
)


def build_prompt(user_text: str) -> str:
    return f"System: {SYSTEM_PROMPT}\nUser: {user_text}\nAssistant:"


@st.cache_resource
def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device


def generate_reply(tokenizer, model, device, user_text: str) -> str:
    prompt = build_prompt(user_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated.split("Assistant:")[-1].strip()


def main():
    st.set_page_config(page_title="Mental Health Support Bot", page_icon="MH")
    st.title("Mental Health Support Chatbot")
    st.caption("Supportive conversation demo fine-tuned on EmpatheticDialogues")

    model_path = st.sidebar.text_input("Model path", "./mental_health_model")

    st.sidebar.info(
        "This bot is for emotional support and wellness reflection. "
        "For emergencies, contact local emergency services or a crisis hotline."
    )

    tokenizer, model, device = load_model(model_path)

    user_text = st.text_area("How are you feeling today?")
    if st.button("Get Supportive Reply"):
        if not user_text.strip():
            st.warning("Please enter a message first.")
        else:
            with st.spinner("Thinking with care..."):
                reply = generate_reply(tokenizer, model, device, user_text.strip())
            st.markdown(f"**Bot:** {reply}")


if __name__ == "__main__":
    main()

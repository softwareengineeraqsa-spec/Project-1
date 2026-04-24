# Mental Health Support Chatbot (Fine-Tuned)

This project fine-tunes a small language model on **EmpatheticDialogues** to produce supportive, emotionally aware responses.

## What is included

- `train.py`: Fine-tunes a base model (`distilgpt2` by default) using Hugging Face `Trainer`.
- `chat_cli.py`: Command-line chatbot for interactive testing.
- `app_streamlit.py`: Optional Streamlit web UI.
- `requirements.txt`: Python dependencies.

## Recommended base model

Default: `distilgpt2` (small and easy to run).  
You can change to another compatible causal LM (for example GPT-Neo variants) with `--model_name`.

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Fine-tune

```bash
python train.py \
  --model_name distilgpt2 \
  --output_dir ./mental_health_model \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 5e-5 \
  --max_length 256
```

Notes:
- The script uses the Hugging Face dataset `empathetic_dialogues`.
- It builds training pairs from conversational turns and prepends a gentle support system prompt.

## 3) Run CLI chatbot

```bash
python chat_cli.py --model_path ./mental_health_model
```

Type `exit` to quit.

## 4) Run Streamlit UI (optional)

```bash
streamlit run app_streamlit.py
```

## Safety reminder

This bot is for **emotional support only**, not crisis care or medical diagnosis.
If a user shows signs of immediate danger, direct them to local emergency services or a crisis hotline.

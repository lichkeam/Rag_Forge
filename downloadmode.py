from transformers import pipeline
import torch

print(f"Using GPU: {torch.cuda.get_device_name(0)}")

pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Who are you?"},
]

output = pipe(
    messages,
    max_new_tokens=128,  # ★ 指定合理的輸出長度
    do_sample=True,  # ★ 讓回覆自然
    temperature=0.7,
)

print(output)

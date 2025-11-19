from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.2-3B-Instruct"

# 只下載模型和 tokenizer 到本地
AutoTokenizer.from_pretrained(model_name, cache_dir="./llama3_model")
AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./llama3_model", low_cpu_mem_usage=True)

# 先讀 config
config = AutoConfig.from_pretrained(model_name)

# 覆蓋 rope_scaling
config.rope_scaling = {"type": "linear", "factor": 32.0}

# 再載入模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model loaded successfully!")

# 測試生成
prompt = "Explain what is machine learning in one sentence:"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("\nGenerating...")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nResult:\n{result}")

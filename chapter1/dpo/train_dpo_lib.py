# train_dpo_lib.py
# accelerate launch train_dpo_lib.py
# import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)
# print(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = DPOConfig(output_dir=f"DPO",
                          logging_steps=10,
                          per_device_train_batch_size=16,
                          use_cpu=True)
print(training_args.device)

trainer = DPOTrainer(model=model, args=training_args,
                     processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
from transformers import AutoTokenizer, AutoModelForCausalLM

# load the model and tokenizer
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# print model
print(model)
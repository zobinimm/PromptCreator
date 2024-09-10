from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

current_directory = os.path.dirname(__file__)
local_model_path = os.path.join(current_directory, 'Helsinki-NLP', 'opus-mt-zh-en')

model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

def translate(text):
    with torch.no_grad():
        encoded = tokenizer([text], return_tensors='pt')
        sequences = model.generate(**encoded)
        return tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]

input = "青春不能回头，所以青春没有终点"
print(input, translate(input))
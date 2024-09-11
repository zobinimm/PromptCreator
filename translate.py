from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# 获取当前文件的目录
current_directory = os.path.dirname(__file__)

# 加载中文到英文的模型和分词器
local_model_path_en = os.path.join(current_directory, 'Helsinki-NLP', 'opus-mt-zh-en')
model_en = AutoModelForSeq2SeqLM.from_pretrained(local_model_path_en)
tokenizer_en = AutoTokenizer.from_pretrained(local_model_path_en)

# 加载英文到中文的模型和分词器
local_model_path_zh = os.path.join(current_directory, 'Helsinki-NLP', 'opus-mt-en-zh')
model_zh = AutoModelForSeq2SeqLM.from_pretrained(local_model_path_zh)
tokenizer_zh = AutoTokenizer.from_pretrained(local_model_path_zh)

def translate_en(text):
    with torch.no_grad():
        encoded = tokenizer_en([text], return_tensors='pt')
        sequences = model_en.generate(**encoded)
        return tokenizer_en.batch_decode(sequences, skip_special_tokens=True)[0]

def translate_zh(text):
    with torch.no_grad():
        encoded = tokenizer_zh([text], return_tensors='pt')
        sequences = model_zh.generate(**encoded)
        return tokenizer_zh.batch_decode(sequences, skip_special_tokens=True)[0]

input_text = "青春不能回头，所以青春没有终点"
translated_text = translate_en(input_text)
back_to_zh = translate_zh(translated_text)

print(f"原文: {input_text}")
print(f"英文翻译: {translated_text}")
print(f"翻译回中文: {back_to_zh}")

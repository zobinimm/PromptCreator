import hashlib
import hmac
import json
import os
import string
import time
from collections import Counter
from collections import deque
from datetime import datetime
from typing import List
from pypinyin import pinyin, Style

import ChatTTS
import numpy as np
import pyJianYingDraft as draft
import requests
import spacy
import torch
from flask import Flask, request, jsonify
from pyJianYingDraft import trange
from scipy.io import wavfile
from spacy.matcher import Matcher
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from libs.gender_predictor.Naive_Bayes_Gender.gender import Gender
from genderize import Genderize

current_directory = os.path.dirname(__file__)

# 加载中文到英文的模型和分词器
local_model_path_en = os.path.join(current_directory, 'Helsinki-NLP', 'opus-mt-zh-en')
model_en = AutoModelForSeq2SeqLM.from_pretrained(local_model_path_en)
tokenizer_en = AutoTokenizer.from_pretrained(local_model_path_en)

# 加载英文到中文的模型和分词器
local_model_path_zh = os.path.join(current_directory, 'Helsinki-NLP', 'opus-mt-en-zh')
model_zh = AutoModelForSeq2SeqLM.from_pretrained(local_model_path_zh)
tokenizer_zh = AutoTokenizer.from_pretrained(local_model_path_zh)

# 记录请求时间的队列
request_times = deque()

# 调用限制
MAX_REQUESTS = 5
TIME_WINDOW = 1  # 时间窗口，单位为秒
STRIP_CHARS = "，"
SENTENCE_LIMIT = 24

# API:create_film_item
def sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

# 创建 Flask 应用程序
app = Flask(__name__)

# 预加载 spaCy 的 Transformer 模型
nlp_en = spacy.load("en_core_web_trf")
nlp_zh = spacy.load("zh_core_web_trf")

# 初始化 Matcher
matcher = Matcher(nlp_en.vocab)

# 定义模式以匹配更多描述词
patterns = [
    [{"POS": "ADJ"}],  # 单个形容词
    [{"POS": "ADJ"}, {"POS": "NOUN"}],  # 形容词 + 名词
    [{"POS": "NOUN"}],  # 单个名词
    [{"POS": "ADJ", "OP": "*", "TAG": "JJ"}, {"POS": "NOUN", "OP": "*"}]  # 形容词及名词短语
]

# 添加每个模式到 matcher
for i, pattern in enumerate(patterns):
    matcher.add(f"DESCRIPTION_PATTERN_{i}", [pattern])


def extract_description(text) -> List[str]:
    doc = nlp_en(text)
    matches = matcher(doc)
    descriptions = []

    # 按顺序提取匹配的文本
    for match_id, start, end in matches:
        span = doc[start:end]
        descriptions.append((start, span.text.lower()))  # 记录位置和描述词

    # 去除包含关系的重复项，保留最完整的描述词
    descriptions.sort(key=lambda x: x[0])  # 按位置排序

    unique_descriptions = []

    for i, (pos, desc) in enumerate(descriptions):
        # 检查是否已有描述词包含当前描述词
        has_flag = False
        for j, (pos1, desc1) in enumerate(descriptions):
            if i != j and desc != desc1:
                if desc in desc1:
                    has_flag = True
                    break
        if not has_flag:
            # 添加当前描述词，并移除包含当前描述词的其他描述词
            unique_descriptions = [existing for existing in unique_descriptions if
                                   not (desc in existing and len(desc) < len(existing))]
            if desc not in unique_descriptions:
                unique_descriptions.append(desc)

    return unique_descriptions

def translate_text(text: str, module: str, source_lang: str, target_lang: str, secret_id: str, secret_key: str) -> str:
    if module == "tencent":
        return tencent_translate_text(text, source_lang, target_lang, secret_id, secret_key)
    elif module == "custom1":
        if source_lang == "zh":
            return custom1_translate_en(text)
        elif source_lang == "en":
            return custom1_translate_zh(text)

def custom1_translate_en(text):
    with torch.no_grad():
        encoded = tokenizer_en([text], return_tensors='pt')
        sequences = model_en.generate(**encoded)
        return tokenizer_en.batch_decode(sequences, skip_special_tokens=True)[0]

def custom1_translate_zh(text):
    with torch.no_grad():
        encoded = tokenizer_zh([text], return_tensors='pt')
        sequences = model_zh.generate(**encoded)
        return tokenizer_zh.batch_decode(sequences, skip_special_tokens=True)[0]

def process_clauses(clauses, min_length=5, max_length=40):
    new_clauses = []
    temp_clause = ""

    for clause in clauses:
        if not any('\u4e00' <= char <= '\u9fff' for char in clause):
            continue
        if temp_clause:
            clause = temp_clause + clause
            temp_clause = ""
        if len(clause) < min_length:
            temp_clause = clause
        elif len(clause) > max_length:
            new_clauses.extend(fix_sentence_len(clause, max_length))
        else:
            new_clauses.append(clause)
    if temp_clause and new_clauses:
        new_clauses[-1] += temp_clause
    elif temp_clause:
        new_clauses.append(temp_clause)
    return new_clauses

def fix_sentence_len(input_text, max_length=40):
    final_sentences = []
    final_sentence = input_text
    while len(final_sentence) > max_length:
        cut_point = final_sentence.rfind(STRIP_CHARS, 0, max_length)
        if cut_point == -1:
            cut_point = max_length
        final_sentences.append(final_sentence[:cut_point].strip())
        final_sentence = final_sentence[cut_point:].lstrip(STRIP_CHARS).strip()
    if final_sentence:
        final_sentences.append(final_sentence)
    return final_sentences

def tencent_translate_text(text: str, source_lang: str, target_lang: str, secret_id: str, secret_key: str) -> str:
    global request_times

    # 记录当前时间
    current_time = time.time()

    # 清理队列中超过时间窗口的请求记录
    while request_times and current_time - request_times[0] > TIME_WINDOW:
        request_times.popleft()

    # 检查是否超过请求限制
    if len(request_times) >= MAX_REQUESTS:
        # 超过限制，等待直到可以再次请求
        wait_time = TIME_WINDOW - (current_time - request_times[0])
        print(f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds.")
        time.sleep(wait_time)
        # 重新记录时间
        current_time = time.time()

    # 记录请求时间
    request_times.append(current_time)

    token = ""
    service = "tmt"
    host = "tmt.tencentcloudapi.com"
    region = "ap-beijing"
    version = "2018-03-21"
    action = "TextTranslate"
    payload = f'{{"SourceText":"{text}","Source":"{source_lang}","Target":"{target_lang}","ProjectId":0}}'
    params = json.loads(payload)
    endpoint = "https://tmt.tencentcloudapi.com"
    algorithm = "TC3-HMAC-SHA256"
    timestamp = int(time.time())
    date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")

    # ************* 步骤 1：拼接规范请求串 *************
    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""
    ct = "application/json; charset=utf-8"
    canonical_headers = "content-type:%s\nhost:%s\nx-tc-action:%s\n" % (ct, host, action.lower())
    signed_headers = "content-type;host;x-tc-action"
    hashed_request_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = (http_request_method + "\n" +
                         canonical_uri + "\n" +
                         canonical_querystring + "\n" +
                         canonical_headers + "\n" +
                         signed_headers + "\n" +
                         hashed_request_payload)

    # ************* 步骤 2：拼接待签名字符串 *************
    credential_scope = date + "/" + service + "/" + "tc3_request"
    hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = (algorithm + "\n" +
                      str(timestamp) + "\n" +
                      credential_scope + "\n" +
                      hashed_canonical_request)

    # ************* 步骤 3：计算签名 *************
    secret_date = sign(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = sign(secret_date, service)
    secret_signing = sign(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    # ************* 步骤 4：拼接 Authorization *************
    authorization = (algorithm + " " +
                     "Credential=" + secret_id + "/" + credential_scope + ", " +
                     "SignedHeaders=" + signed_headers + ", " +
                     "Signature=" + signature)

    # ************* 步骤 5：构造并发起请求 *************
    headers = {
        "Authorization": authorization,
        "Content-Type": "application/json; charset=utf-8",
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Version": version
    }
    if region:
        headers["X-TC-Region"] = region
    if token:
        headers["X-TC-Token"] = token

    try:
        # 使用 requests 库发送 POST 请求
        response = requests.post(endpoint, headers=headers, data=payload)
        response_json = response.json()
        # 从 Response 对象中提取 TargetText 字段
        target_text = response_json.get('Response', {}).get('TargetText', '字段未找到')
        return target_text
    except requests.RequestException as err:
        print(f"Error: {err}")

@app.route('/createfilmchar', methods=['POST'])
def create_film_char():
    # 获取输入参数
    data = request.get_json()
    # 解析 JSON 数据
    file_path = data.get('file_path')
    language = data.get('language')
    req_lora_sd = data.get('lora_sd', [])
    req_lora_config = data.get('lora_config', [])

    # 验证输入参数
    if not file_path or not language:
        return jsonify({"error": "Missing required parameters."}), 400

    try:
        if language.lower() == "english":
            nlp = nlp_en
            gender = Genderize()
        elif language.lower() == "chinese":
            nlp = nlp_zh
            gender = Gender()
        else:
            return jsonify({"error": "Unsupported language."}), 400

        # 存放整理后的文本行
        cleaned_lines = []
        add_line = ''
        CONTROL_CHARS = ''.join(map(chr, range(0, 32)))
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()  # 去掉行首尾空白字符
                if language.lower() == "english":
                    line = ' '.join(
                        ''.join(char for char in word if char not in CONTROL_CHARS)
                        for word in line.split()
                    )
                elif language.lower() == "chinese":
                    line = ''.join(char for char in line if char not in string.whitespace and char not in CONTROL_CHARS)
                if not line:
                    continue
                # 判断最后一个字符是否为标点符号
                doc = nlp(line)
                is_last_token_punctuation = doc[-1].is_punct if doc else False

                if is_last_token_punctuation:
                    if add_line:
                        cleaned_lines.append(add_line + line)
                    else:
                        cleaned_lines.append(line)
                    add_line = ''
                else:
                    add_line = add_line + line

        characters = []
        total_name_counts = Counter()
        name_to_pinyin = {}
        for line in cleaned_lines:
            doc = nlp(line)
            names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            for name in names:
                if language.lower() == "english":
                    name_pinyin = name
                elif language.lower() == "chinese":
                    name_pinyin = ''.join([item[0] for item in pinyin(name, style=Style.NORMAL)])
                name_to_pinyin[name] = name_pinyin

        for name in name_to_pinyin:
            count = sum(line.count(name) for line in cleaned_lines)
            total_name_counts[name] = count

        for name, times in total_name_counts.items():
            name_pinyin = name_to_pinyin[name]
            lora_name = ''
            lora_alias = ''
            lora_alprompt = ''
            lora_trigger = ''
            lora_prefix = ''
            if language.lower() == "english":
                gender_probabilities = gender.get(name)
                if gender_probabilities:
                    if gender_probabilities[0]['gender'] == 'female':
                        gender_label = 'female'
                    elif gender_probabilities[0]['gender'] == 'male':
                        gender_label = 'male'
                    else:
                        gender_label = 'unknown'
                else:
                    gender_label = 'unknown'
            elif language.lower() == "chinese":
                gender_probabilities = gender.predict(name)[1]

                if gender_probabilities['M'] > gender_probabilities['F']:
                    gender_label = 'male'
                elif gender_probabilities['F'] > gender_probabilities['M']:
                    gender_label = 'female'
                else:
                    gender_label = 'unknown'

            for lora_config in req_lora_config:
                if lora_config.get('LoraGender') == gender_label:
                    if req_lora_sd is None or not req_lora_sd:
                        break
                    else:
                        lora_alias = lora_config.get('LoraAlias')
                        lora_alprompt = lora_config.get('LoraAlPrompt')
                        lora_trigger = lora_config.get('LoraTrigger')
                        lora_prefix = lora_config.get('LoraPrefix')
                        for lora_sd in req_lora_sd:
                            if lora_sd.get('alias') == lora_alias:
                                lora_name = lora_sd.get('name')
                                break
            characters.append({
                "char_name": name,
                "char_nm_pinyin": name_pinyin,
                "char_times": times,
                "char_gender": gender_label,
                "lora_name": lora_name,
                "lora_alias": lora_alias,
                "lora_alprompt": lora_alprompt,
                "lora_trigger": lora_trigger,
                "lora_prefix": lora_prefix,
            })
        return jsonify(characters)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/createfilmitem', methods=['POST'])
def create_film_item():
    # 获取输入参数
    data = request.get_json()
    # 解析 JSON 数据
    file_path = data.get('file_path')
    language = data.get('language')
    translation_module = data.get('translation_module')
    appid = data.get('appid')  # Tencent 翻译 API APPID
    secret_key = data.get('secret_key')  # Tencent 翻译 API 密钥
    create_characters = data.get('create_characters')
    req_characters = data.get('characters', [])
    req_lora_sd = data.get('lora_sd', [])
    req_lora_config = data.get('lora_config', [])

    # 验证输入参数
    if not file_path or not language or not translation_module:
        return jsonify({"error": "Missing required parameters."}), 400

    if (translation_module == "tencent" or translation_module == "baidu") and (not appid or not secret_key):
        return jsonify({"error": "Missing required parameters."}), 400

    try:
        if language.lower() == "english":
            nlp = nlp_en
            source_lang = "en"
            target_lang = "zh"  # 翻译目标语言为中文
        elif language.lower() == "chinese":
            nlp = nlp_zh
            gender = Gender()
            source_lang = "zh"
            target_lang = "en"  # 翻译目标语言为英文
        else:
            return jsonify({"error": "Unsupported language."}), 400

        # 存放整理后的文本行
        cleaned_lines = []
        add_line = ''
        CONTROL_CHARS = ''.join(map(chr, range(0, 32)))
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()  # 去掉行首尾空白字符
                if language.lower() == "english":
                    line = ' '.join(
                        ''.join(char for char in word if char not in CONTROL_CHARS)
                        for word in line.split()
                    )
                elif language.lower() == "chinese":
                    line = ''.join(char for char in line if char not in string.whitespace and char not in CONTROL_CHARS)
                if not line:
                    continue
                # 判断最后一个字符是否为标点符号
                doc = nlp(line)
                is_last_token_punctuation = doc[-1].is_punct if doc else False

                if is_last_token_punctuation:
                    if add_line:
                        cleaned_lines.append(add_line + line)
                    else:
                        cleaned_lines.append(line)
                    add_line = ''
                else:
                    add_line = add_line + line

        # 使用 spaCy 将整理后的文本按句子分解，得到句子数组
        sentence_original_array = []
        sentence_audio_array = []
        sentence_array = []
        modified_sentences = []
        characters = []
        total_name_counts = Counter()
        name_to_pinyin = {}
        if create_characters:
            for line in cleaned_lines:
                doc = nlp(line)
                names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
                for name in names:
                    if language.lower() == "english":
                        name_pinyin = name
                    elif language.lower() == "chinese":
                        name_pinyin = ''.join([item[0] for item in pinyin(name, style=Style.NORMAL)])
                    name_to_pinyin[name] = name_pinyin
            for name in name_to_pinyin:
                count = sum(line.count(name) for line in cleaned_lines)
                total_name_counts[name] = count
        else:
            for req_character in req_characters:
                total_name_counts[req_character.get('char_name')] += 1
                if not req_character.get('char_nm_pinyin'):
                    req_character['char_nm_pinyin'] = ''.join([item[0] for item in pinyin(req_character.get('char_name'), style=Style.NORMAL)])
                name_to_pinyin[req_character.get('char_name')] = req_character.get('char_nm_pinyin')
                if req_character.get('char_times') == 0:
                    for name in name_to_pinyin:
                        count = sum(line.count(name) for line in cleaned_lines)
                        req_character['char_times'] = count

        doc = nlp(''.join(cleaned_lines))
        sentences = [sent.text for sent in doc.sents if sent.text.strip()]
        if language.lower() == "english":
            sentence_original_array.extend(sentences)
        elif language.lower() == "chinese":
            processed_clauses = process_clauses(sentences)
            i = 0
            while i < len(processed_clauses):
                current_sentence = processed_clauses[i]
                if i + 1 < len(processed_clauses) and len(current_sentence + processed_clauses[i + 1]) <= SENTENCE_LIMIT:
                    sentence_original_array.append(current_sentence + processed_clauses[i + 1])
                    i += 2
                else:
                    sentence_original_array.append(current_sentence)
                    i += 1

        for sentence in sentence_original_array:
            words = nlp(sentence)
            modified_sentence = ""
            for token in words:
                if "地" in token.text and token.pos_ != "NOUN":
                    modified_sentence += "的"
                else:
                    modified_sentence += token.text
            sentence_audio_array.append(modified_sentence)
            for name, times in total_name_counts.items():
                if name in sentence:
                    name_pinyin = name_to_pinyin[name]
                    sentence = sentence.replace(name, f'[{name_pinyin}]')
            modified_sentences.append(sentence)
        sentence_array.extend(modified_sentences)

        translated_sentences = [
            translate_text(sentence, translation_module, source_lang, target_lang, appid, secret_key) for sentence in
            sentence_array]

        if language.lower() == "english":
            keywords = [extract_description(sentence) for sentence in sentence_array]
        elif language.lower() == "chinese":
            keywords = [extract_description(translated_sentence) for translated_sentence in translated_sentences]
            if create_characters:
                for name, times in total_name_counts.items():
                    name_pinyin = name_to_pinyin[name]
                    lora_name = ''
                    lora_alias = ''
                    lora_alprompt = ''
                    lora_trigger = ''
                    lora_prefix = ''
                    gender_probabilities = gender.predict(name)[1]

                    if gender_probabilities['M'] > gender_probabilities['F']:
                        gender_label = 'male'
                    elif gender_probabilities['F'] > gender_probabilities['M']:
                        gender_label = 'female'
                    else:
                        gender_label = 'unknown'
                    for lora_config in req_lora_config:
                        if lora_config.get('LoraGender') == gender_label:
                            lora_alias = lora_config.get('LoraAlias')
                            lora_alprompt = lora_config.get('LoraAlPrompt')
                            lora_trigger = lora_config.get('LoraTrigger')
                            lora_prefix = lora_config.get('LoraPrefix')
                            for lora_sd in req_lora_sd:
                                if lora_sd.get('alias') == lora_alias:
                                    lora_name = lora_sd.get('name')
                                    break

                    characters.append({
                        "char_name": name,
                        "char_nm_pinyin": name_pinyin,
                        "char_times": times,
                        "char_gender": gender_label,
                        "lora_name": lora_name,
                        "lora_alias": lora_alias,
                        "lora_alprompt": lora_alprompt,
                        "lora_trigger": lora_trigger,
                        "lora_prefix": lora_prefix,
                    })
            else:
                characters.extend(req_characters)

        # 根据语言参数调整返回的 JSON 结构
        if language.lower() == "english":
            # 英文句子设为 TranslatedText，翻译后的中文设为 OriginalText
            response_data = {
                "original_text": translated_sentences,
                "translated_text": sentence_original_array,
                "audio_text": sentence_audio_array,
                "pinyin_text": sentence_array,
                "prompt_key": keywords,
                "characters": characters
            }
        elif language.lower() == "chinese":
            # 中文句子设为 OriginalText，翻译后的英文设为 TranslatedText
            response_data = {
                "original_text": sentence_original_array,
                "translated_text": translated_sentences,
                "audio_text": sentence_audio_array,
                "pinyin_text": sentence_array,
                "prompt_key": keywords,
                "characters": characters
            }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API:create_film_audio
class TorchSeedContext:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = torch.random.get_rng_state()
        torch.manual_seed(self.seed)

    def __exit__(self, type, value, traceback):
        torch.random.set_rng_state(self.state)

def replace_arabic_with_chinese(text: str) -> str:
    arabic_to_chinese = {
        '0': '零',
        '1': '一',
        '2': '二',
        '3': '三',
        '4': '四',
        '5': '五',
        '6': '六',
        '7': '七',
        '8': '八',
        '9': '九'
    }
    chinese_text = ''.join(arabic_to_chinese[char] if char in arabic_to_chinese else char for char in text)
    return chinese_text

def calculate_audio_duration(wav_data: np.ndarray, sample_rate: int) -> float:
    duration = len(wav_data) / sample_rate
    return duration

@app.route('/createfilmaudio', methods=['POST'])
def create_film_audio():
    data = request.get_json()
    # 解析 JSON 数据
    text = data.get('text')
    file_path = data.get('file_path')
    seed = data.get('seed')
    temperature = data.get('temperature')
    top_P = data.get('top_P')
    top_K = data.get('top_K')

    try:
        chat = ChatTTS.Chat()
        chat.load()

        with TorchSeedContext(seed):
            rand_spk = chat.sample_random_speaker()
        # speaker_vector = '-4.741,0.419,-3.355,3.652,-1.682,-1.254,9.719,1.436,0.871,12.334,-0.175,-2.653,-3.132,0.525,1.573,-0.351,0.030,-3.154,0.935,-0.111,-6.306,-1.840,-0.818,9.773,-1.842,-3.433,-6.200,-4.311,1.162,1.023,11.552,2.769,-2.408,-1.494,-1.143,12.412,0.832,-1.203,5.425,-1.481,0.737,-1.487,6.381,5.821,0.599,6.186,5.379,-2.141,0.697,5.005,-4.944,0.840,-4.974,0.531,-0.679,2.237,4.360,0.438,2.029,1.647,-2.247,-1.716,6.338,1.922,0.731,-2.077,0.707,4.959,-1.969,5.641,2.392,-0.953,0.574,1.061,-9.335,0.658,-0.466,4.813,1.383,-0.907,5.417,-7.383,-3.272,-1.727,2.056,1.996,2.313,-0.492,3.373,0.844,-8.175,-0.558,0.735,-0.921,8.387,-7.800,0.775,1.629,-6.029,0.709,-2.767,-0.534,2.035,2.396,2.278,2.584,3.040,-6.845,7.649,-2.812,-1.958,8.794,2.551,3.977,0.076,-2.073,-4.160,0.806,3.798,-1.968,-4.690,5.702,-4.376,-2.396,1.368,-0.707,4.930,6.926,1.655,4.423,-1.482,-3.670,2.988,-3.296,0.767,3.306,1.623,-3.604,-2.182,-1.480,-2.661,-1.515,-2.546,3.455,-3.500,-3.163,-1.376,-12.772,1.931,4.422,6.434,-0.386,-0.704,-2.720,2.177,-0.666,12.417,4.228,0.823,-1.740,1.285,-2.173,-4.285,-6.220,2.479,3.135,-2.790,1.395,0.946,-0.052,9.148,-2.802,-5.604,-1.884,1.796,-0.391,-1.499,0.661,-2.691,0.680,0.848,3.765,0.092,7.978,3.023,2.450,-15.073,5.077,3.269,2.715,-0.862,2.187,13.048,-7.028,-1.602,-6.784,-3.143,-1.703,1.001,-2.883,0.818,-4.012,4.455,-1.545,-14.483,-1.008,-3.995,2.366,3.961,1.254,-0.458,-1.175,2.027,1.830,2.682,0.131,-1.839,-28.123,-1.482,16.475,2.328,-13.377,-0.980,9.557,0.870,-3.266,-3.214,3.577,2.059,1.676,-0.621,-6.370,-2.842,0.054,-0.059,-3.179,3.182,3.411,4.419,-1.688,-0.663,-5.189,-5.542,-1.146,2.676,2.224,-5.519,6.069,24.349,2.509,4.799,0.024,-2.849,-1.192,-16.989,1.845,6.337,-1.936,-0.585,1.691,-3.564,0.931,0.223,4.314,-2.609,0.544,-1.931,3.604,1.248,-0.852,2.991,-1.499,-3.836,1.774,-0.744,0.824,7.597,-1.538,-0.009,0.494,-2.253,-1.293,-0.475,-3.816,8.165,0.285,-3.348,3.599,-4.959,-1.498,-1.492,-0.867,0.421,-2.191,-1.627,6.027,3.667,-21.459,2.594,-2.997,5.076,0.197,-3.305,3.998,1.642,-6.221,3.177,-3.344,5.457,0.671,-2.765,-0.447,1.080,2.504,1.809,1.144,2.752,0.081,-3.700,0.215,-2.199,3.647,1.977,1.326,3.086,34.789,-1.017,-14.257,-3.121,-0.568,-0.316,11.455,0.625,-6.517,-0.244,-8.490,9.220,0.068,-2.253,-1.485,3.372,2.002,-3.357,3.394,1.879,16.467,-2.271,1.377,-0.611,-5.875,1.004,12.487,2.204,0.115,-4.908,-6.992,-1.821,0.211,0.540,1.239,-2.488,-0.411,2.132,2.130,0.984,-10.669,-7.456,0.624,-0.357,7.948,2.150,-2.052,3.772,-4.367,-11.910,-2.094,3.987,-1.565,0.618,1.152,1.308,-0.807,1.212,-4.476,0.024,-6.449,-0.236,5.085,1.265,-0.586,-2.313,3.642,-0.766,3.626,6.524,-1.686,-2.524,-0.985,-6.501,-2.558,0.487,-0.662,-1.734,0.275,-9.230,-3.785,3.031,1.264,15.340,2.094,1.997,0.408,9.130,0.578,-2.239,-1.493,11.034,2.201,6.757,3.432,-4.133,-3.668,2.099,-6.798,-0.102,2.348,6.910,17.910,-0.779,4.389,1.432,-0.649,5.115,-1.064,3.580,4.129,-4.289,-2.387,-0.327,-1.975,-0.892,5.327,-3.908,3.639,-8.247,-1.876,-10.866,2.139,-3.932,-0.031,-1.444,0.567,-5.543,-2.906,1.399,-0.107,-3.044,-4.660,-1.235,-1.011,9.577,2.294,6.615,-1.279,-2.159,-3.050,-6.493,-7.282,-8.546,5.393,2.050,10.068,3.494,8.810,2.820,3.063,0.603,1.965,2.896,-3.049,7.106,-0.224,-1.016,2.531,-0.902,1.436,-1.843,1.129,6.746,-2.184,0.801,-0.965,-7.555,-18.409,6.176,-3.706,2.261,4.158,-0.928,2.164,-3.248,-4.892,-0.008,-0.521,7.931,-10.693,4.320,-0.841,4.446,-1.591,-0.702,4.075,3.323,-3.406,-1.198,-5.518,-0.036,-2.247,-2.638,2.160,-9.644,-3.858,2.402,-2.640,1.683,-0.961,-3.076,0.226,5.106,0.712,0.669,2.539,-4.340,-0.892,0.732,0.775,-2.757,4.365,-2.368,5.368,0.342,-0.655,0.240,0.775,3.686,-4.008,16.296,4.973,1.851,4.747,0.652,-2.117,6.470,2.189,-8.467,3.236,3.745,-1.332,3.583,-2.504,5.596,-2.440,0.995,-2.267,-3.322,3.490,1.156,1.716,0.669,-3.640,-1.709,5.055,6.265,-3.963,2.863,14.129,5.180,-3.590,0.393,0.234,-3.978,6.946,-0.521,1.925,-1.497,-0.283,0.895,-3.969,5.338,-1.808,-3.578,2.699,2.728,-0.895,-2.175,-2.717,2.574,4.571,1.131,2.187,3.620,-0.388,-3.685,0.979,2.731,-2.164,1.628,-1.006,-7.766,-11.033,-10.985,-2.413,-1.967,0.790,0.826,-1.623,-1.783,3.021,1.598,-0.931,-0.605,-1.684,1.408,-2.771,-2.354,5.564,-2.296,-4.774,-2.830,-5.149,2.731,-3.314,-1.002,3.522,3.235,-1.598,1.923,-2.755,-3.900,-3.519,-1.673,-2.049,-10.404,6.773,1.071,0.247,1.120,-0.794,2.187,-0.189,-5.591,4.361,1.772,1.067,1.895,-5.649,0.946,-2.834,-0.082,3.295,-7.659,-0.128,2.077,-1.638,0.301,-0.974,4.331,11.711,4.199,1.545,-3.236,-4.404,-1.333,0.623,1.414,-0.240,-0.816,-0.808,-1.382,0.632,-5.238,0.120,10.634,-2.026,1.702,-0.469,1.252,1.173,3.015,-8.798,1.633,-5.323,2.149,-6.481,11.635,3.072,5.642,5.252,4.702,-3.523,-0.594,4.150,1.392,0.554,-4.377,3.646,-0.884,1.468,0.779,2.372,-0.101,-5.702,0.539,-0.440,5.149,-0.011,-1.899,-1.349,-0.355,0.076,-0.100,-0.004,5.346,6.276,0.966,-3.138,-2.633,-3.124,3.606,-3.793,-3.332,2.359,-0.739,-3.301,-2.775,-0.491,3.283,-1.394,-1.883,1.203,1.097,2.233,2.170,-2.980,-15.800,-6.791,-0.175,-4.600,-3.840,-4.179,6.568,5.935,-0.431,4.623,4.601,-1.726,0.410,2.591,4.016,8.169,1.763,-3.058,-1.340,6.276,4.682,-0.089,1.301,-4.817'
        # speaker = torch.tensor([float(x) for x in speaker_vector.split(',')])

        params_infer_code = ChatTTS.Chat.InferCodeParams(
            prompt="[speed_6]",
            spk_emb=rand_spk,
            temperature=temperature,
            top_P=top_P,
            top_K=top_K,
            manual_seed=seed,
        )

        wavs = chat.infer(replace_arabic_with_chinese(text), skip_refine_text=True, params_infer_code=params_infer_code, )

        for i in range(len(wavs)):
            if i == 0:
                wavfile.write(file_path, 24000, wavs[i])
                duration = calculate_audio_duration(wavs[i], 24000)

        response_data = {
            "file_path": file_path,
            "duration": duration
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/createfilmdraft', methods=['POST'])
def create_film_draft():
    data = request.get_json()
    pro_name = data.get('pro_name')
    draft_path = data.get('draft_path')
    data_path = data.get('data_path')
    width = data.get('width')
    height = data.get('height')
    fixed_width = width * 0.53

    dump_path  = os.path.join(os.path.dirname(data_path), 'Draft')
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)


    with open(data_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    extra_material = {
        "fixed_width": fixed_width,
    }

    start_audio_time = 0
    # 创建剪映草稿
    script = draft.Script_file(width, height)
    script.add_meta_info(os.path.join(draft_path, pro_name))
    # 添加音频、视频和文本轨道
    script.add_track(draft.Track_type.audio).add_track(draft.Track_type.video).add_track(draft.Track_type.text)
    # 遍历每个元素
    for item in json_data:
        original_text = item.get('OriginalText')
        audio_path = item.get('AudioPath')
        audio_time = item.get('AudioTime')

        image_path = ''
        for img_group in item.get('ImageGroup', []):
            if img_group.get('IsCheckedImg'):
                image_path = img_group.get('ImagePath')
                break

        audio_material = draft.Audio_material(audio_path)
        sticker_material = draft.Video_material(image_path)
        script.add_material(audio_material).add_material(sticker_material)
        audio_segment = draft.Audio_segment(audio_material, trange(start_audio_time, audio_material.duration))
        sticker_segment = draft.Video_segment(sticker_material, trange(start_audio_time, audio_material.duration))
        script.add_segment(audio_segment).add_segment(sticker_segment)
        text_segment = draft.Text_segment(original_text, trange(start_audio_time, audio_material.duration),
                                          style=draft.Text_style(color=(1.0, 1.0, 0.0)),
                                          clip_settings=draft.Clip_settings(transform_y=-0.8), extra_material_val=extra_material)
        script.add_segment(text_segment)
        start_audio_time += audio_material.duration

    script.dump(dump_path)
    response_data = {
        "file_path": dump_path
    }
    return jsonify(response_data)

# 运行 Flask 应用程序
if __name__ == '__main__':
    app.run(debug=False, port=7855)

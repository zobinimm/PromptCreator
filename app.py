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
                    if req_lora_sd is None or not req_lora_sd:
                        break
                    else:
                        lora_alias = lora_config.get('LoraAlias')
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

        for line in cleaned_lines:
            doc = nlp(line)
            sentences = [sent.text.strip() for sent in doc.sents]
            for sentence_org in sentences:
                if language.lower() == "english":
                    sentence_original_array.append(sentence_org)
                elif language.lower() == "chinese":
                    for fix_sentence in fix_sentence_len(sentence_org):
                        sentence_original_array.append(fix_sentence)

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

        params_infer_code = ChatTTS.Chat.InferCodeParams(
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
    fixed_width = width * 0.6

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

# -*- coding: utf-8 -*-
import hashlib
import hmac
import json
import time
from datetime import datetime
import requests

def sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()
text = "她来了，出去抓溪水里的垃圾，然后……被烧伤了。"
source_lang = "zh"
target_lang = "en"

secret_id = ""
secret_key = ""

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
    response = requests.post(endpoint, headers=headers, data=payload.encode("utf-8"))
    print(response.text)
    # 将响应解析为 JSON
    response_json = response.json()
    # 从 Response 对象中提取 TargetText 字段
    target_text = response_json.get('Response', {}).get('TargetText', '字段未找到')
    print("TargetText:", target_text)

except requests.RequestException as err:
    print(f"Error: {err}")

import os
import random
import requests
import hashlib
from ...tool import Tool


def build_tool(config) -> Tool:
    tool = Tool(
        "Translator Info",
        "Translate a given text from one language to another.",
        name_for_model="Translator",
        description_for_model="Plugin for translating text from one language to another.",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="shihaoliang0828@gmail.com",
        legal_info_url="hello@legal.com",
    )
    subscription_key = os.getenv("BAIDU_TRANSLATE_KEY", None)
    if subscription_key is None:
        raise Exception("BAIDU_TRANSLATE_KEY is not set")
    secret_key = os.getenv("BAIDU_SECRET_KEY", None)
    if secret_key is None:
        raise Exception("BAIDU_SECRET_KEY is not set")
    endpoint = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    fromLang = "auto"
    salt = random.randint(32768, 65536)
    header = {"Content-Type": "application/x-www-form-urlencoded"}

    @tool.get("/get_translation")
    def get_translation(text: str, tgt_lang: str) -> str:
        sign = subscription_key + text + str(salt) + secret_key
        md = hashlib.md5()
        md.update(sign.encode(encoding="utf-8"))
        sign = md.hexdigest()
        data = {
            "appid": subscription_key,
            "q": text,
            "from": fromLang,
            "to": tgt_lang,
            "salt": salt,
            "sign": sign,
        }
        response = requests.post(endpoint, params=data, headers=header)
        text = response.json()
        results = text["trans_result"][0]["dst"]
        return results

    return tool

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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

    BASE_MODEL = (
        config["model"] if "model" in config else "facebook/nllb-200-distilled-600M"
    )
    SRC_LANG = config["src_lang"] if "src_lang" in config else "eng_Latn"
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, use_auth_token=True, src_lang=SRC_LANG
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, use_auth_token=True)

    @tool.get("/get_translation")
    def get_translation(
        input_text: str or list, tgt_lang: str, max_length: int
    ) -> str or list:
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=max_length
        )
        if isinstance(input_text, str):
            translations = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]
        elif isinstance(input_text, list):
            translations = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )
        return translations

    return tool

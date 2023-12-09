from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import py3langid as langid
from iso639 import languages

from typing import Dict
from copy import deepcopy
import os


def detect_lang(text: str):
    lang_code = langid.classify(text)[0]
    lang_name = languages.get(part1=lang_code[:2]).name
    return lang_name


class Translator:
    def __init__(self, openai_api_key: str = None, model_name: str = "gpt-3.5-turbo"):
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.init_flag = False

    def init_model(self):
        llm = self.create_openai_model(self.openai_api_key, self.model_name)
        prompt = self.create_prompt()
        self.chain = LLMChain(llm=llm, prompt=prompt)
        self.init_flag = True

    def __call__(self, inputs: Dict[str, str]) -> Dict[str, str]:
        if not self.init_flag:
            self.init_model()

        question = inputs["input"]
        answer = inputs["output"]

        src_lang = detect_lang(answer)
        tgt_lang = detect_lang(question)

        if src_lang != tgt_lang:
            translated_answer = self.chain.run(text=answer, language=tgt_lang)
            outputs = deepcopy(inputs)
            outputs["output"] = translated_answer
            return outputs
        else:
            return inputs

    def create_openai_model(self, openai_api_key: str, model_name: str) -> OpenAI:
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        llm = OpenAI(
            model_name=model_name, temperature=0.0, openai_api_key=openai_api_key
        )
        return llm

    def create_prompt(self) -> PromptTemplate:
        template = """
        Translate to {language}: {text} => 
        """
        prompt = PromptTemplate(input_variables=["text", "language"], template=template)
        return prompt


if __name__ == "__main__":
    lang = {
        "zh": {
            "question": "帮我介绍下《深海》这部电影",
            "answer": "《深海》是一部中国大陆的动画、奇幻电影，由田晓鹏导演，苏鑫、王亭文、滕奎兴等人主演。剧情简介是在大海的最深处，藏着所有秘密。一位现代少女（参宿）误入梦幻的 深海世界，却因此邂逅了一段独特的生命旅程。![img](https://img3.doubanio.com/view/photo/s_ratio_poster/public/p2635450820.webp)",
        },
        "ja": {
            "question": "映画「深海」について教えてください",
            "answer": "「深海」は、中国本土のアニメーションおよびファンタジー映画で、Tian Xiaopeng が監督し、Su Xin、Wang Tingwen、Teng Kuixing などが出演しています。 あらすじは、海の最深部にはすべての秘密が隠されているというもの。 夢のような深海の世界に迷い込んだ現代少女（さんすけ）は、それをきっかけに独特の人生の旅に出くわす。 ![img](https://img3.doubanio.com/view/photo/s_ratio_poster/public/p2635450820.webp)",
        },
        "ko": {
            "question": "영화 딥씨에 대해 알려주세요",
            "answer": '"Deep Sea"는 Tian Xiaopeng 감독, Su Xin, Wang Tingwen, Teng Kuixing 등이 출연한 중국 본토의 애니메이션 및 판타지 영화입니다. 시놉시스는 바다 가장 깊은 곳에 모든 비밀이 숨겨져 있다는 것입니다. 현대 소녀(산스케)는 꿈 같은 심해 세계로 방황하지만 그것 때문에 독특한 삶의 여정을 만난다. ![img](https://img3.doubanio.com/view/photo/s_ratio_poster/public/p2635450820.webp)',
        },
        "en": {
            "question": "Tell me about the movie '深海'",
            "answer": '"Deep Sea" is an animation and fantasy film in mainland China, directed by Tian Xiaopeng, starring Su Xin, Wang Tingwen, Teng Kuixing and others. The synopsis is that in the deepest part of the sea, all secrets are hidden. A modern girl (Sansuke) strays into the dreamy deep sea world, but encounters a unique journey of life because of it. ![img](https://img3.doubanio.com/view/photo/s_ratio_poster/public/p2635450820.webp)',
        },
        "de": {
            "question": "Erzähl mir von dem Film '深海'",
            "answer": '"Deep Sea" ist ein Animations- und Fantasyfilm in Festlandchina unter der Regie von Tian Xiaopeng mit Su Xin, Wang Tingwen, Teng Kuixing und anderen in den Hauptrollen. Die Zusammenfassung ist, dass im tiefsten Teil des Meeres alle Geheimnisse verborgen sind. Ein modernes Mädchen (Sansuke) verirrt sich in die verträumte Tiefseewelt, trifft dabei aber auf eine einzigartige Lebensreise. ![img](https://img3.doubanio.com/view/photo/s_ratio_poster/public/p2635450820.webp)',
        },
        "fr": {
            "question": "Parlez-moi du film 'Deep Sea'",
            "answer": "\"Deep Sea\" est un film d'animation et fantastique en Chine continentale, réalisé par Tian Xiaopeng, avec Su Xin, Wang Tingwen, Teng Kuixing et d'autres. Le synopsis est que dans la partie la plus profonde de la mer, tous les secrets sont cachés. Une fille moderne (Sansuke) s'égare dans le monde onirique des profondeurs marines, mais rencontre un voyage de vie unique à cause de cela. ![img](https://img3.doubanio.com/view/photo/s_ratio_poster/public/p2635450820.webp)",
        },
        "ru": {
            "question": "Расскажите о фильме 'Глубокое море'",
            "answer": "«Глубокое море» — это анимационный и фэнтезийный фильм в материковом Китае, снятый Тянь Сяопином, в главных ролях Су Синь, Ван Тинвэнь, Тэн Куйсин и другие. Суть в том, что в самой глубокой части моря скрыты все секреты. Современная девушка (Сансукэ) заблудилась в мечтательном глубоководном мире, но из-за этого столкнулась с уникальным жизненным путешествием. ![img](https://img3.doubanio.com/view/photo/s_ratio_poster/public/p2635450820.webp)",
        },
    }

    translator = Translator()
    for source in lang:
        for target in lang:
            print(source, "=>", target, end=":\t")
            question = lang[target]["question"]
            answer = lang[source]["answer"]
            inputs = {"input": question, "output": answer}

            result = translator(inputs)
            translated_answer = result["output"]

            if (
                detect_lang(question)
                == detect_lang(translated_answer)
                == languages.get(part1=target).name
            ):
                print("Y")
            else:
                print("N")
                print("====================")
                print("Question:\t", detect_lang(question), " - ", question)
                print("Answer:\t", detect_lang(answer), " - ", answer)
                print(
                    "Translated Anser:\t",
                    detect_lang(translated_answer),
                    " - ",
                    translated_answer,
                )
                print("====================")

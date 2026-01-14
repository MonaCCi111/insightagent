import json
from langchain_gigachat import GigaChat
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline


class InsightAnalyzer:
    def __init__(self, giga_credentials):
        self.llm = GigaChat(
            credentials=giga_credentials,
            model='GigaChat-2',
            verify_ssl_certs=False,
            temperature=0
        )

        self.sentiment_analyzer = pipeline(
            'sentiment-analysis',
            model='blacnhefort/rubert-base-cased-sentiment',
            device=-1
        )
        self._init_chains()

    def _init_chains(self):
        theme_template = \
            '''Ты — опытный аналитик обратной связи. Определи основную тему отзыва клиента.
            Тема должна быть одной из следующих категорий: [Качество товара, Доставка, Служба поддержки, Цена, Общее впечатление].
            
            Отзыв: {review_text}
            
            Верни ТОЛЬКО название категории, без кавычек, точек и дополнительного текста.'''

        theme_prompt = PromptTemplate(template=theme_template, input_variables=["review-text"])
        self.theme_chain = LLMChain(llm=self.llm, prompt=theme_prompt)

        insight_template = \
            '''На основе отзыва клиента сгенерируй структурированный анализ
            Отзыв: {review_text}
            Тональность: {sentiment}
            Тема: {theme}
            
            Проанализируй и верни ответ в формату JSON:
            {{
                "key_problem_or_strength": "Основная проблема или преимущество (одно предложение)",
                "root_cause": "Возможная причина проблемы (если есть)",
                "actionable_recommendation": "Конкретная рекомендация для бизнеса",
                "priority": "Приоритет (high/medium/low)"
            }}
            Только JSON, никакого другого текста. Старайся быть довольно кратким.'''

        insight_prompt = PromptTemplate(
            template=insight_template,
            input_variables=['review_text', 'sentiment', 'theme']
        )
        self.insight_chain = LLMChain(llm=self.llm, prompt=insight_prompt)


    def analyze_review(self, text):
        sentiment_result = self.sentiment_analyzer(text)[0]
        sentiment_map = {'NEGATIVE': 'negative', 'NEUTRAL': 'neutral', 'POSITIVE': 'positive'}
        sentiment = sentiment_map[sentiment_result['label']]
        theme = self.theme_chain.run(review_text=text).strip()
        insight_output = self.insight_chain.run(
            review_text=text,
            sentiment=sentiment,
            theme=theme
        )

        try:
            insights = json.loads(insight_output.strip())
        except json.JSONDecodeError:
            insights = {'error': 'Не удалось спарсить инсайты'}

        return {
            'text': text,
            'sentiment': sentiment,
            'theme': theme,
            'insights': insights
        }
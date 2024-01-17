
"""
Given the intents and (correct) entities found in the user's messages,
decides what to do/answer. This is a goal-based dialog manager.
Relies on a configuration of goals and on a stack of conversation contexts.
"""
import abc
from collections.abc import Sequence

# 1.The message is sent to the system via rest into Dialog Policy (DP). Сообщение попадает в чат-бот
# 2.DP send request CLASSIFY_TEXT to Intent Recognizer (IR) to classify message. Отправляем сообщение на классификацию
# 3.IR send back to the DP result of classification CLASSIFICATION_RESULT - intent, projects. Классифицируем намерения пользователя и домен.
# 4.DP send a request MESSAGE_TO_SKILL to each Smart App(SA) classified. Выбираем скил для ответа в зависимости от вопроса.
# 5.Each SA response ANSWER_TO_USER to DP. Возвращаем ответ с контекстом, првоеряем ответ классифицируем смог ли ответить

# ??6.When DP got all responces from SAs, DP sends to IR BLENDER_REQUEST with all ANSWER_TO_USERs included
# ???7.IR respond BLENDER_RESPONSE to DP
# 8.DP respond ANSWER_TO_USER via rest Возвращаем ответ или доп. вопрос или другйо исход.


#!/usr/bin/env python3

import requests
import time
import argparse
import os
import json

from requests.compat import urljoin


class BotHandler(object):
    """
        BotHandler is a class which implements all back-end of the bot.
        It has tree main functions:
            'get_updates' — checks for new messages
            'send_message' – posts new message to user
            'get_answer' — computes the most relevant on a user's question
    """

    def __init__(self, token, dialogue_manager):
        self.token = token
        self.api_url = "https://api.telegram.org/bot{}/".format(token)
        self.dialogue_manager = dialogue_manager

    def get_updates(self, offset=None, timeout=30):
        params = {"timeout": timeout, "offset": offset}
        raw_resp = requests.get(urljoin(self.api_url, "getUpdates"), params)
        try:
            resp = raw_resp.json()
        except json.decoder.JSONDecodeError as e:
            print("Failed to parse response {}: {}.".format(raw_resp.content, e))
            return []

        if "result" not in resp:
            return []
        return resp["result"]

    def send_message(self, chat_id, text):
        params = {"chat_id": chat_id, "text": text}
        return requests.post(urljoin(self.api_url, "sendMessage"), params)

    def get_answer(self, question):
        if question == '/start':
            return "Hi, I am your project bot. How can I help you today?"
        return self.dialogue_manager.generate_answer(question)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default='')
    return parser.parse_args()


def is_unicode(text):
    return len(text) == len(text.encode())

class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.

        #### YOUR CODE HERE ####
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim).reshape(1 ,-1)
        best_thread = pairwise_distances_argmin(question_vec ,thread_embeddings)

        return thread_ids[best_thread]

class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_classification = intent_classification(query)

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.action = ActionUtterance()


        # Chit-chat bot - ответ не связанный с qa knowledge based
        self.create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param


        self.chitchat_bot = ChatBot('Ron Obvious')

        # Create a new trainer for the chatbot
        trainer = ChatterBotCorpusTrainer(self.chitchat_bot)

        # Train the chatbot based on the english corpus
        trainer.train("chatterbot.corpus.english")

    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.

        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)

        # Chit-chat part:
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.
            response = self.chitchat_bot.get_response(question)
            return response

        # Goal-oriented part:
        else:
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0]

            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(question, tag)[0]

            return self.ANSWER_TEMPLATE % (tag, thread_id)


class Dialogue(abc.ABC):
    """
    Объект диалога принимает высказывания, выполняет их парсинг
    и интерпретацию, а затем изменяет внутреннее состояние. После этого
    он может сформулировать ответ.
    """

    def listen(self, text, response=True, **kwargs):
        """
        Принимает текст text высказывания и выполняет его парсинг. Передает
        результат методу interpret для определения ответа. Если ответ
        необходим, вызывается метод respond, генерирующий текст ответа
        на основе последнего поступившего текста и текущего состояния
        объекта Dialog.
        :param text:
        :param response:
        :param kwargs:
        :return:
        """

        # парсинг входлного текста
        sents = self.parse(text)

        # интерпретация
        sents, confidence, intent, kwarg = self.interpret(sents, **kwargs)

        # определение ответа
        if response:
            reply = self.respond(sents, confidence, intent, **kwargs)
        else:
            reply = None

        # передача инициативы
        return reply, confidence


    #
    @abc.abstractmethod
    def parse(self, text):
        """
        Каждый диалог может реализовать свою стратегию парсинга, некоторые
        с использованием внешних механизмов, другие - с применением
        регулярных выражений или стандартных парсеров.
        :param text:
        :return:
        """

        return []


    @abc.abstractmethod
    async def interpret(self, sents, **kwargs):
        """
        Интерпретирует высказывание, передаваемое как список предложений,
        прошедших парсинг, обновляет внутреннее состояние диалога, вычисляет
        оценку достоверности интерпретации. Также может возвращать аргументы,
        необходимые конкретному механизму создания ответа.

        :param sents:
        :param kwargs:
        :return:
        """

        return sents, 0.0, 0, kwargs


    @abc.abstractmethod
    def respond(self, sents, confidence, **kwargs):
        """
        Создает ответ, опираясь на входящие высказывания и текущее состояние
        диалога, а также на любые именованные аргументы, переданные методами
        listen и interpret.
        Eсли оценка достоверности равна 0.0, метод может вернуть None или запрос на пояснения.
        Если достоверность оценивается не очень высоко, ответ может включать предположения или
        приблизительные формулировки, а при высокой достоверности — содержать строгие и точные формулировки.
        :param sents:
        :param confidence:
        :param kwargs:
        :return:
        """

        return None


class SimpleConversation(Dialogue, Sequence):
    """
    Простая беседа
    Основная задача класса SimpleConversation — управлять состоянием последовательности диалогов, хранящихся во внутреннем атрибуте.
    Этот класс также наследует класс collections.abc.Sequence из стандартной библиотеки, который позволит классу SimpleConversation
    действовать подобно индексируемому списку диалогов (благодаря абстрактному методу __getitem__) и получать количество диалогов в коллекции (методом __len__):
    """
    def __init__(self, dialogs):
        self._dialogs = dialogs

    def __getitem__(self, idx):
        return self._dialogs[idx]

    def __len__(self):
        return len(self._dialogs)

    def listen(self, text, response=True, **kwargs):
        """
        Просто возвращает ответ с лучшей оценкой достоверности
        :param text:
        :param response:
        :param kwargs:
        :return:
        Передадим входной текст методам listen всех внутренних экземпляров Dialog,
        которые, в свою очередь, вызовут внутренние методы parse, interpret и respond
        """

        responses = [
            dialog.listen(text, response, **kwargs)
            for dialog in self._dialogs
        ]

        # responses - список кортежей (reply, confidence)

        return max(responses, key=itemgetter(1))


    def parse(self, text):
        """
        Возвращает все результаты парсинга, полученные внутренними
        диалогами, для отладки
        :param text:
        :return:
        """

        label = await self.discussion_classifier(text)
        if not label:
            text = ""

        # return [dialog.parse(text) for dialog in self._dialogs]

        return text


    async def discussion_classifier(self, query: str, prompt=domain_classifier_prompt) -> int:
        """
        Классификация вопроса пользователя, имеется ли проблема
        """
        logging.info(f"сработала функция discussion_classifier")
        # ONLY FOR TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ЗАГЛУШКА
        # return 1
        llm_chain = LLMChain(
            llm=OpenAI(
                temperature=0,
                # model_name="gpt-4",
                model_name="gpt-3.5-turbo-instruct",
                max_tokens=1,
                openai_api_key=openai.api_key,
                request_timeout=600,
                max_retries=3
            ),
            prompt=prompt,
        )

        try:
            # Является ли реплика запросом на консултацию или тезхническим вопросом.
            label = await llm_chain.arun(text=query)

        except Exception as e:
            logging.error(f"функция discussion_classifier вызвала ошибку: {e}")
            label = "F"

        # проверяем класс
        if "T" in label:
            label = 1

        else:
            label = 0

        logging.info(f"резлуьтат работы функции discussion_classifier для {query} : {label}")
        return label


    def interept(self, sents: str, **kwargs):
        """
        Возвращает все результаты интерпретации, полученные внутренними
        диалогами, для отладки
        Также добавим поддержку оценки достоверности, чтобы можно было вести беседу согласно уверенности
        в достоверности интерпретации входного текста.
        :param sents:
        :param kwargs:
        :return:
        """

        if len(sents) == 0:
            return sents, 0.0, kwargs

        intent, confidence = await self.intent_classification(sents)

        # return [dialog.interept(sents, **kwargs) for dialog in self._dialogs]

        return sents, intent, confidence


    async def intent_classification(self, query: str, prompt=prompt):
        """
        Классификация намерения
        :param query:
        :param prompt:
        :return:
        """
        intents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        label = 0
        llm_chain = LLMChain(
            llm=OpenAI(
                temperature=0,
                # model_name="gpt-4",
                model_name="gpt-3.5-turbo-instruct",
                max_tokens=2,
                openai_api_key=openai.api_key,
                request_timeout=600,
                max_retries=3
            ),
            prompt=prompt,
        )

        try:
            label = await llm_chain.arun(text=query)

        except Exception as e:
            logging.error(
                f" user_id = 666 ;;; ошибка: {e} в работе функции intent_classification", exc_info=True
            )

        logging.info(f"резлуьтат работы функции intent_classification {query} : {label}")

        if label not in intents:
            label = 0

        return label

    def respond(self, sents, confidence, intent, **kwargs):
        """
        Возвращает все ответы, созданные внутренними диалогами, для отладки
        :param sents:
        :param confidence:
        :param kwargs:
        :return:
        """

        if intent == 1 and confidence >0.5:
            qa_response = await self.qa_retriever(retriever, sents)
            return qa_response




        return [
            dialog.respond(sents, confidence, ** kwargs)
            for dialog in self._dialogs
        ]


    async def qa_retriever(self, retriever, question: str, text) -> str:
        "ответ на вопрос на оснвое контекста отдельно"
        llm = ChatOpenAI(
            model_name="gpt-4-1106-preview",
            # model_name="gpt-4",
            # model_name="gpt-3.5-turbo-16k",
            openai_api_key=openai.api_key,
            max_tokens=1200,
            temperature=0)

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        # chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)
        try:
            # ans = chain({"input_documents": docs[:5], "question": question}, return_only_outputs=True)
            ans = await qa.arun(question)
            logging.info(
                f" user_id = 666 ;;; рузультат qa retriever: {ans}")  # user_id = 1 означает что бот запущен
            # ans = ans["output_text"]

        except Exception as e:
            logging.error(
                f" user_id = 666 ;;; ошибка: {e} в работе модели", exc_info=True
            )
            ans = text

        return ans





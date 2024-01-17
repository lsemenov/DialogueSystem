import openai
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from model.prompt import prompt
from model.classifier_prompt import classifier_prompt
import logging
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from model.prompt import prompt
from model.prompt import QA_CHAIN_PROMPT
import logging
import openai


# load_dotenv()
# openai_api_key: str = getenv('OPENAI_API_KEY')


async def chat_model(question: str, dialog: str = "", history: list = None) -> list[str, list]:
    if history is None:
        history = [[], []]

    log: list = []

    llm_chain = LLMChain(
        llm=ChatOpenAI(
            temperature=0,
            # model_name="gpt-3.5-turbo-16k",
            # model_name="gpt-4",
            model_name="gpt-4-1106-preview",
            max_tokens=1200,
            openai_api_key=openai.api_key,
        ),
        prompt=prompt,
        verbose=True,
    )
    # Get embedding engine ready
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    try:
        logging.info(
            f" user_id = 2 ;;; модель получила на вход: {question}"
        )  # user_id = 2 означает работу модели

        # Create query
        query = ",".join(history[0][-5:][::-1])
        query = question + query

        # load vector store
        db = FAISS.load_local("model/faiss_index", embeddings)

        # Init your retriever. Asking for just 4 document back
        retriever = db.as_retriever(search_kwarg={"k": 5})
        # retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7})
        docs = retriever.get_relevant_documents(query)
        text = "\n".join([x.page_content for x in docs[:5]])

        # qa_ans = await qa_retriever(docs, text, question)
        qa_ans = await qa_retriever(retriever, question, text)

        ans = await llm_chain.arun(
            human_input=question, context=qa_ans, history=dialog
        )
        answer = f"{ans}"

    except Exception as e:
        log.append(e)
        logging.error(
            f" user_id = 666 ;;; ошибка: {e} в работе модели", exc_info=True
        )
        answer = "⚠️ К сожалению произошла ошибка, диалог завершен, попробуйте обратиться к чат боту позже"

    return [answer, log]


async def qa_retriever(retriever, question: str, text) -> str:
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




async def discussion_classifier(query: str, prompt=prompt) -> int:
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



async def filtered(replicas: list[dict]) -> list[str]:
    "Фильтрация обращения пользователя от шума"
    filtered_replicas = []
    for replica in replicas:
        label = await discussion_classifier(replica["user"])
        if label:
            filtered_replicas.append(replica)
    return filtered_replicas



async def classifier_forward_message(answer: str) -> bool:
    # ONLY FOR TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ЗАГЛУШКА
    # return True
    # return False
    """
    Классификация ответа модели для последующей переадресации оператору
    """
    llm_chain = LLMChain(
        llm=OpenAI(
            temperature=0,
            # model_name="gpt-4",
            model_name="gpt-3.5-turbo-instruct",
            max_tokens=1,
            openai_api_key=openai.api_key,
        ),
        prompt=classifier_prompt,
    )

    try:
        # является ли реплика запросом на консультацию или техническим вопросом
        label = await llm_chain.arun(text=answer)

    except Exception as e:
        label = "F"
        logging.error(
            f" user_id = 666 ;;; ошибка: {e} в работе модели классификации ответа", exc_info=True
        )

    logging.info(f"резлуьтат работы функции classifier_forward_message {answer} : {label}")

    # возвращает ответ модели классификатора
    return "T" in label


async def intent_classification(query: str, prompt=prompt):
    """
    Классификация намерения
    :param query:
    :param prompt:
    :return:
    """
    intents = [0,1,2,3,4,5,6,7,8,9]
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



import re
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_gigachat import GigaChat

# --- Конфигурация ---
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
GIGACHAT_CREDENTIALS = "MDE5YzdiZjItZWZjNi03NjEyLWE3ZmItODI3MzQyNzVhMzkwOjYwNmIzM2Y2LTJlZjUtNDY4MC1hOTMyLWRhNTJiNDY4YWI1OQ=="  # Вставьте сюда ваш ключ
GIGACHAT_SCOPE = "GIGACHAT_API_PERS"

# Загружаем эмбеддинги и базу
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Инициализируем GigaChat
llm = GigaChat(
    credentials=GIGACHAT_CREDENTIALS,
    scope=GIGACHAT_SCOPE,
    model="GigaChat",
    temperature=0.4,
    max_tokens=1000,
    verify_ssl_certs=False
)

def extract_task_number(question):
    """Извлекает номер задания из вопроса (число от 13 до 19)."""
    match = re.search(r'\b(1[3-9])\b', question)
    if match:
        return int(match.group(1))
    return None

def ask(question):
    print(f"\nВопрос: {question}")

    # Определяем номер задания
    task_num = extract_task_number(question)
    filter_dict = {"task_number": task_num} if task_num else None
    if filter_dict:
        print(f"Фильтр по заданию {task_num}")
    else:
        print("Номер задания не найден, поиск без фильтра")

    # Получаем релевантные документы
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "filter": filter_dict})
    docs = retriever.invoke(question)

    if not docs:
        print("Документы не найдены.")
        return

    # Формируем контекст из документов
    context = "\n\n".join([doc.page_content for doc in docs])

    # Промпт для LLM
    prompt = f"""Ты — эксперт ЕГЭ по математике, проверяющий вторую часть.
Ответь на вопрос, используя только информацию из предоставленных документов.
Если в документах нет ответа, скажи, что информации недостаточно.

Документы:
{context}

Вопрос: {question}

Ответ:"""

    # Вызываем GigaChat
    response = llm.invoke(prompt)

    print("\n--- ОТВЕТ ---")
    print(response)

    print("\n--- ИСТОЧНИКИ ---")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. (task {doc.metadata.get('task_number', '?')}) {doc.page_content[:150]}...")
        print(f"   Метаданные: {doc.metadata}")
        print()

if __name__ == "__main__":
    question = input("Введите ваш вопрос: ")
    ask(question)
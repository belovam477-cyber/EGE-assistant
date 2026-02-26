import streamlit as st
import re
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# --- Конфигурация ---
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEEPSEEK_API_KEY = "sk-fe6c278f78514b51be27df9913e0beb8"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# Кэшируем загрузку тяжёлых ресурсов
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

@st.cache_resource
def load_llm():
    return ChatOpenAI(
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL,
        temperature=0.2,
        max_tokens=1000
    )

vectorstore = load_vectorstore()
llm = load_llm()

def extract_task_number(question):
    """Извлекает номер задания (13–19) из вопроса."""
    match = re.search(r'\b(1[3-9])\b', question)
    return int(match.group(1)) if match else None

def ask_question(question):
    # Определяем номер задания для фильтрации
    task_num = extract_task_number(question)
    filter_dict = {"task_number": task_num} if task_num else None

    # Ищем документы (k=5, можно увеличить до 10-15, т.к. DeepSeek поддерживает 128k контекста)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "filter": filter_dict})
    docs = retriever.invoke(question)

    if not docs:
        return "Документы не найдены. Попробуйте переформулировать вопрос или уточните номер задания."

    # Формируем контекст из найденных документов
    context = "\n\n".join([doc.page_content for doc in docs])

    # Промпт для DeepSeek
    prompt = f"""Ты — эксперт ЕГЭ по математике, проверяющий вторую часть.
Ответь на вопрос, используя только информацию из предоставленных документов.
Если в документах нет ответа, скажи, что информации недостаточно.

Документы:
{context}

Вопрос: {question}

Ответ:"""

    # Вызываем DeepSeek
    response = llm.invoke(prompt)
    return response.content

# --- Интерфейс Streamlit ---
st.set_page_config(page_title="RAG-помощник по ЕГЭ (DeepSeek)", page_icon="📚", layout="wide")
st.title("📚 RAG-помощник по ЕГЭ (математика, часть 2)")
st.markdown("Задайте вопрос по оформлению заданий **13–19**. Ответ основан на методических материалах ФИПИ 2026, используется модель DeepSeek.")

# Поле ввода
question = st.text_input("✏️ Ваш вопрос:", placeholder="Например: как оформлять задание 17 с подобием треугольников?")

if st.button("🎯 Получить ответ"):
    if question.strip():
        with st.spinner("🔍 Ищу в базе знаний и формирую ответ..."):
            answer = ask_question(question)
        st.success("✅ Ответ:")
        st.write(answer)
    else:
        st.warning("⚠️ Пожалуйста, введите вопрос.")

# Боковая панель с информацией
with st.sidebar:
    st.header("ℹ️ О приложении")
    st.write("""
    **Как это работает:**
    1. Ваш вопрос преобразуется в вектор (модель `intfloat/multilingual-e5-small`).
    2. Векторная база Chroma ищет 5 самых похожих фрагментов.
    3. Найденные фрагменты передаются в DeepSeek вместе с вопросом.
    4. DeepSeek формирует ответ строго по документам.
    
    **Фильтрация:** автоматически определяет номер задания (13–19) и ищет только по нему.
    """)
    st.info("💡 Примеры вопросов:\n- Как оформлять задание 15 с методом рационализации?\n- Какие критерии для 14 задания?\n- Нужно ли чертить рисунок в 17 задании?")
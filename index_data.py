import json
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- Конфигурация ---
INPUT_FILE = "data_clean.jsonl"
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

print("Текущая директория:", os.getcwd())
print("Файлы в папке:", os.listdir())
print("Первая строка файла data_clean.jsonl:")
with open(INPUT_FILE, 'r', encoding='utf-8-sig') as f:
    first = f.readline()
    print(repr(first))

def load_jsonl(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                print(f"Строка {i}: пустая, пропущена")
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Ошибка в строке {i}: {e}")
                print(f"Содержимое строки (repr): {repr(line)}")
                # можно вывести первые несколько символов в hex, чтобы увидеть непечатные символы
                print(f"Первые 20 байт: {line[:20].encode('utf-8')}")
                raise  # остановить выполнение
            text = data["text"]
            metadata = data.get("metadata", {})
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
    return documents

print("Загрузка данных...")
docs = load_jsonl(INPUT_FILE)
print(f"Загружено {len(docs)} документов.")

# Если нужно разбить на более мелкие чанки — раскомментируйте:
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# split_docs = text_splitter.split_documents(docs)
# print(f"После разбиения: {len(split_docs)} чанков.")
split_docs = docs  # если разбиение не требуется

print("Загрузка модели эмбеддингов...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("Создание векторной базы...")
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)

vectorstore.persist()
print(f"Векторная база сохранена в {CHROMA_PATH}")
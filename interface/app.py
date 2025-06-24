import streamlit as st
import pandas as pd
from kafka import KafkaProducer
import json
import time
import os
import uuid
import psycopg2
import matplotlib.pyplot as plt
from psycopg2 import OperationalError

# Конфигурация Kafka
KAFKA_CONFIG = {
    "bootstrap_servers": os.getenv("KAFKA_BROKERS", "kafka:9092"),
    "topic": os.getenv("KAFKA_TOPIC", "transactions")
}

def init_db_connection():
    """Инициализация подключения к PostgreSQL"""
    try:
        return psycopg2.connect(
            dbname="fraud_db",
            user="admin",
            password="password",
            host="postgres",
            connect_timeout=3
        )
    except OperationalError as e:
        st.error("Не удалось подключиться к базе данных. Убедитесь, что сервис PostgreSQL запущен.")
        st.stop()
    except Exception as e:
        st.error(f"Неожиданная ошибка при подключении к БД: {str(e)}")
        st.stop()

def check_table_exists(conn):
    """Проверка существования таблицы transactions"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'transactions'
                );
            """)
            return cur.fetchone()[0]
    except Exception as e:
        st.error(f"Ошибка при проверке таблицы: {str(e)}")
        return False

def load_file(uploaded_file):
    """Загрузка CSV файла в DataFrame"""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {str(e)}")
        return None

def send_to_kafka(df, topic, bootstrap_servers):
    """Отправка данных в Kafka с уникальным ID транзакции"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            security_protocol="PLAINTEXT"
        )
        
        df['transaction_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        progress_bar = st.progress(0)
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            producer.send(
                topic, 
                value={
                    "transaction_id": row['transaction_id'],
                    "data": row.drop('transaction_id').to_dict()
                }
            )
            progress_bar.progress((idx + 1) / total_rows)
            time.sleep(0.01)
            
        producer.flush()
        return True
    except Exception as e:
        st.error(f"Ошибка отправки данных: {str(e)}")
        return False

def show_results():
    """Отображение результатов из PostgreSQL"""
    conn = None
    try:
        conn = init_db_connection()
        
        if not check_table_exists(conn):
            st.warning("Таблица transactions не существует. Дождитесь обработки первых транзакций.")
            return
        
        # Топ-10 фродовых транзакций
        st.subheader("Топ-10 фродовых транзакций")
        df_fraud = pd.read_sql(
            """SELECT transaction_id, score, 
                      TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI:SS') as timestamp 
               FROM transactions 
               WHERE fraud_flag = 1 
               ORDER BY created_at DESC 
               LIMIT 10""",
            conn
        )
        
        if not df_fraud.empty:
            st.dataframe(df_fraud)
        else:
            st.info("Нет фродовых транзакций в базе")
        
        # Гистограмма скоров
        st.subheader("Распределение скоров (последние 100 транзакций)")
        df_scores = pd.read_sql(
            "SELECT score FROM transactions ORDER BY created_at DESC LIMIT 100",
            conn
        )
        
        if not df_scores.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_scores['score'], bins=20, alpha=0.7, color='skyblue')
            ax.set_xlabel('Score', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.grid(axis='y', alpha=0.75)
            st.pyplot(fig)
        else:
            st.info("Нет данных для построения гистограммы")
            
    except Exception as e:
        st.error(f"Ошибка при работе с базой данных: {str(e)}")
    finally:
        if conn:
            conn.close()

# Инициализация состояния
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

# Основной интерфейс
st.title("📤 Отправка данных в Kafka")

# Боковая панель для результатов
if st.sidebar.button("Посмотреть результаты"):
    show_results()

# Основной блок загрузки файлов
uploaded_file = st.file_uploader(
    "Загрузите CSV файл с транзакциями",
    type=["csv"]
)

if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
    st.session_state.uploaded_files[uploaded_file.name] = {
        "status": "Загружен",
        "df": load_file(uploaded_file)
    }
    st.success(f"Файл {uploaded_file.name} успешно загружен!")

# Список загруженных файлов
if st.session_state.uploaded_files:
    st.subheader("🗂 Список загруженных файлов")
    
    for file_name, file_data in st.session_state.uploaded_files.items():
        cols = st.columns([4, 2, 2])
        
        with cols[0]:
            st.markdown(f"**Файл:** `{file_name}`")
            st.markdown(f"**Статус:** `{file_data['status']}`")
        
        with cols[2]:
            if st.button(f"Отправить {file_name}", key=f"send_{file_name}"):
                if file_data["df"] is not None:
                    with st.spinner("Отправка..."):
                        success = send_to_kafka(
                            file_data["df"],
                            KAFKA_CONFIG["topic"],
                            KAFKA_CONFIG["bootstrap_servers"]
                        )
                        if success:
                            st.session_state.uploaded_files[file_name]["status"] = "Отправлен"
                            st.rerun()
                else:
                    st.error("Файл не содержит данных")
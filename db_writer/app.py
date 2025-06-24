import psycopg2
from confluent_kafka import Consumer
import json
import logging
from time import sleep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_table_exists():
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                dbname="fraud_db",
                user="admin",
                password="password",
                host="postgres"
            )
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    score FLOAT NOT NULL,
                    fraud_flag INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_fraud_flag ON transactions(fraud_flag);
                CREATE INDEX IF NOT EXISTS idx_created_at ON transactions(created_at);
            """)
            conn.commit()
            logger.info("Table 'transactions' created or already exists")
            return conn
        except psycopg2.OperationalError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"PostgreSQL not ready, retrying... ({attempt + 1}/{max_retries})")
            sleep(retry_delay)

def consume_messages():
    conn = ensure_table_exists()
    consumer = Consumer({
        'bootstrap.servers': 'kafka:9092',
        'group.id': 'db-writer-group',
        'auto.offset.reset': 'earliest'
    })
    consumer.subscribe(['scoring'])

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        
        try:
            data = json.loads(msg.value())
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO transactions (transaction_id, score, fraud_flag)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (transaction_id) 
                    DO UPDATE SET score = EXCLUDED.score, 
                                fraud_flag = EXCLUDED.fraud_flag,
                                created_at = CURRENT_TIMESTAMP
                """, (data['transaction_id'], data['score'], data['fraud_flag']))
                conn.commit()
            logger.info(f"Inserted transaction {data['transaction_id']}")
        except json.JSONDecodeError:
            logger.error("Invalid JSON message")
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            conn.rollback()

if __name__ == "__main__":
    consume_messages()
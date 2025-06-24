# ML-Ops-2-Kafka

**DISCLAIMER**: Сервис подготовлен в рамках домашнего задания по курсу MLOps. Датасеты предоставлены в рамках соревнования [TETA-ML-1-2025](https://www.kaggle.com/competitions/teta-ml-1-2025).

Система для обнаружения мошеннических транзакций в реальном времени с использованием ML-модели, Kafka для потоковой обработки и PostgreSQL для хранения результатов.

## 🏗️ Архитектура системы

### Компоненты системы:
1. **interface (Streamlit UI)**:
    - Интерфейс для загрузки транзакционных данных через CSV
   - Генерация уникальных transaction_id
   - Отправка сообщений в Kafka топик `transactions`
   - Просмотр результатов через PostgreSQL витрину:
     - Топ-10 фродовых транзакций
     - Гистограмма распределения скоров

2. **fraud_detector (ML Service)**:
   - LightGBM модель (`model.txt`) с порогом 0.95
   - Препроцессинг:
     - Логарифмирование amount/population
     - Временные признаки (час, ночные транзакции)
     - Категориальные риски
     - Частотное кодирование
   - Запись результатов в топик `scoring`

3. **db_writer (PostgreSQL Service)**:
   - Чтение из топика `scoring`
   - Сохранение в БД с полями:
     - transaction_id
     - score
     - fraud_flag
     - timestamp
   - Создает витрину данных для анализа

4. **Kafka Infrastructure**:
   - Zookeeper + Kafka брокер
   - kafka-setup: автоматически создает топики
   - Kafka UI: веб-интерфейс для мониторинга
   - PostgreSQL 13 (витрина fraud_db)

## 🚀 Быстрый старт

### Требования:
- Docker 20.10+
- Docker Compose 2.0+
- 4+ GB свободной памяти

### Запуск системы:
```bash
git clone https://github.com/RushSofi/ML-Ops-2-Kafka.git
cd ML-Ops-2-Kafka 
```
# Сборка и запуск
```bash
docker-compose up --build
```

# Доступ к сервисам:

* Streamlit UI: http://localhost:8501

* Kafka UI: http://localhost:8080

* Логи сервисов:
```bash
docker-compose logs <service_name>  # Например: fraud_detector, kafka, interface
```

# Тестовые данные

* Используйте test.csv из соревнования. Пример структуры:
```bash
transaction_time,amount,cat_id,population_city,...
2023-01-01 12:30:00,150.50,shopping_net,500000,...
```
* Для первых тестов рекомендуется загружать небольшой семпл данных (до 100 транзакций) за раз, чтобы исполнение кода не заняло много времени.

# 🛠️ Использование
1. Отправка транзакций

  - Загрузите CSV через интерфейс

  - Нажмите "Отправить"

  - Мониторьте прогресс в Kafka UI

2. Просмотр результатов

  - В боковой панели нажмите "Посмотреть результаты"

  - Система покажет:

        10 последних фродовых транзакций

        Гистограмму скоров (100 последних)

# Форматы сообщений

Входные (transactions):
```bash
{
  "transaction_id": "uuid",
  "data": {
    "amount": 150.50,
    "cat_id": "shopping_net",
    ...
  }
}
```

Выходные (scoring):
```bash
{
  "transaction_id": "uuid",
  "score": 0.968,
  "fraud_flag": 1
}
```
# Структура проекта
```bash
.
├── fraud_detector/
│   ├── src/
│   │   ├── preprocessing.py    # Фичинжениринг
│   │   ├── scorer.py           # LightGBM модель
│   ├── app/
│   │   ├── app.py                  # Kafka обработчик
│   ├── models/
│   │   ├── model.txt 
│   ├── train_data/
│   │   ├── train.csv
│   ├── requirements.txt
│   └── Dockerfile
├── db_writer/                  # Сервис записи в PostgreSQL
│   ├── requirements.txt
│   ├── Dockerfile
│   └── app.py 
├── interface/
│   ├── .streamlit
│   │   └── config.toml
│   ├── requirements.txt
│   ├── Dockerfile
│   └── app.py                  # Streamlit UI
├── docker-compose.yaml
└── README.md
```
# Настройки инфраструктуры
Kafka:
  Топики:
    - transactions (3 партиции)
    - scoring (3 партиции)
  Репликация: 1 (для разработки)
  Партиции: 3

PostgreSQL:
  БД: fraud_db
  Таблица: transactions


Если сервисы не стартуют:

  - Проверьте свободные порты (8501, 8080, 5432)

  - Увеличьте память Docker (минимум 4GB)

Если нет данных в PostgreSQL:

  - Проверьте подключение db_writer к Kafka

  - Убедитесь, что топик scoring содержит сообщения

Для полного рестарта:
```bash
docker-compose down -v
docker-compose up --build
```
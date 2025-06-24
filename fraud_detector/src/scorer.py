import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.95
logger.info('Importing pretrained LightGBM model...')
model = lgb.Booster(model_file='./models/model.txt')
model_th = DEFAULT_THRESHOLD

def make_pred(dt, source_info="kafka"):
    """Обработка одного сообщения из Kafka"""
    # Проверка фичей
    required_features = model.feature_name()
    missing = set(required_features) - set(dt.columns)
    extra = set(dt.columns) - set(required_features)
    
    if missing:
        logger.warning(f"Missing features: {missing}")
        for feat in missing:
            dt[feat] = 0  # Заполняем нулями
    
    if extra:
        logger.warning(f"Extra features: {extra}")
        dt = dt[required_features]  # Оставляем только нужные
    
    scores = model.predict(dt)
    fraud_flag = (scores >= model_th).astype(int)
    
    logger.info(f'Prediction complete for {source_info} transaction')
    return scores, fraud_flag

def get_feature_importances(top_n=5):
    importance = model.feature_importance(importance_type='gain')
    features = model.feature_name()
    return dict(sorted(zip(features, importance), 
               key=lambda x: x[1], reverse=True)[:top_n])

def plot_score_distribution(scores, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, alpha=0.7)
    plt.title('Distribution of Prediction Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(output_path)
    plt.close()
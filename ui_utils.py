# -*- coding: utf-8 -*-
"""
Утилиты для веб-интерфейса StackMix OCR
"""

import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image


def get_experiment_results(output_dir):
    """Получает список всех экспериментов из директории"""
    experiments = []
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return experiments
    
    for exp_dir in output_path.iterdir():
        if exp_dir.is_dir():
            # Ищем файлы с результатами
            checkpoints = list(exp_dir.glob("*.pt"))
            predictions = list(exp_dir.glob("pred__*.csv"))
            
            if checkpoints or predictions:
                experiments.append({
                    'name': exp_dir.name,
                    'path': str(exp_dir),
                    'checkpoints': [c.name for c in checkpoints],
                    'predictions': [p.name for p in predictions]
                })
    
    return experiments


def load_predictions(pred_file):
    """Загружает предсказания из CSV файла"""
    try:
        df = pd.read_csv(pred_file)
        return df
    except Exception as e:
        return None


def calculate_metrics_summary(df):
    """Вычисляет сводку метрик из dataframe с предсказаниями"""
    if df is None or df.empty:
        return None
    
    # Проверяем наличие необходимых колонок
    if 'pred_text' not in df.columns or 'gt_text' not in df.columns:
        return None
    
    total = len(df)
    correct = (df['pred_text'] == df['gt_text']).sum()
    accuracy = correct / total * 100 if total > 0 else 0
    
    return {
        'total': total,
        'correct': correct,
        'incorrect': total - correct,
        'accuracy': accuracy
    }


def create_metrics_plot(metrics_summary):
    """Создает график метрик"""
    if not metrics_summary:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Correct', 'Incorrect']
    values = [metrics_summary['correct'], metrics_summary['incorrect']]
    colors = ['#4CAF50', '#F44336']
    
    ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title(f"Prediction Results (Accuracy: {metrics_summary['accuracy']:.2f}%)")
    ax.set_ylim(0, max(values) * 1.2)
    
    # Добавляем значения на столбцы
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Сохраняем в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


def format_experiment_info(experiment):
    """Форматирует информацию об эксперименте"""
    info = f"""
### 📁 Эксперимент: {experiment['name']}

**Путь:** `{experiment['path']}`

**Checkpoints ({len(experiment['checkpoints'])}):**
"""
    for cp in experiment['checkpoints']:
        info += f"\n- 💾 {cp}"
    
    info += f"\n\n**Predictions ({len(experiment['predictions'])}):**"
    for pred in experiment['predictions']:
        info += f"\n- 📊 {pred}"
    
    return info


def get_available_configs():
    """Получает список доступных конфигураций датасетов"""
    try:
        from configs import CONFIGS
        return list(CONFIGS.keys())
    except:
        return []


def validate_paths(data_dir, output_dir):
    """Проверяет и создает необходимые директории"""
    errors = []
    
    data_path = Path(data_dir)
    if not data_path.exists():
        errors.append(f"Директория данных не существует: {data_dir}")
    
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Не удалось создать директорию результатов: {str(e)}")
    
    return errors


def get_dataset_statistics(data_dir, dataset_name):
    """Получает статистику по датасету"""
    dataset_path = Path(data_dir) / dataset_name
    marking_file = dataset_path / "marking.csv"
    
    if not marking_file.exists():
        return None
    
    try:
        df = pd.read_csv(marking_file)
        
        stats = {
            'total': len(df),
            'train': len(df[df['stage'] == 'train']),
            'valid': len(df[df['stage'] == 'valid']),
            'test': len(df[df['stage'] == 'test']),
            'avg_text_length': df['text'].str.len().mean(),
            'max_text_length': df['text'].str.len().max(),
            'min_text_length': df['text'].str.len().min()
        }
        
        return stats
    except Exception as e:
        return None


def create_dataset_statistics_plot(stats):
    """Создает график статистики датасета"""
    if not stats:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # График распределения по stage
    stages = ['Train', 'Valid', 'Test']
    counts = [stats['train'], stats['valid'], stats['test']]
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    
    ax1.bar(stages, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Sample Count')
    ax1.set_title('Dataset Split')
    ax1.set_ylim(0, max(counts) * 1.2)
    
    for i, v in enumerate(counts):
        ax1.text(i, v + max(counts) * 0.02, str(v), ha='center', va='bottom', fontweight='bold')
    
    # График длины текста
    text_stats = ['Min', 'Avg', 'Max']
    text_lengths = [stats['min_text_length'], stats['avg_text_length'], stats['max_text_length']]
    
    ax2.bar(text_stats, text_lengths, color='#9C27B0', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Character Count')
    ax2.set_title('Text Length Statistics')
    ax2.set_ylim(0, max(text_lengths) * 1.2)
    
    for i, v in enumerate(text_lengths):
        ax2.text(i, v + max(text_lengths) * 0.02, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Сохраняем в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

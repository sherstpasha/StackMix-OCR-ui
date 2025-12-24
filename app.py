# -*- coding: utf-8 -*-
"""
StackMix OCR Web Interface
Веб-интерфейс для обучения и генерации данных OCR
"""

import gradio as gr
import subprocess
import os
import threading
import time
from pathlib import Path
import pandas as pd
import json
import torch
import cv2
import numpy as np
from configs.base import BaseConfig

# Утилиты для работы с Unicode путями
def imwrite_unicode(path, img):
    """Сохраняет изображение с поддержкой Unicode путей"""
    ext = os.path.splitext(str(path))[1]
    result, encoded_img = cv2.imencode(ext, img)
    if result:
        with open(path, 'wb') as f:
            f.write(encoded_img)
        return True
    return False

# Глобальные переменные для отслеживания процесса
current_process = None
current_thread = None
stop_event = None
progress_status = {'epoch': (0, 0), 'iter': (0, 0)}
tb_process = None
tb_port = 6006


def validate_dataset_path(data_dir, marking_csv_path):
    """Проверяет наличие датасета и файла разметки"""
    data_path = Path(data_dir)
    marking_file = Path(marking_csv_path)
    
    if not data_path.exists():
        return False, f"Директория датасета не найдена: {data_path}"
    
    if not marking_file.exists():
        return False, f"Файл разметки не найден: {marking_file}"
    
    # Проверяем структуру CSV
    try:
        df = pd.read_csv(marking_file)
        required_columns = ['sample_id', 'path', 'stage', 'text']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"В marking.csv отсутствуют столбцы: {missing_cols}"
        
        return True, f"Датасет найден ({len(df)} записей)"
    except Exception as e:
        return False, f"Ошибка чтения marking.csv: {str(e)}"


def run_training_process(
    data_dir,
    marking_csv_path,
    experiment_name,
    output_dir,
    experiment_description,
    image_w,
    image_h,
    num_epochs,
    batch_size,
    num_workers,
    use_blot,
    use_augs,
    use_stackmix,
    use_pretrained_backbone,
    seed,
    checkpoint_path,
    neptune_project,
    neptune_token,
    mwe_tokens_dir,
    stop_evt=None
):
    """Запускает процесс обучения с логированием в TensorBoard"""
    global current_process, tb_process, tb_port
    
    try:
        # Подготовка директорий
        experiment_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Создаём TensorBoard writer ОБЯЗАТЕЛЬНО
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_logdir = os.path.join(experiment_dir, 'tensorboard')
            os.makedirs(tb_logdir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_logdir)
            writer.add_text('config/experiment_name', experiment_name, 0)
            writer.add_text('config/description', experiment_description, 0)
            writer.add_text('config/dataset', marking_csv_path, 0)
            writer.add_text('config/params', f"epochs={num_epochs}, batch_size={batch_size}, image_size={image_w}x{image_h}", 0)
            writer.add_text('config/augmentations', f"blot={use_blot}, augs={use_augs}, stackmix={use_stackmix}", 0)
            writer.flush()
            
            # Автоматически запускаем TensorBoard и открываем в браузере
            # Запускаем ПОСЛЕ создания writer чтобы файлы уже существовали
            if tb_process is None:
                try:
                    import sys
                    import webbrowser
                    # TensorBoard смотрит на папку эксперимента (не на tensorboard внутри)
                    cmd = [sys.executable, '-m', 'tensorboard', '--logdir', experiment_dir, '--port', str(tb_port), '--bind_all']
                    tb_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    time.sleep(3)  # Даём больше времени TensorBoard запуститься
                    webbrowser.open(f'http://localhost:{tb_port}/')
                    writer.add_text('tensorboard/url', f'TensorBoard открыт: http://localhost:{tb_port}/', 0)
                    writer.flush()
                except Exception as e:
                    writer.add_text('tensorboard/error', f'Не удалось запустить TensorBoard: {e}', 0)
                    writer.flush()
        except ImportError as e:
            raise ImportError("TensorBoard is required. Install it with: pip install tensorboard") from e
        
        # Создаем и запускаем обучение напрямую через Python
        import sys
        sys.path.insert(0, '.')
        
        from src.experiment import OCRExperiment
        from src.model import get_ocr_model
        from src.dataset import DatasetRetriever
        from src.ctc_labeling import CTCLabeling
        from src import utils
        from src.predictor import Predictor
        from src.metrics import string_accuracy, cer, wer
        from configs.base import BaseConfig
        import torch
        import albumentations as A
        from torch.utils.data import SequentialSampler, RandomSampler
        
        # Читаем датасет
        df = pd.read_csv(marking_csv_path, index_col='sample_id')
        
        # Нормализуем пути: заменяем обратные слеши на прямые
        df['path'] = df['path'].str.replace('\\', '/')
        
        # ВАЖНО: Удаляем строки с NaN/пустыми текстами
        df = df.dropna(subset=['text'])
        df = df[df['text'].astype(str).str.strip() != '']
        
        if len(df) == 0:
            raise ValueError("После фильтрации пустых текстов датасет пуст!")
        
        writer.add_text('dataset/info', f'Строк после фильтрации: {len(df)}', 0)
        
        # Собираем уникальные символы из текстов
        all_text = ' '.join(df['text'].astype(str).values)
        chars = ''.join(sorted(set(all_text)))
        
        # Создаем конфиг
        config_name = f"custom_{experiment_name}"
        config = BaseConfig(
            data_dir=data_dir,
            dataset_name=config_name,
            image_w=image_w,
            image_h=image_h,
            chars=chars,
            corpus_name="custom"
        )
        
        config.params['experiment_name'] = experiment_name
        config.params['experiment_description'] = experiment_description
        config.params['bs'] = batch_size
        config.params['num_workers'] = num_workers
        config.params['num_epochs'] = num_epochs
        
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        writer.add_text('config/device', str(device), 0)
        
        ctc_labeling = CTCLabeling(config)
        
        # Подготовка трансформаций
        transforms = None
        if use_blot and use_augs:
            from src.blot import get_blot_transforms
            transforms = A.Compose([
                get_blot_transforms(config),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.25),
                A.Rotate(limit=3, interpolation=1, border_mode=0, p=0.5),
                A.ImageCompression(quality_range=(75, 100), p=0.5),
            ], p=1.0)
        elif use_blot:
            from src.blot import get_blot_transforms
            transforms = get_blot_transforms(config)
        elif use_augs:
            transforms = A.Compose([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.25),
                A.Rotate(limit=3, interpolation=1, border_mode=0, p=0.5),
                A.ImageCompression(quality_range=(75, 100), p=0.5),
            ], p=1.0)
        
        # Датасеты
        train_dataset_kwargs = {'transforms': transforms}
        if use_stackmix and mwe_tokens_dir:
            from src.stackmix import StackMix
            stackmix = StackMix(
                mwe_tokens_dir=mwe_tokens_dir,
                data_dir=data_dir,
                dataset_name=config_name,
                image_h=image_h,
            )
            stackmix.load()
            train_dataset_kwargs['stackmix'] = stackmix
        
        df_train = df[~df['stage'].isin(['valid', 'test'])]
        train_dataset = DatasetRetriever(df_train, config, ctc_labeling, **train_dataset_kwargs)
        valid_dataset = DatasetRetriever(df[df['stage'] == 'valid'], config, ctc_labeling)
        test_dataset = DatasetRetriever(df[df['stage'] == 'test'], config, ctc_labeling)
        
        # Модель
        model = get_ocr_model(config, pretrained=bool(use_pretrained_backbone))
        model = model.to(device)
        
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), **config.params['optimizer']['params'])
        
        # DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=RandomSampler(train_dataset),
            pin_memory=False,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=utils.kw_collate_fn
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(valid_dataset),
            pin_memory=False,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=utils.kw_collate_fn
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            **config.params['scheduler']['params'],
        )
        
        # Простая реализация цикла обучения с полным логированием в TensorBoard
        best_cer = float('inf')
        epoch_count = int(num_epochs)
        global_step = 0
        
        writer.add_text('training/start', f'Начало обучения на {num_epochs} эпох', 0)
        writer.flush()
        
        for epoch in range(1, epoch_count + 1):
            # Проверка остановки
            if stop_evt is not None and stop_evt.is_set():
                writer.add_text('training/status', f'Остановка по запросу пользователя на эпохе {epoch}', epoch)
                break

            model.train()
            epoch_losses = []
            n_batches = len(train_loader)
            for batch_idx, batch in enumerate(train_loader, start=1):
                if stop_evt is not None and stop_evt.is_set():
                    writer.add_text('training/status', f'Остановка по запросу пользователя (эпоха {epoch}, батч {batch_idx})', global_step)
                    break

                # forward/backward
                lengths = batch['encoded_length'].to(device, dtype=torch.int32)
                encoded = batch['encoded'].to(device, dtype=torch.int32)
                outputs = model(batch['image'].to(device, dtype=torch.float32))

                preds_size = torch.IntTensor([outputs.size(1)] * batch['encoded'].shape[0])
                preds = outputs.log_softmax(2).permute(1, 0, 2)

                loss = criterion(preds, encoded, preds_size, lengths)
                loss_value = loss.detach().cpu().item()
                epoch_losses.append(loss_value)

                if torch.isfinite(loss):
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    try:
                        scheduler.step()
                    except Exception:
                        pass

                # Логируем в TensorBoard всё
                writer.add_scalar('train/loss', loss_value, global_step)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                global_step += 1
                
                # Периодический flush (каждые 100 батчей)
                if batch_idx % 100 == 0:
                    writer.flush()

                # Обновляем прогресс
                progress_status['epoch'] = (epoch, epoch_count)
                progress_status['iter'] = (batch_idx, n_batches)

            # Логируем среднюю loss за эпоху
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            writer.add_scalar('train/avg_epoch_loss', avg_epoch_loss, epoch)
            
            # Валидация в конце эпохи
            model.eval()
            predictor = Predictor(model, device)
            try:
                predictions = predictor.run_inference(valid_loader)
            except Exception as e:
                writer.add_text('validation/error', f'Ошибка при валидации: {str(e)}', epoch)
                predictions = []

            if predictions:
                df_pred = pd.DataFrame([{
                    'id': p['id'],
                    'pred_text': ctc_labeling.decode(p['raw_output'].argmax(1)),
                    'gt_text': p['gt_text']
                } for p in predictions]).set_index('id')

                cer_metric = round(cer(df_pred['pred_text'], df_pred['gt_text']), 5)
                wer_metric = round(wer(df_pred['pred_text'], df_pred['gt_text']), 5)
                acc_metric = round(string_accuracy(df_pred['pred_text'], df_pred['gt_text']), 5)

                writer.add_scalar('val/cer', cer_metric, epoch)
                writer.add_scalar('val/wer', wer_metric, epoch)
                writer.add_scalar('val/acc', acc_metric, epoch)
                writer.add_text('val/metrics', f'CER: {cer_metric}, WER: {wer_metric}, ACC: {acc_metric}', epoch)

                # Сохраняем лучший по CER
                if cer_metric < best_cer:
                    best_cer = cer_metric
                    torch.save({'model_state_dict': model.state_dict(), 'cer': cer_metric}, os.path.join(experiment_dir, 'best_cer.pt'))
                    writer.add_text('checkpoints/best', f'Новая лучшая модель на эпохе {epoch} с CER={cer_metric}', epoch)

                # Сохраняем предсказания
                df_pred.to_csv(os.path.join(experiment_dir, f'pred__epoch_{epoch}.csv'))

            # Сохраняем последний чекпоинт
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, os.path.join(experiment_dir, 'last.pt'))

        writer.add_text('training/status', 'Обучение завершено успешно', epoch_count)
        writer.flush()
        writer.close()
            
    except Exception as e:
        if 'writer' in locals():
            import traceback
            error_msg = traceback.format_exc()
            writer.add_text('training/error', f'Ошибка при обучении: {str(e)}\n\n{error_msg}', 0)
            writer.flush()
            writer.close()
        raise
    finally:
        current_process = None


def auto_update_logs_loop():
    """Функция для периодического обновления логов в отдельном потоке"""
    global auto_update_active
    while auto_update_active:
        time.sleep(1)
        # Обновление произойдёт через кнопку обновления
        

def handle_start_training(
    data_dir,
    marking_csv_path,
    experiment_name,
    output_dir,
    experiment_description,
    image_w,
    image_h,
    num_epochs,
    batch_size,
    num_workers,
    use_blot,
    use_augs,
    use_stackmix,
    use_pretrained_backbone,
    seed,
    checkpoint_path,
    neptune_project,
    neptune_token,
    mwe_tokens_dir
):
    """Обработчик кнопки запуска обучения"""
    global current_thread, current_process, stop_event, auto_update_active
    
    if current_process is not None or (current_thread and current_thread.is_alive()):
        return "Обучение уже запущено! Остановите текущий процесс перед запуском нового."
    
    # Валидация
    if not experiment_name.strip():
        return "Укажите название эксперимента"
    
    if not marking_csv_path.strip():
        return "Укажите путь к marking.csv"
    
    # Проверяем датасет
    is_valid, message = validate_dataset_path(data_dir, marking_csv_path)
    if not is_valid:
        return f"Ошибка: {message}"
    
    # Сброс прогресса
    progress_status['epoch'] = (0, 0)
    progress_status['iter'] = (0, 0)
    
    # Запускаем в отдельном потоке
    stop_event = threading.Event()
    
    # Обёртка для отлова ошибок
    def training_wrapper():
        try:
            run_training_process(
                data_dir, marking_csv_path, experiment_name, output_dir,
                experiment_description, image_w, image_h, num_epochs, batch_size,
                num_workers, use_blot, use_augs, use_stackmix,
                use_pretrained_backbone, seed, checkpoint_path, neptune_project,
                neptune_token, mwe_tokens_dir, stop_event
            )
        except Exception as e:
            import traceback
            error_msg = f"ОШИБКА при запуске обучения:\n{str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            # Пишем в файл для отладки
            try:
                os.makedirs(os.path.join(output_dir, experiment_name), exist_ok=True)
                with open(os.path.join(output_dir, experiment_name, 'error.log'), 'w', encoding='utf-8') as f:
                    f.write(error_msg)
            except Exception as e2:
                print(f"Failed to write error log: {e2}")
    
    current_thread = threading.Thread(target=training_wrapper)
    current_thread.daemon = True
    current_thread.start()
    current_process = True
    
    return f"Обучение запущено! Эксперимент: {experiment_name}\n\nTensorBoard автоматически откроется в браузере на http://127.0.0.1:6006/"


def stop_training():
    """Останавливает процесс обучения"""
    global current_process, stop_event, current_thread

    if current_process is None:
        return "Обучение не запущено"

    # Устанавливаем событие остановки
    if stop_event is not None:
        stop_event.set()

    # Дождёмся завершения потока (короткий таймаут)
    try:
        if current_thread is not None:
            current_thread.join(timeout=5)
    except Exception:
        pass

    current_process = None
    return "Обучение остановлено"


def start_tensorboard(output_dir, experiment_name, port=6006):
    """Запускает tensorboard для указанного эксперимента на фоновом процессе"""
    global tb_process, tb_port
    tb_port = int(port)
    logdir = os.path.join(output_dir, experiment_name, 'tensorboard')
    if not os.path.exists(logdir):
        return False, f"Logdir не найден: {logdir}"

    if tb_process is not None:
        return True, f"TensorBoard уже запущен на порту {tb_port}"

    try:
        import sys
        cmd = [sys.executable, '-m', 'tensorboard', '--logdir', logdir, '--port', str(tb_port), '--host', '127.0.0.1']
        tb_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(1)
        return True, f"TensorBoard запущен: http://127.0.0.1:{tb_port}/"
    except Exception as e:
        return False, f"Не удалось запустить TensorBoard: {e}"


def stop_tensorboard():
    global tb_process
    if tb_process is None:
        return False, "TensorBoard не запущен"
    try:
        tb_process.terminate()
        tb_process.wait(timeout=3)
    except Exception:
        try:
            tb_process.kill()
        except Exception:
            pass
    tb_process = None
    return True, "TensorBoard остановлен"


def start_tb_ui(output_dir, experiment_name, port):
    ok, msg = start_tensorboard(output_dir, experiment_name, port)
    return msg


def stop_tb_ui():
    ok, msg = stop_tensorboard()
    return msg


def check_dataset(data_dir, marking_csv_path):
    """Проверяет датасет"""
    is_valid, message = validate_dataset_path(data_dir, marking_csv_path)
    
    if is_valid:
        df = pd.read_csv(marking_csv_path)
        
        # Проверяем пустые/NaN текстовые поля
        nan_count = df['text'].isna().sum()
        empty_count = (df['text'].astype(str).str.strip() == '').sum()
        
        train_count = len(df[df['stage'] == 'train'])
        valid_count = len(df[df['stage'] == 'valid'])
        test_count = len(df[df['stage'] == 'test'])
        
        # Проверяем существование файлов изображений
        missing_images = []
        for idx, row in df.head(5).iterrows():
            img_path = Path(data_dir) / row['path'].replace('\\', '/')
            if not img_path.exists():
                missing_images.append(f"  - {row['path']} -> {img_path}")
        
        missing_warning = ""
        if missing_images:
            missing_warning = f"""
**ВНИМАНИЕ: Некоторые изображения не найдены!**

Проверьте, что `data_dir` указывает на корневую папку, относительно которой заданы пути в CSV.

Не найдены:
```
{chr(10).join(missing_images[:3])}
```

Убедитесь, что:
1. `data_dir` = корневая папка (где лежат train/, val/, и т.д.)
2. Пути в CSV относительны к `data_dir`
3. Используйте прямые слеши (/) или двойные обратные (\\\\) в путях
"""
        
        # Предупреждение о пустых текстах
        empty_warning = ""
        if nan_count > 0 or empty_count > 0:
            empty_warning = f"""
**⚠️ ВНИМАНИЕ: Обнаружены пустые тексты!**

- Пустых (NaN): {nan_count}
- Пустых строк: {empty_count}

Эти строки будут автоматически удалены при обучении.
"""
        
        info = f"""
**Информация о датасете**

- Статус: Датасет найден и валиден
- Путь к marking.csv: {marking_csv_path}
- Data directory: {data_dir}
- Всего записей: {len(df)}
- Train: {train_count}
- Valid: {valid_count}
- Test: {test_count}

{empty_warning}

{missing_warning}

**Первые записи:**
```
{df.head(3).to_string()}
```
"""
        return info
    else:
        return f"Ошибка: {message}"


def prepare_char_masks(checkpoint_path, data_dir, marking_csv_path, image_w, image_h, batch_size, num_workers):
    """Подготавливает маски символов из обученной модели"""
    try:
        from src.dataset import DatasetRetriever
        from src.ctc_labeling import CTCLabeling
        from src.model import get_ocr_model
        from src.predictor import Predictor
        from src.char_masks import CharMasks
        from src import utils
        from torch.utils.data import SequentialSampler
        import json
        
        # Проверяем checkpoint
        if not Path(checkpoint_path).exists():
            return f"Ошибка: Checkpoint не найден: {checkpoint_path}"
        
        if not Path(marking_csv_path).exists():
            return f"Ошибка: marking.csv не найден: {marking_csv_path}"
        
        # Загружаем датасет С ИНДЕКСОМ sample_id - это важно!
        df = pd.read_csv(marking_csv_path, index_col='sample_id')
        df['path'] = df['path'].str.replace('\\', '/')
        
        # ВАЖНО: Удаляем строки с NaN/пустыми текстами
        df = df.dropna(subset=['text'])
        df = df[df['text'].astype(str).str.strip() != '']
        
        if len(df) == 0:
            return "Ошибка: После фильтрации пустых текстов датасет пуст!"
        
        # Получаем уникальные символы
        all_chars = set()
        for text in df['text'].values:
            all_chars.update(str(text))
        chars = ''.join(sorted(list(all_chars)))
        
        # Создаем конфиг
        config = BaseConfig(
            data_dir=data_dir,
            dataset_name='custom',
            chars=chars,
            corpus_name='custom_corpus',
            image_w=int(image_w),
            image_h=int(image_h),
        )
        
        # Дополнительные параметры
        config.params['experiment_name'] = 'char_masks_prep'
        config.params['experiment_description'] = 'Preparing char masks'
        config.params['num_epochs'] = 0
        config.params['bs'] = int(batch_size)
        config.params['num_workers'] = int(num_workers)
        
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        # CTCLabeling
        ctc_labeling = CTCLabeling(config)
        
        # Создаем тренировочный датасет
        train_df = df[~df['stage'].isin(['valid', 'test'])]
        train_dataset = DatasetRetriever(train_df, config, ctc_labeling)
        
        # Загружаем модель
        model = get_ocr_model(config)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(device)
        
        # DataLoader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['bs'],
            sampler=SequentialSampler(train_dataset),
            pin_memory=False,
            drop_last=False,
            num_workers=config['num_workers'],
            collate_fn=utils.kw_collate_fn
        )
        
        # Инференс
        predictor = Predictor(model, device)
        train_inference = predictor.run_inference(train_loader)
        
        # Создаем маски
        char_masks = CharMasks(config, ctc_labeling, add=0, blank_add=0)
        all_masks, bad = char_masks.run(train_inference)
        
        # Конвертируем numpy int64 в обычные int для JSON
        def convert_to_serializable(obj):
            """Рекурсивно конвертирует numpy типы в Python типы"""
            if isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            return obj
        
        all_masks_serializable = convert_to_serializable(all_masks)
        
        # Сохраняем в подпапку custom (как ожидает StackMix)
        output_subdir = Path(data_dir) / 'custom'
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_path = output_subdir / 'all_char_masks.json'
        
        with open(output_path, 'w') as f:
            json.dump(all_masks_serializable, f)
        
        return f"✓ Маски символов подготовлены!\n\nХорошие маски: {len(all_masks)}\nПлохие маски: {len(bad)}\n\nСохранено: {output_path}"
        
    except Exception as e:
        import traceback
        return f"Ошибка при подготовке масок:\n{str(e)}\n\n{traceback.format_exc()}"


def prepare_stackmix_tokens(data_dir, marking_csv_path, image_h, tokens_dir, num_workers):
    """ШАГ 2: Создаёт токены StackMix из масок символов"""
    try:
        from src.stackmix import StackMix
        
        # Проверяем all_char_masks.json в подпапке custom
        masks_path = Path(data_dir) / 'custom' / 'all_char_masks.json'
        if not masks_path.exists():
            # Проверяем также в корне (на случай старого формата)
            old_masks_path = Path(data_dir) / 'all_char_masks.json'
            if old_masks_path.exists():
                # Копируем в правильное место
                import shutil
                masks_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(old_masks_path, masks_path)
            else:
                return f"Ошибка: Сначала нужно подготовить маски символов (Шаг 1)!\n\nФайл не найден: {masks_path}"
        
        if not Path(marking_csv_path).exists():
            return f"Ошибка: marking.csv не найден: {marking_csv_path}"
        
        # Создаем mwe_tokens_dir
        mwe_tokens_dir = Path(tokens_dir)
        mwe_tokens_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем разметку С ИНДЕКСОМ sample_id
        marking = pd.read_csv(marking_csv_path, index_col='sample_id')
        marking['path'] = marking['path'].str.replace('\\', '/')
        
        # Удаляем строки с NaN/пустыми текстами
        marking = marking.dropna(subset=['text'])
        marking = marking[marking['text'].astype(str).str.strip() != '']
        
        if len(marking) == 0:
            return "Ошибка: После фильтрации пустых текстов датасет пуст!"
        
        # Создаем StackMix
        stackmix = StackMix(
            mwe_tokens_dir=str(mwe_tokens_dir),
            data_dir=data_dir,
            dataset_name='custom',
            image_h=int(image_h),
            p_background_smoothing=0.1
        )
        
        status = "=== Создание токенов StackMix ===\n\n"
        
        # Проверяем данные
        train_records = marking[~marking['stage'].isin(['valid', 'test'])]
        train_count = len(train_records)
        status += f"Тренировочных записей: {train_count}\n"
        
        if train_count == 0:
            return "Ошибка: В датасете нет тренировочных записей (stage='train')!"
        
        # Показываем примеры путей
        sample_paths = train_records['path'].head(3).tolist()
        status += f"Примеры путей:\n"
        for p in sample_paths:
            full_path = Path(data_dir) / p
            status += f"  - {p} -> {'✓' if full_path.exists() else '✗'}\n"
        status += "\n"
        
        status += f"Запуск создания токенов (workers={num_workers})...\n"
        
        try:
            stackmix.prepare_stackmix_dir(marking, num_workers=int(num_workers))
        except Exception as e:
            import traceback
            return f"Ошибка при создании токенов:\n{str(e)}\n\n{traceback.format_exc()}"
        
        # Проверяем результат
        try:
            stackmix.load()
            if stackmix.stackmix_data is None or len(stackmix.stackmix_data) == 0:
                return "Ошибка: Токены не созданы! Проверьте пути к изображениям."
            
            status += f"\n✓ Токены успешно созданы!\n\n"
            status += f"📊 Статистика:\n"
            status += f"  - Всего токенов: {len(stackmix.stackmix_data)}\n"
            status += f"  - Уникальных токенов: {stackmix.stackmix_data['text'].nunique()}\n"
            status += f"  - Сохранено в: {mwe_tokens_dir}/custom/\n\n"
            status += f"📁 Путь к токенам: {mwe_tokens_dir}\n\n"
            status += "Теперь можно переходить к Шагу 3 - генерация изображений."
            
        except Exception as e:
            return f"Ошибка при загрузке токенов:\n{str(e)}"
        
        return status
        
    except Exception as e:
        import traceback
        return f"Ошибка:\n{str(e)}\n\n{traceback.format_exc()}"


def generate_images_from_corpus(tokens_dir, data_dir, marking_csv_path, text_file, image_h, output_dir, num_samples):
    """ШАГ 3: Генерирует изображения из корпуса текстов"""
    try:
        from src.stackmix import StackMix
        from src.ctc_labeling import CTCLabeling
        import cv2
        
        # Автоматически корректируем путь к токенам, если пользователь указал custom на конце
        mwe_tokens_dir = Path(tokens_dir)
        if mwe_tokens_dir.name.lower() == 'custom':
            mwe_tokens_dir = mwe_tokens_dir.parent
        real_tokens_dir = mwe_tokens_dir / 'custom'
        stackmix_csv = real_tokens_dir / 'stackmix.csv'
        
        if not stackmix_csv.exists():
            return f"Ошибка: Сначала создайте токены (Шаг 2)!\n\nФайл не найден: {stackmix_csv}"
        
        if not text_file or not Path(text_file).exists():
            return f"Ошибка: Укажите файл с текстом для генерации!"
        
        # Загружаем разметку для получения символов
        marking = pd.read_csv(marking_csv_path, index_col='sample_id')
        marking = marking.dropna(subset=['text'])
        
        all_chars = set()
        for text in marking['text'].values:
            all_chars.update(str(text))
        chars = ''.join(sorted(list(all_chars)))
        
        # Создаем конфиг
        config = BaseConfig(
            data_dir=data_dir,
            dataset_name='custom',
            chars=chars,
            corpus_name='custom_corpus',
            image_w=512,
            image_h=int(image_h),
        )
        
        ctc_labeling = CTCLabeling(config)
        
        # Создаем StackMix и загружаем токены
        stackmix = StackMix(
            mwe_tokens_dir=str(mwe_tokens_dir),
            data_dir=data_dir,
            dataset_name='custom',
            image_h=int(image_h),
            p_background_smoothing=0.1
        )
        stackmix.load()
        
        status = "=== Генерация изображений ===\n\n"
        status += f"Путь к токенам: {real_tokens_dir}\n"
        status += f"Загружено токенов: {len(stackmix.stackmix_data)}\n"
        status += f"Уникальных: {stackmix.stackmix_data['text'].nunique()}\n\n"
        
        # Проверяем символы в корпусе
        corpus_chars = set()
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            for line in lines:
                corpus_chars.update(line)
        
        dataset_chars = set(chars)
        missing_chars = corpus_chars - dataset_chars - {'\n', '\r', '\t', ' '}
        
        if missing_chars:
            status += f"⚠️ В корпусе есть символы, которых нет в датасете:\n"
            status += f"   {sorted(missing_chars)[:20]}{'...' if len(missing_chars) > 20 else ''}\n\n"
        
        # Загружаем корпус (копируем во временную директорию рядом с токенами)
        corpus_temp = mwe_tokens_dir / 'corpus_temp.txt'
        import shutil
        shutil.copy(text_file, corpus_temp)
        
        stackmix.load_corpus(ctc_labeling, str(corpus_temp))
        status += f"Корпус загружен: {len(stackmix.corpus)} подходящих строк\n\n"
        
        if len(stackmix.corpus) == 0:
            return status + "Ошибка: В корпусе нет строк с подходящими символами!"
        
        # Генерируем (создаем директорию для изображений)
        gen_dir = Path(output_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        
        actual_samples = min(int(num_samples), len(stackmix.corpus))
        status += f"Генерация {actual_samples} изображений (параллельно)...\n"
        
        # Параллельная генерация
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import random
        
        # Выбираем тексты заранее (чтобы избежать race condition)
        selected_texts = [random.choice(stackmix.corpus) for _ in range(actual_samples)]
        
        generated = []
        errors = []
        errors_lock = threading.Lock()
        generated_lock = threading.Lock()
        
        def generate_single(idx, text):
            """Генерирует одно изображение"""
            try:
                image = stackmix.run_stackmix(text)
                if image is not None:
                    img_path = gen_dir / f'gen_{idx:05d}.png'
                    imwrite_unicode(img_path, image)
                    return {
                        'sample_id': idx,
                        'path': f'gen_{idx:05d}.png',
                        'text': text,
                        'stage': 'train'
                    }
                else:
                    return None, f"Попытка {idx}: run_stackmix вернул None"
            except Exception as e:
                return None, f"Попытка {idx}: {type(e).__name__}: {str(e)}"
        
        num_workers = min(8, actual_samples)  # Не больше 8 воркеров
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(generate_single, i, text): i 
                      for i, text in enumerate(selected_texts)}
            
            for future in as_completed(futures):
                result = future.result()
                if isinstance(result, dict):
                    with generated_lock:
                        generated.append(result)
                elif isinstance(result, tuple) and result[0] is None:
                    with errors_lock:
                        errors.append(result[1])
        
        # Сортируем по sample_id для консистентности
        generated.sort(key=lambda x: x['sample_id'])
        
        # Сохраняем разметку
        gen_marking = pd.DataFrame(generated)
        marking_path = gen_dir / 'marking.csv'
        gen_marking.to_csv(marking_path, index=False)
        
        status += f"\n✓ Генерация завершена!\n\n"
        status += f"📊 Результат:\n"
        status += f"  - Сгенерировано: {len(generated)} изображений\n"
        status += f"  - Ошибок: {len(errors)}\n"
        
        if errors:
            status += f"\n⚠️ Ошибки:\n"
            for err in errors[:5]:  # Показываем первые 5 ошибок
                status += f"  - {err}\n"
            if len(errors) > 5:
                status += f"  ... и ещё {len(errors) - 5} ошибок\n"
        
        status += f"\n  - Папка: {gen_dir}\n"
        status += f"  - Разметка: {marking_path}\n\n"
        
        if len(generated) > 0:
            status += "✅ Сгенерированные данные готовы для обучения!"
        else:
            status += "❌ Не удалось сгенерировать изображения! Проверьте ошибки выше."
        
        return status
        
    except Exception as e:
        import traceback
        return f"Ошибка:\n{str(e)}\n\n{traceback.format_exc()}"






# Создаем интерфейс Gradio
with gr.Blocks(title="StackMix OCR Training") as app:
    gr.Markdown("""
    # StackMix OCR - Обучение и генерация
    
    **Важно для работы с путями:**
    - `data_dir` должен быть корневой папкой, откуда начинаются пути в marking.csv
    - Пути в CSV должны быть относительными к `data_dir`
    - Используйте прямые слеши `/` или двойные обратные `\\\\` в путях
    
    **Пример:**
    - data_dir: `C:/Users/USER/Desktop/dataset`
    - путь в CSV: `train/img/image1.png`
    - полный путь: `C:/Users/USER/Desktop/dataset/train/img/image1.png`
    """)
    
    with gr.Tabs():
        # ========== ВКЛАДКА: ОБУЧЕНИЕ ==========
        with gr.Tab("Обучение модели"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Основные параметры")
                    
                    data_dir = gr.Textbox(
                        label="Директория с данными",
                        value="./data",
                        placeholder="Путь к папке с данными"
                    )
                    
                    marking_csv_path = gr.Textbox(
                        label="Путь к marking.csv",
                        value="./data/marking.csv",
                        placeholder="Полный путь к файлу marking.csv"
                    )
                    
                    experiment_name = gr.Textbox(
                        label="Название эксперимента",
                        value="OCR_experiment",
                        placeholder="my_experiment_v1"
                    )
                    
                    output_dir = gr.Textbox(
                        label="Директория для результатов",
                        value="./exp",
                        placeholder="Путь для сохранения моделей"
                    )
                    
                    experiment_description = gr.Textbox(
                        label="Описание эксперимента",
                        value="Training OCR model",
                        lines=2
                    )
                    
                    check_dataset_btn = gr.Button("Проверить датасет", size="sm")
                    dataset_info = gr.Markdown("")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Параметры обучения")
                    
                    with gr.Row():
                        image_w = gr.Number(label="Ширина изображения", value=256, precision=0)
                        image_h = gr.Number(label="Высота изображения", value=32, precision=0)
                    
                    with gr.Row():
                        num_epochs = gr.Number(label="Количество эпох", value=200, precision=0)
                        batch_size = gr.Number(label="Batch size", value=256, precision=0)
                    
                    with gr.Row():
                        num_workers = gr.Number(label="Num workers", value=8, precision=0)
                        seed = gr.Number(label="Random seed", value=6955, precision=0)
                    
                    gr.Markdown("### Аугментации")
                    
                    with gr.Row():
                        use_augs = gr.Checkbox(label="Использовать аугментации", value=True)
                        use_blot = gr.Checkbox(label="Использовать Blot", value=False)
                        use_stackmix = gr.Checkbox(label="Использовать StackMix", value=False)
                    
                    gr.Markdown("### Дополнительные параметры")
                    
                    use_pretrained_backbone = gr.Checkbox(
                        label="Pretrained backbone", 
                        value=True
                    )
                    
                    checkpoint_path = gr.Textbox(
                        label="Checkpoint для продолжения (опционально)",
                        value="",
                        placeholder="Оставьте пустым для нового обучения"
                    )
            
            with gr.Accordion("Neptune AI (опционально)", open=False):
                with gr.Row():
                    neptune_project = gr.Textbox(
                        label="Neptune Project",
                        placeholder="username/project-name",
                        value=""
                    )
                    neptune_token = gr.Textbox(
                        label="Neptune API Token",
                        type="password",
                        value=""
                    )
            
            with gr.Accordion("StackMix параметры (опционально)", open=False):
                mwe_tokens_dir = gr.Textbox(
                    label="Директория MWE tokens",
                    value="",
                    placeholder="Путь к токенам для StackMix"
                )
            
            gr.Markdown("---")
            
            with gr.Row():
                start_btn = gr.Button("Запустить обучение", variant="primary", size="lg")
                stop_btn = gr.Button("Остановить обучение", variant="stop", size="lg")

            with gr.Row():
                tb_port = gr.Number(label="TensorBoard port", value=6006, precision=0)
                start_tb_btn = gr.Button("Запустить TensorBoard вручную", size="sm")
                stop_tb_btn = gr.Button("Остановить TensorBoard", size="sm")
                tb_status = gr.Textbox(label="TensorBoard status", interactive=False, lines=1)
            
            status_msg = gr.Textbox(label="Статус", interactive=False, lines=3)

        
        # ========== ВКЛАДКА: ГЕНЕРАЦИЯ ==========
        with gr.Tab("Генерация данных"):
            gr.Markdown("""
            ### Генерация синтетических данных с помощью StackMix
            
            **Процесс генерации в три шага:**
            
            1. **Подготовить маски символов** - извлекает маски из обученной модели → `all_char_masks.json`
            2. **Создать токены** - нарезает изображения на токены (n-граммы символов)
            3. **Сгенерировать изображения** - комбинирует токены по текстам из корпуса
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Параметры данных")
                    gen_checkpoint_path = gr.Textbox(
                        label="Путь к checkpoint модели",
                        placeholder="exp/my_experiment/best_cer.pt"
                    )
                    
                    gen_data_dir = gr.Textbox(
                        label="Директория с данными",
                        value="./data"
                    )
                    
                    gen_marking_csv = gr.Textbox(
                        label="Путь к marking.csv",
                        value="./data/marking.csv"
                    )
                    
                    gen_text_file = gr.Textbox(
                        label="Файл с текстом для генерации (txt)",
                        placeholder="path/to/text_corpus.txt",
                        info="Текстовый файл, одна строка = одно предложение"
                    )
                    
                with gr.Column():
                    gr.Markdown("#### Параметры генерации")
                    gen_image_w = gr.Number(label="Ширина изображения", value=512, precision=0)
                    gen_image_h = gr.Number(label="Высота изображения", value=64, precision=0)
                    gen_batch_size = gr.Number(label="Batch size (для масок)", value=128, precision=0)
                    gen_num_workers = gr.Number(label="Num workers", value=8, precision=0)
                    gen_num_samples = gr.Number(label="Количество изображений", value=1000, precision=0)
                    
                    gen_tokens_dir = gr.Textbox(
                        label="Директория для токенов (Шаг 2)",
                        value="./stackmix_tokens",
                        info="Путь где будут сохранены токены"
                    )
                    
                    gen_output_dir = gr.Textbox(
                        label="Директория для изображений (Шаг 3)",
                        value="./generated_data",
                        info="Путь где будут сохранены сгенерированные изображения"
                    )
            
            gr.Markdown("---")
            
            with gr.Row():
                prepare_masks_btn = gr.Button("1️⃣ Подготовить маски", variant="primary", size="lg")
                prepare_tokens_btn = gr.Button("2️⃣ Создать токены", variant="primary", size="lg")
                generate_images_btn = gr.Button("3️⃣ Сгенерировать изображения", variant="secondary", size="lg")
            
            gen_status = gr.Textbox(label="Статус генерации", interactive=False, lines=10)
    
    # Привязка обработчиков - ОБУЧЕНИЕ
    check_dataset_btn.click(
        fn=check_dataset,
        inputs=[data_dir, marking_csv_path],
        outputs=dataset_info
    )
    
    start_btn.click(
        fn=handle_start_training,
        inputs=[
            data_dir, marking_csv_path, experiment_name, output_dir,
            experiment_description, image_w, image_h, num_epochs, batch_size,
            num_workers, use_blot, use_augs, use_stackmix,
            use_pretrained_backbone, seed, checkpoint_path, neptune_project,
            neptune_token, mwe_tokens_dir
        ],
        outputs=status_msg
    )
    
    stop_btn.click(
        fn=stop_training,
        inputs=None,
        outputs=status_msg
    )

    # TensorBoard controls
    start_tb_btn.click(
        fn=start_tb_ui,
        inputs=[output_dir, experiment_name, tb_port],
        outputs=tb_status
    )

    stop_tb_btn.click(
        fn=stop_tb_ui,
        inputs=None,
        outputs=tb_status
    )
    
    # Обработчики для генерации
    prepare_masks_btn.click(
        fn=prepare_char_masks,
        inputs=[
            gen_checkpoint_path, gen_data_dir, gen_marking_csv,
            gen_image_w, gen_image_h, gen_batch_size, gen_num_workers
        ],
        outputs=gen_status
    )
    
    prepare_tokens_btn.click(
        fn=prepare_stackmix_tokens,
        inputs=[
            gen_data_dir, gen_marking_csv,
            gen_image_h, gen_tokens_dir, gen_num_workers
        ],
        outputs=gen_status
    )
    
    generate_images_btn.click(
        fn=generate_images_from_corpus,
        inputs=[
            gen_tokens_dir, gen_data_dir, gen_marking_csv, gen_text_file,
            gen_image_h, gen_output_dir, gen_num_samples
        ],
        outputs=gen_status
    )


if __name__ == "__main__":
    print("Запуск StackMix OCR Web Interface...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )

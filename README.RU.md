# Детекция автомобилей с БПЛА с использованием модели YOLO

## 1. Постановка задачи

### Цель проекта проекта

Обучить модель машинного обучения для детекции легковых автомобилей на борту БПЛА в режиме реального времени

<br>

### Задача модели мошинного обучения

Детекция одного класса: Легковые автомобили

<br>

### Требования предъявляемые к модели машинного обучения

**1. Точность:** Обнаруживать легковые автомобили с точностью не менее 90%;

**2. Скорость:** Работать в режиме реального времени;

**3. Ресурсы:** Модель может работать на оборудовании доступном на борту БПЛА. Максимальная вычеслительная мощность при минимальном весе. Например:Jetson Nano или Jetson Xavier NX;

**4. Энергопотребление:** Для минимального сокращения времени полета БПЛА необходимо минимизировать энергопотребление. (Jetson Nano: 5-10 Вт, Jetson Xavier NX: 10-20 Вт).

## 2. Создание окружения

   ### Установка virtualenv (если не установлено)
   ```python
   pip install virtualenv
   ```
   ### Создание виртуального окружения
   ```python
   virtualenv yolov12_env
   ```
## 3. Активация окружения

   ### Активация виртуального окружения
   #### На Windows
   ```python
   yolov12_env\Scripts\activate
   ```

   #### На macOS/Linux
   ```python
   source yolov12_env/bin/activate
   ```
   ```python
   source yolo_env/bin/activate
   ```
## 4. Установка ultralytics
   ```python
   pip install ultralytics
   ```
при возникновении ошибок можно скачать и установить пакет отдельно. \
Например: 
```python
pip install /home/user/Загрузки/torch-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl
```


## 5. Контроль состоянтя GPU NVIDIA во время обучения модели
```bash
nvidia-smi
```
полезны следующие данные:
1. **GPU-Util** - на сколько процентов загружен GPU (при загрузке менее 50% можно увеличивать батч)
2. **Memory-Usage** - количество используемой/выделенной видеопамяти для обучения
3. **Temp** - температура GPU (при температуре выше 80 градусов нужно проверить систему охлаждения, т.к. высокая температура приведет к снижению производительности GPU)

## 6. Tmux
**tmux** - позволяет запускать процессы в терминале, которые продолжат работать даже после того, как вы выйдете из сеанса терминала.
### 1. Установка tmux
```bash
sudo apt update
sudo apt install tmux
```

### 2. Запуск tmux
```bash
tmux
```

### 3. Отключится от сессии tmux
```bash
Ctrl + b, затем d
```
### 4. Вернуться в seанс tmux:
```bash
tmux attach
```
### 5. Просмотреть список запущенных сеансов tmux
```bash
tmux ls
```
### 6. Подключиться к конкретному сеансу, указав его ID
```bash
tmux attach -t <имя_или_id_сеанса>
```
### 7. Завершение сеанса по имени сессии
```bash

# Инструкция по Pruning (удалению связей) для YOLOv8n от Ultralytics

Pruning (обрезка) нейронной сети - это процесс удаления наименее важных весов или целых нейронов для уменьшения размера модели и ускорения её работы при сохранении приемлемой точности.

## Подготовка

1. Установите необходимые зависимости:
```bash
pip install ultralytics torch torchvision torch-pruner
```

2. Убедитесь, что у вас установлена последняя версия Ultralytics:
```bash
pip install --upgrade ultralytics
```

## Процесс Pruning для YOLOv8n

### 1. Загрузка предобученной модели

```python
from ultralytics import YOLO

# Загрузка предобученной модели YOLOv8n
model = YOLO('yolov8n.pt')
model.train(data='coco128.yaml', epochs=100)  # Опционально: дообучение перед pruning
```

### 2. Реализация Pruning с помощью Torch

```python
import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

# Загрузка модели
model = YOLO('yolov8n.pt').model
model.train()  # Переводим модель в режим обучения

# Выбираем слои для pruning (например, все сверточные слои)
parameters_to_prune = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        parameters_to_prune.append((module, 'weight'))

# Применяем L1 unstructured pruning к выбранным слоям
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,  # Удаляем 30% весов (можно регулировать)
)

# Удаляем маски pruning, делая изменения постоянными
for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

# Сохраняем обрезанную модель
torch.save(model.state_dict(), 'yolov8n_pruned.pt')
```

### 3. Пост-обработка и дообучение

После pruning важно дообучить модель:

```python
# Создаем новую модель YOLO с обрезанными весами
pruned_model = YOLO('yolov8n.yaml').load('yolov8n_pruned.pt')

# Дообучение обрезанной модели
results = pruned_model.train(
    data='coco128.yaml',
    epochs=50,  # Обычно требуется меньше эпох для дообучения
    imgsz=640,
    lr0=0.01,  # Можно уменьшить learning rate
    patience=10
)
```

### 4. Оценка результатов pruning

```python
# Оценка оригинальной модели
original_model = YOLO('yolov8n.pt')
metrics_original = original_model.val()

# Оценка обрезанной модели
metrics_pruned = pruned_model.val()

print(f"Original mAP: {metrics_original.box.map}")
print(f"Pruned mAP: {metrics_pruned.box.map}")
```

## Альтернативный подход с использованием специализированных библиотек

### Использование Torch-Pruner

```python
from ultralytics import YOLO
from torch_pruner import SparsityPruner

# Загрузка модели
model = YOLO('yolov8n.pt').model

# Конфигурация pruner
config = {
    'pruning_method': 'l1',
    'target_sparsity': 0.3,  # Желаемый уровень спарсности
    'pruning_scope': 'global',
    'update_frequency': 1000
}

# Инициализация pruner
pruner = SparsityPruner(model, config)

# Процесс pruning (можно интегрировать в тренировочный цикл)
for epoch in range(10):
    for batch in dataloader:
        # Обычный forward/backward pass
        loss = model(batch)
        loss.backward()
        
        # Шаг pruning
        pruner.step()
        
        # Оптимизация весов
        optimizer.step()
        optimizer.zero_grad()

# Сохранение обрезанной модели
torch.save(model.state_dict(), 'yolov8n_pruned_torchpruner.pt')
```

## Рекомендации по Pruning

1. **Стратегия pruning**:
   - Начните с небольшого уровня pruning (10-20%)
   - Постепенно увеличивайте уровень, проверяя качество модели
   - Для YOLO лучше работает структурированный pruning (целых фильтров)

2. **Выбор слоев**:
   - Концентрируйтесь на больших сверточных слоях
   - Избегайте pruning последних слоев перед детекцией

3. **Дообучение**:
   - Всегда дообучайте модель после pruning
   - Используйте меньший learning rate для дообучения
   - Увеличьте количество эпох дообучения при высоком уровне pruning

4. **Оценка**:
   - Сравнивайте не только точность, но и скорость inference
   - Проверяйте модель на репрезентативных данных

## Визуализация результатов Pruning

```python
import numpy as np
import matplotlib.pyplot as plt

# Функция для анализа спарсности
def analyze_sparsity(model):
    sparsities = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            sparsity = float(torch.sum(module.weight == 0)) / float(module.weight.nelement())
            sparsities.append(sparsity)
            print(f"Слой {name}: спарсность {sparsity:.2%}")
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(sparsities)), sparsities)
    plt.title("Уровень спарсности по слоям")
    plt.ylabel("Спарсность")
    plt.xlabel("Номер слоя")
    plt.show()

analyze_sparsity(model)
```

## Экспорт обрезанной модели

После pruning и дообучения можно экспортировать модель:

```python
pruned_model.export(format='onnx')  # Экспорт в ONNX
pruned_model.export(format='torchscript')  # Экспорт в TorchScript
```

## Заключение

Pruning YOLOv8n может значительно уменьшить размер модели и ускорить её работу с минимальной потерей точности при правильном подходе. Рекомендуется экспериментировать с разными уровнями и типами pruning, чтобы найти оптимальный баланс между производительностью и точностью для вашего конкретного случая использования.






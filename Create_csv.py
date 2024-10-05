import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Загрузка модели
model = load_model('vgg16_cats_and_dogs.keras')  # Укажите путь к вашей модели

# Параметры изображений
img_width, img_height = 150, 150
batch_size = 1  # Установим batch_size на 1, чтобы обрабатывать по одному изображению

# Пути к тестовым данным
test_data_dir = 'ready_photos/test'

# Генератор данных для тестирования
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # Важно, чтобы не перемешивать данные, так как нам нужно сопоставление с именами файлов
)

# Получаем предсказания
predictions = model.predict(test_generator, steps=test_generator.samples)

# Преобразуем предсказания в метки классов (0 или 1)
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]

# Реальные метки классов
true_classes = test_generator.classes

# Имена файлов
filenames = test_generator.filenames

# Создаем DataFrame для результата
results_df = pd.DataFrame({
    'Filename': filenames,
    'True Class': true_classes,
    'True Label': ['Собака' if cls == 1 else 'Кот' for cls in true_classes],  # Текстовые метки
    'Predicted Class': predicted_classes,
    'Predicted Label': ['Собака' if pred == 1 else 'Кот' for pred in predicted_classes],  # Текстовые метки
    'Correct': [true == pred for true, pred in zip(true_classes, predicted_classes)]  # Правильность предсказания
})

# Сохраняем результат в CSV файл
csv_output_path = 'test_predictions_report.csv'
results_df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')  # encoding для корректного отображения кириллицы

print(f"Результаты сохранены в файл: {csv_output_path}")

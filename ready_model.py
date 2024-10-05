import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers, optimizers

# Устанавливаем пути к данным
train_data_dir = 'ready_photos/train'
validation_data_dir = 'ready_photos/val'
test_data_dir = 'ready_photos/test'

# Параметры изображения и модели
img_width, img_height = 150, 150  # Размеры изображений
batch_size = 16  # Размер пакета для обучения
epochs = 25  # Количество эпох для обучения

nb_train_samples = 250  # Количество тренировочных данных
nb_validation_samples = 50  # Количество валидационных данных
nb_test_samples = 50  # Количество тестовых данных

# Используем предобученную модель VGG16, без верхних полносвязных слоев
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Замораживаем сверточные слои, чтобы не изменять их при обучении
conv_base.trainable = False

# Строим модель
model = models.Sequential()
model.add(conv_base)  # Добавляем предобученную VGG16 в нашу модель
model.add(layers.Flatten())  # Разворачиваем данные для подачи в полносвязные слои
model.add(layers.Dense(256, activation='relu'))  # Добавляем полносвязный слой
model.add(layers.Dropout(0.5))  # Dropout для предотвращения переобучения
model.add(layers.Dense(1, activation='sigmoid'))  # Выходной слой для бинарной классификации

# Компиляция модели
model.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Генераторы данных
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Генератор для тренировочных данных
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Генератор для валидационных данных
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Генератор для тестовых данных
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Обучение модели
history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(test_generator, steps=nb_test_samples // batch_size)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Сохраняем модель
model.save('vgg16_cats_and_dogs.keras')

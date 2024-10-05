from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Загрузка модели
model = load_model('vgg16_cats_and_dogs.keras')  # Укажите путь к вашему файлу модели

# Параметры изображений
img_width, img_height = 150, 150
batch_size = 16

# Пути к данным
test_data_dir = 'ready_photos/test'

# Генератор данных для тестирования
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f"Test accuracy: {test_acc * 100:.2f}%")

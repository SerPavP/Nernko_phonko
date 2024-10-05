from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model('vgg16_cats_and_dogs.keras')

# Проверка структуры модели
model.summary()

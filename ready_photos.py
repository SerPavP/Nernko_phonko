import shutil
import os

# Устанавливаем основные директории
data_dir_cats = 'cats_dogs\\train\\cats'
data_dir_dogs = 'cats_dogs\\train\\dogs'
# Главная директория для готовых данных
ready_photos_dir = 'ready_photos'
# Каталоги внутри 'ready_photos'
train_dir = os.path.join(ready_photos_dir, 'train')
val_dir = os.path.join(ready_photos_dir, 'val')
test_dir = os.path.join(ready_photos_dir, 'test')

test_data_portion = 0.15
val_data_portion = 0.15
nb_images = 200

# Функция для создания директории
def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)  # Удаляем старую директорию, если она существует
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name, "cats"))
    os.makedirs(os.path.join(dir_name, "dogs"))

# Создаем главную директорию 'ready_photos' и директории внутри неё
if not os.path.exists(ready_photos_dir):
    os.makedirs(ready_photos_dir)

create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)

# Функция для копирования изображений
def copy_images(start_index, end_index, source_dir_cats, source_dir_dogs, dest_dir):
    for i in range(start_index, end_index):
        cat_image_path = os.path.join(source_dir_cats, "cat." + str(i) + ".jpg")
        dog_image_path = os.path.join(source_dir_dogs, "dog." + str(i) + ".jpg")

        # Проверяем, существует ли изображение кота
        if os.path.exists(cat_image_path):
            shutil.copy2(cat_image_path, os.path.join(dest_dir, "cats"))
        else:
            print(f"Warning: {cat_image_path} not found.")

        # Проверяем, существует ли изображение собаки
        if os.path.exists(dog_image_path):
            shutil.copy2(dog_image_path, os.path.join(dest_dir, "dogs"))
        else:
            print(f"Warning: {dog_image_path} not found.")


start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))
start_test_data_idx = int(nb_images * (1 - test_data_portion))
print("Train data index end:", start_val_data_idx)
print("Validation data index end:", start_test_data_idx)     

# Копируем изображения в разные директории внутри 'ready_photos'
copy_images(0, start_val_data_idx, data_dir_cats, data_dir_dogs, train_dir)
copy_images(start_val_data_idx, start_test_data_idx, data_dir_cats, data_dir_dogs, val_dir)
copy_images(start_test_data_idx, nb_images, data_dir_cats, data_dir_dogs, test_dir)

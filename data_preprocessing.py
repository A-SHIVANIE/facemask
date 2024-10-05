import cv2
import os
import numpy as np
from keras.utils import to_categorical  # Updated import statement

# Path to your dataset
data_path = 'dataset'  # Replace with the actual path to your dataset
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]

# Create a label dictionary
label_dict = dict(zip(categories, labels))

img_size = 100
data = []
target = []

# Preprocess each image in the dataset
for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (img_size, img_size))
            data.append(resized)
            target.append(label_dict[category])
        except Exception as e:
            print(f'Exception: {e}')

# Convert lists to NumPy arrays
data = np.array(data) / 255.0
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
target = np.array(target)

# Convert labels to categorical using to_categorical
new_target = to_categorical(target, num_classes=len(categories))  # Specify the number of classes

# Save preprocessed data
np.save('data.npy', data)
np.save('target.npy', new_target)

print("Data preprocessing completed and saved successfully.")






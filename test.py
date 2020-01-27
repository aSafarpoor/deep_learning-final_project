import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import cv2
path = "/home/amir/Code/bama_ir/dataset/CLEANED_TEST/pics/"
MPATH = "/home/amir/Code/bama_ir/temp/"

input_shape = (225, 150, 3)

train_datagen = ImageDataGenerator(rescale=1./255)
DG = train_datagen.flow_from_directory(
    path,
    target_size=input_shape[:-1],
    batch_size=1)

model = load_model(MPATH+"model_main.h5")

img = cv2.imread(
    path+"/پژو-206/CarImage8376360_637092586498969802_0_thumb_450_300.jpg")
img = cv2.resize(img, (150, 225))
result = model.predict(np.array([img]))

index = np.argmax(result)

for i in DG.class_indices:
    if DG.class_indices[i] == index:
        print(i)
        break
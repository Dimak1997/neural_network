from PIL import Image
import matplotlib as plt
import numpy as np
import os
import timeit
import keras
import sys
from keras.models import Sequential
from keras.models import save_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from os.path import basename
from sklearn.model_selection import train_test_split
import pickle

#paths to find train and test data

epochs = 20
path = '/home/andrey/PycharmProjects/flowers/venv/model.hf5'

#Установка параметров для загрузки скрипта
#epochs = 2
#path = '/home/dmitriy/PycharmProjects/untitled/venv/model.hf5'

#Директории изображений
train_Images_1_dir = '/home/andrey/PycharmProjects/flowers/flowers/daisy/'
train_Images_2_dir = '/home/andrey/PycharmProjects/flowers/flowers/dandelion/'
train_Images_3_dir = '/home/andrey/PycharmProjects/flowers/flowers/rose/'
train_Images_4_dir = '/home/andrey/PycharmProjects/flowers/flowers/sunflower/'
train_Images_5_dir = '/home/andrey/PycharmProjects/flowers/flowers/tulip/'

#Получение количества изображений для формирования выборки
train_Images_1_sample = len(os.listdir(train_Images_1_dir))
train_Images_2_sample = len(os.listdir(train_Images_2_dir))
train_Images_3_sample = len(os.listdir(train_Images_3_dir))
train_Images_4_sample = len(os.listdir(train_Images_4_dir))
train_Images_5_sample = len(os.listdir(train_Images_5_dir))

#Пустой массив для изображений
images_to_analyze=np.zeros([4323,100,100,3])

#Пустой массив меток класса
labels_to_analyze = np.array([])

#Загрузка изображений класса 0
for index,filename in enumerate(os.listdir(train_Images_1_dir)):
    img = Image.open(train_Images_1_dir+filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    images_to_analyze[index, :, :, :] = im
    labels_to_analyze = np.append(labels_to_analyze, 0)

#Загрузка изображений класса 1
for index2,filename in enumerate(os.listdir(train_Images_2_dir)):
    img = Image.open(train_Images_2_dir+filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    images_to_analyze[index+index2, :, :, :] = im
    labels_to_analyze = np.append(labels_to_analyze, 1)

#Загрузка изображений класса 2
for index3,filename in enumerate(os.listdir(train_Images_3_dir)):
    img = Image.open(train_Images_3_dir+filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    images_to_analyze[index+index2+index3, :, :, :] = im
    labels_to_analyze = np.append(labels_to_analyze, 2)

#Загрузка изображений класса 3
for index4,filename in enumerate(os.listdir(train_Images_4_dir)):
    img = Image.open(train_Images_4_dir+filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    images_to_analyze[index+index2+index3+index4, :, :, :] = im
    labels_to_analyze = np.append(labels_to_analyze, 3)

#Загрузка изображений класса 4
for index4,filename in enumerate(os.listdir(train_Images_5_dir)):
    img = Image.open(train_Images_5_dir+filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    images_to_analyze[index+index2+index3+index4, :, :, :] = im
    labels_to_analyze = np.append(labels_to_analyze, 3)

#Нормализация изображений
images_to_analyze = images_to_analyze/255

#Разделение изображений на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(images_to_analyze, labels_to_analyze, test_size=0.1)


y_train = keras.utils.to_categorical(y_train, num_classes=5)
y_test = keras.utils.to_categorical(y_test, num_classes=5)

#Запись в файл тестовой выборки
f = open(r'file.txt', 'wb')
obj = X_test
obj2 = y_test
pickle.dump(obj, f)
pickle.dump(obj2, f)
f.close()

#Модель свёрточной нейронной сети
model = Sequential()
model.add(Conv2D(8,(3,3),input_shape=(100,100,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(12,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=512,activation='relu'))
model.add(Dense(units=5,activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=int(epochs), shuffle=True)

#score = model.evaluate(X_test, y_test)
#print(score)
#model.save("path/model.h5")

#Сохранение модели в файл
save_model(model, path, overwrite=True, include_optimizer=True)
print("Model successfully saved")

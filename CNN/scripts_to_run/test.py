from keras.models import model_from_json
from keras.models import load_model
import cv2
import numpy as np
import pickle

#Загрузка тестовой выборки из файла
f = open(r'file.txt', 'rb')
X_test = pickle.load(f)
y_test = pickle.load(f)
f.close()

#Загрузка модели нейронной сети из файла
filepath = '/home/andrey/PycharmProjects/flowers/venv/model.hf5'
loaded_model = load_model(filepath,custom_objects=None,compile=True)

#Результат проверки на тестовой выборке
score = loaded_model.evaluate(X_test, y_test)
print(score)

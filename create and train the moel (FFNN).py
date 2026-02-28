import os
import pandas as pd
import numpy as np

folder="/home/abdelali/Downloads/sign_language_landmarks"

coordinates=[]
labels=[]

for files in os.listdir(folder):
    df=pd.read_csv(os.path.join(folder,files),header=None)
    for row in df.values:
        coordinates.append(row[:-1])
        labels.append(row[-1])

x=np.array(coordinates,dtype=np.float32)
y=np.array(labels)

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

le=LabelEncoder()
y_encoded=le.fit_transform(y)
y_categorical=to_categorical(y_encoded)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y_categorical,test_size=0.3,random_state=42)

from keras.models import Sequential
from keras.layers import Dense,Dropout

model=Sequential([
    Dense(128,activation='relu',input_shape=(63,)),
    Dropout(0.3),
    Dense(64,activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1],activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

history=model.fit(
    x_train,y_train,
    validation_data=(x_test,y_test),
    epochs=50,
    batch_size=32
)

loss, acc =model.evaluate(x_test,y_test)
print(f"Test accuracy: {acc*100:.2f}%")

model.save('asl_landmarks')
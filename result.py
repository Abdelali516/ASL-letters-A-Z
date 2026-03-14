import cv2
import mediapipe as mp
import numpy as np 
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time
import string
from personai import function

model=load_model("sign_language_model")
le=LabelEncoder()
le.fit(list(string.ascii_lowercase))

cap=cv2.VideoCapture(0)
hands=mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=0.7)
mp_draw=mp.solutions.drawing_utils

current_letter=None
current_time=None
last_hand_time=time.time()
typed=False
sentence=""

while cap.isOpened():

    now=time.time()

    ret,frame=cap.read()
    if not ret:
        print("Couldn't catch a frame !")
        break

    frame=cv2.flip(frame,1)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=hands.process(rgb)

    predicted_letter=None
    if result.multi_hand_landmarks:
        last_hand_time=now

        lm=result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame,lm,mp.solutions.hands.HAND_CONNECTIONS)

        features=np.array([[v for p in lm.landmark for v in (p.x,p.y,p.z)]],dtype=np.float32)
        predicted_letter=le.inverse_transform([np.argmax(model.predict(features,verbose=0))])[0]

    
    if predicted_letter!=current_letter :
        current_letter,current_time,typed=predicted_letter,now,False
    
    if current_letter and not typed and (now-current_time)>=1:
        sentence+=current_letter
        print('User:',sentence.title(),end="\r",flush=True)
        typed=True
    
    if sentence and (now-last_hand_time)>=0.5:
        print(" ",end="",flush=True)

    elif sentence and (now-last_hand_time)>=1:
        ai_model='llama3.1:8b'
        print("\nBot:",end="",flush=True)
        function(ai_model,sentence)
        sentence=""

    cv2.imshow("Video",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

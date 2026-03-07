import cv2
import mediapipe as mp
import csv 

file="(letter)_landmarks.csv" # replace "letter" by the letter you want to save it's landmarl

cap=cv2.VideoCapture(0)
hands=mp.solutions.hands.Hands()
mp_draw=mp.solutions.drawing_utils

while not cap.isOpened():
    print("Couldn't find a camera !")
    exit()

while True:
    labels=[]
    letter=input("Enter the correspending letter for the hand sign you will do:").lower().strip()
    if letter=="exit":
        break
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            print("Couldn't catch a frame !")
            break

        frame=cv2.flip(frame,1)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=hands.process(rgb)

        if res.multi_hand_landmarks:
            for HANDS in res.multi_hand_landmarks:
                row=[]
                for lm in HANDS.landmark:
                    row+=[lm.x,lm.y,lm.z]
                row.append(letter)
                labels.append(row)

                mp_draw.draw_landmarks(frame,HANDS,mp.solutions.hands.HAND_CONNECTIONS)
    
        cv2.imshow("video",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    with open(file,"a") as f:
        writer=csv.writer(f)
        writer.writerows(labels)

cap.release()
cv2.destroyAllWindows()
hands.close
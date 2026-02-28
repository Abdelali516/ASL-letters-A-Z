import cv2
import mediapipe as mp
import csv

cap=cv2.VideoCapture(0)
hands=mp.solutions.hands.Hands()
mp_draw=mp.solutions.drawing_utils

while not cap.isOpened():
    print("Sorry couldn't find a camera !")
    exit()

while True:
    labels_landmarks=[]
    letter=input("Enter the letter for which you want to display the hand sign:").lower().strip()
    if letter == 'exit':
        break
    
    while True:
        ret,frame=cap.read()
        if not ret:
            print("Couldn't catch a frame:")
            break

        frame=cv2.flip(frame,1)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=hands.process(rgb)

        if result.multi_hand_landmarks:
            for HANDS in result.multi_hand_landmarks:
                row=[]
                for lm in HANDS.landmark:
                    row+=[lm.x,lm.y,lm.z]
                row.append(letter)
                labels_landmarks.append(row)

                mp_draw.draw_landmarks(frame,HANDS,mp.solutions.hands.HAND_CONNECTIONS)
        
        cv2.imshow("Capture letters landmarks:",frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    with open("a.csv",'a',newline="") as file: # and i keep doing the same thing for all the letters 
        
        writer=csv.writer(file)
        writer.writerows(labels_landmarks)

cap.release
cv2.destroyAllWindows()
hands.close()
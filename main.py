import cv2 as cv
import time
import mediapipe as mp



cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    ret, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
       for handLms in results.multi_hand_landmarks:
           for id, lm in enumerate(handLms.landmark):
               h, w, c = img.shape
               cx, cy = int(lm.x*w), int(lm.y*h)
               print(id, cx, cy)
               
           mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
    cv.imshow('Cam', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv.putText(img, str(int(fps)), (10, 70),
               cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 6)

cap.release()
cv.destroyAllWindows()
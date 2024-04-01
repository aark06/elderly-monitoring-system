import cv2
import time
from cvzone.HandTrackingModule import HandDetector
import streamlit as st


def alertFingers(): 
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)


    start_time = 0
    alert_printed = False

    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)  # With Draw

        if hands:
            for hand in hands: 
                fingers = detector.fingersUp(hand)
                if all(fingers): 
                    
                    if start_time == 0:
                        start_time = time.time()
                    elif (time.time() - start_time) >= 5 and not alert_printed:
                        st.write("ALERT Fingers!")
                        alert_printed = True
                        break
                else: 
                    alert_printed = False
                    start_time = 0


        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
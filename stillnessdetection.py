import cv2
import time
import streamlit as st

def stillnessDetection():
    cap = cv2.VideoCapture(0)  
    ret, frame1 = cap.read()  
    last = time.time()
    flag = False
    while True:
        ret, frame2 = cap.read()  
        # obtaining absolute difference between two consecutive frames
        diff = cv2.absdiff(frame1, frame2) 
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  
        # gaussian blur to remove noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  
        ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)  
        dilated = cv2.dilate(thresh, None, iterations=3)  
        contours, ret = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
        
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            # ignoring small contours
            if cv2.contourArea(contour) < 700:
                continue  
            last = time.time()
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)  
            flag = True
            last = time.time()
        
        # print an alert message if no motion is detected for over 15 seconds
        if time.time() - last > 15:
            print('alert')
            st.markdown("<h1 style='text-align: center; color: red;'>STILLNESS ALERT!</h1>", unsafe_allow_html=True)
            break
        cv2.imshow("Motion Detection", frame1)  
        frame1 = frame2  
        ret, frame2 = cap.read()  
        
        flag = False

        key = cv2.waitKey(1)  
        if key == ord('q'):  
            break

    cap.release()  
    cv2.destroyAllWindows()  
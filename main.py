import face_recognition
import cv2
import time
import playsound 
import numpy as np
from threading import Thread, main_thread
from scipy.spatial import distance as dist


MIN_AER= 0.30
EYE_AR_COSEC_FRAMES=10
COUNTER= 0
ALARM_ON=False
def sound_alarm(soundfile):
    playsound.playsound(soundfile)
    

def eye_aspect_ratio(eye):
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[5])
    C=dist.euclidean(eye[0],eye[3])
    ear=(A+B)/(2*C)
    return ear

def main():
    global COUNTER, ALARM_ON
    video_capture = cv2.video_capture(0)
    video_capture.set(3,320)
    video_capture.set(3,320)
    while True:
        ret,frame= video_capture.read0
        face_landmarks_list= face_recognition.face_landmarks(frame)
        for face_landmark in face_landmarks_list:
            lefteye = face_landmark['left_eye']
            righteye = face_landmark['right_eye']

            
            leftEAR = eye_aspect_ratio(lefteye)
            righttEAR = eye_aspect_ratio(righteye)
            ear = (leftEAR+righttEAR)/2

            lpts=np.array(lefteye)
            rpts=np.array(righteye)

            cv2.polylines(frame,[lpts],True,(255,255,0),1)
            cv2.polylines(frame,[rpts],True,(255,255,0),1)
            
            if ear<MIN_AER:
                COUNTER+=1
                if COUNTER>EYE_AR_COSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON=True
                        t=Thread(target=sound_alarm, args=('filename'))
                        t.daemon=True
                        t.start()


                cv2.putText(frame,"Alert! you are feeling sleepy",(5,10),
                 cv2.FONT_HERSHEY_SIMPLEX,0.4(0,0,255),1)
            else:
                COUNTER=0
                ALARM_ON=False 
            cv2.putText(frame,"ear.2f".format(ear),(300,10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
            cv2.imshow("sleep detection", frame)        
            if cv2.waitKey(1)==ord('q'):
                break
            video_capture.release()
            cv2.destroyAllWindows()
        
        if __name__== "__main__":
            main()

        
              
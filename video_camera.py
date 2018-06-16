from __future__ import print_function, division
from facial_recognizer import facial_recognizer
import cv2
import datetime

class video_camera:
    def __init__(self, video_folder, face_decection):
        self.video_folder = video_folder
        self.face_decection = face_decection

        print("Opening Video Camera...")
        self.cap = cv2.VideoCapture(0)

        if(self.cap.isOpened()):
            print("Video Camera Opened Successfully")
        else:
            print("Error Opening Video Camera")

    def start_recording(self):
        frame_count = 0

        while(True):

            now = datetime.datetime.now()

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            filename = self.video_folder+"/"+str(now.month)+":"+str(now.day)+":"+str(now.year)+"-"+str(now.hour)+":"+str(now.minute)+":"+str(now.second)+'.avi'
            print("saving video: "+filename)
            out = cv2.VideoWriter(filename,fourcc, 20.0, (640,480))

            total = 0
            while(frame_count < 100):
                ret, frame = self.cap.read()

                out.write(frame)

                people = self.face_decection.identify(frame)
                for person in people:
                    print("{}: Found {}".format(total, person))
                total = total + 1

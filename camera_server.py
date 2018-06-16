from facial_recognizer import facial_recognizer
from video_camera import video_camera

if __name__ == "__main__":

	face_decection = facial_recognizer("input/classifiers/haarcascade_frontalface_default.xml", "input/svm_training_photos", "input/svm_training_photos_cleaned")
	video_camera = video_camera("output/saved_videos", face_decection)
	
	video_camera.start_recording()
	#print("Opening Video Camera...")
	#cap = cv2.VideoCapture(0)

	#fourcc = cv2.VideoWriter_fourcc(*'XVID')
	#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

	#ret, frame = cap.read()

	#if(cap.isOpened()):
	#	print("Video Camera Opened Successfully")
	#else:
	#	print("Error Opening Video Camera")

	#total = 0
	#while(True):
	#	ret, frame = cap.read()

	#	out.write(frame)

	#	people = face_decection.identify(frame)
	#	for person in people:
	#		print("{}: Found {}".format(total, person))
	#	total = total + 1

	#	if cv2.waitKey(1) & 0xFF == ord('q'):
	#		break

	# When everything done, release the capture
	#cap.release()
	#out.release()
	#cv2.destroyAllWindows()
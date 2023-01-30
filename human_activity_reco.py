# USAGE
# python human_activity_reco.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt --input example_activities.mp4
# python human_activity_reco.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt

import numpy as np
import argparse
import imutils
import sys
import cv2
from playsound import playsound
from playSiren import MyThread
import matplotlib.pyplot as plt
import math

model="resnet-34_kinetics.onnx"
classes="action_recognition_kinetics.txt"

# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to trained human activity recognition model")
# ap.add_argument("-c", "--classes", required=True,
# 	help="path to class labels file")
# ap.add_argument("-i", "--input", type=str, default="",
# 	help="optional path to video file")
# args = vars(ap.parse_args())
def humanFightAlgo(file):

	CLASSES = open(classes).read().strip().split("\n")
	print(CLASSES)
	SAMPLE_DURATION = 16
	SAMPLE_SIZE = 112

	print("[INFO] loading human activity recognition model...")
	net = cv2.dnn.readNet(model)

	print("[INFO] accessing video stream...")
	vs = cv2.VideoCapture(file if file else 0)
	count = 0
	noncount = 0
	allFrames = 0
	# VNVframes = 0
	siren = False
	frames25 = 25
	while True:

		frames = []

		for i in range(0, SAMPLE_DURATION):
			(grabbed, frame) = vs.read()

			if not grabbed:
				print("[INFO] no frame read from stream - exiting")
				vs.release()
				VNVFrames = noncount + count
				print("Total frames=", VNVFrames)
				print("Violence frame=", count)
				print("Non-Violence frame=", noncount)
				cv2.destroyAllWindows()
				viopct = (count / VNVFrames) * 100
				nonviopct = (noncount / VNVFrames) * 100
				plt.figure(figsize=(5, 20))
				colors_list = ['Red', 'Green']
				format = ["Violence " + str(round(viopct, 2)) + "%", "Non Violence " + str(round(nonviopct, 2)) + "%"]
				data = [viopct, nonviopct]
				graph = plt.bar(format, data, color=colors_list)
				plt.title('Percentage of Violence and Non Violence in Video')
				plt.show()
				sys.exit(0)

			frame = imutils.resize(frame, width=800)
			frames.append(frame)

		blob = cv2.dnn.blobFromImages(frames, 1.0,
									  (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
									  swapRB=True, crop=True)
		blob = np.transpose(blob, (1, 0, 2, 3))
		blob = np.expand_dims(blob, axis=0)
		# print(blob)

		net.setInput(blob)
		outputs = net.forward()
		label = CLASSES[np.argmax(outputs)]
		# print(label)
		voilenceClasses = (
			"punching person (boxing)", "wrestling", "arm wrestling",
			"squat", "deadlifting", "capoeira", "archery", "side kick"
														   "playing ice hockey")
		cv2.rectangle(frames[0], (0, 0), (300, 40), (0, 0, 0), -1)
		cv2.putText(frame, "", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
					0.8, (255, 255, 255), 2)

		cv2.imshow("Activity Recognition", frame)
		key = cv2.waitKey(1) & 0xFF

		for frame in frames:
			allFrames += 1
			for l in voilenceClasses:
				if (label == l):
					count += 1

					if (count == frames25 and siren == False):
						print(count, "Violence frames found\nSiren Alarm Played!!!")
						siren = True
						t1 = MyThread()
						t1.start()
					# input("press ENTER to stop playback")
					# p.terminate()
					# playsound('audio/siren.wav')
					# else:
					# 	siren=False
					# if(siren==True):

					# print("Violence Detected : "+str(count))
					# cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
					# cv2.putText(frame, str(count), (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
					# 			0.8, (255, 255, 255), 2)

					cv2.imshow("Activity Recognition", frame)
					key = cv2.waitKey(1) & 0xFF

					if key == ord("q"):
						break
				else:
					noncount += 1

# for frame in frames:
# 	cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
# 	cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
# 		0.8, (255, 255, 255), 2)
#
# 	cv2.imshow("Activity Recognition", frame)
# 	key = cv2.waitKey(1) & 0xFF
#
# 	if key == ord("q"):
# 		break

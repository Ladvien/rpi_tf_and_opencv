# Lots of thanks to  Adrian Rosebrock
# https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/

# TODO: Load prediction classes
# TODO: Make prediction in main loop.
# TODO: Show target box and full image


#################################
# Imports
#################################
import time
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import imutils
import numpy as np
import json

import tensorflow as tf

#################################
# Parameters
#################################
model_save_dir           = '/home/pi/lego_sorter/data/'
model_filepath           = model_save_dir + 'model.h5'
model_classes_filepath   = model_save_dir + 'classes.json'

grab_area                = 800          # Amount of the image to save (in pxs)
view_width               = 3280
view_height              = 2464

output_size              = (300, 300)   # Actual saved image size.

rotate_angle 			 = 180         
flip_code 				 = 0            # Mirrors image.
x_offset                 = 0         	# Nudges center position
y_offset                 = 0            # Nudges center position


show_sample_window       = True
gray_scale               = False

font = cv2.FONT_HERSHEY_COMPLEX
lineType                    = 2

instructions_font_color     = (0, 255, 0)
instructions_font_scale     = view_width * 0.001
instructions_position       = (int(view_width * 0.01), int(view_height * 0.95))

prediction_font_color      	= (0, 0, 255)
prediction_font_scale      	= view_width * 0.001
prediction_position         = (int(view_width * 0.01), int(view_height * 0.05))

#################################
# Setup Model
#################################
print('Loading model...')
model = tf.keras.models.load_model(model_filepath)

classes = dict()
with open(model_classes_filepath) as f:
    classes = json.load(f)

# Invert the lookup
classes = {v: k for k, v in classes.items()}
print('Finished.')
 
#################################
# Setup Webcam
#################################
print('Initializing camera')

camera = PiCamera()
print(f'PiCam resolution: {camera.MAX_RESOLUTION}')

average_prediction_times = []

try:
	if gray_scale:
		camera.color_effects = (128,128)
	camera.resolution = (view_width, view_height)
	camera.framerate = 12
	rawCapture = PiRGBArray(camera, size = (view_width, view_height)) # Width, Height

	# Camera warmup.
	time.sleep(0.1)

	cv2.namedWindow('lego_classifier')
	cv2.moveWindow('lego_classifier', 0, 0)

	print('Starting main loop.')

	# capture frames from the camera
	for frame in camera.capture_continuous(rawCapture, format = 'bgr', use_video_port = False):

		# Time processing.
		start_time = time.time()

		# grab the raw NumPy array representing the image, then initialize the timestamp
		# and occupied/unoccupied text
		image = frame.array
		original_image = image

		# Trim image to target size.
		image = image[150:1050, 150:1050]

		c_x, c_y = int(image.shape[0] / 2), int(image.shape[1] / 2)

		c_x += x_offset
		c_y += y_offset
		
		# Calculate area of interest from center of image.
		y_start = (c_y - int(grab_area / 2))
		y_end = c_y + int(grab_area / 2)
		x_start = (c_x - int(grab_area / 2))
		x_end = c_x + int(grab_area / 2)
		
		# Crop the image.
		image = image[y_start: y_end, x_start: x_end]

		image = imutils.rotate(image, rotate_angle)
		image = cv2.flip(image, flip_code)
		image = cv2.resize(image, (output_size[0], output_size[1]), interpolation = cv2.INTER_AREA)

		# Convert to grayscale if needed.
		if gray_scale:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			image = image.reshape([output_size[0], output_size[0], 1])

		# Display the resulting frame
		classifier_frame = np.expand_dims(image, axis=0)
		predicted_class = model.predict_classes(classifier_frame)[0]
		human_readable_class = classes[predicted_class]

		# Display instructions.
		cv2.putText(original_image,'"q" to quit', 
				instructions_position, 
				font, 
				instructions_font_scale,
				instructions_font_color,
				lineType,
				cv2.LINE_AA)

		# Display prediction
		cv2.putText(original_image, 
				human_readable_class, 
				prediction_position, 
				font, 
				prediction_font_scale,
				prediction_font_color,
				lineType,
				cv2.LINE_AA)

		resized_image = cv2.resize(original_image, (int(0.2 * view_width), int(0.2 * view_height))) 

		# show the frame
		cv2.imshow('lego_classifier', resized_image)
		key = cv2.waitKey(1) & 0xFF

		# clear the stream in preparation for the next frame
		rawCapture.truncate(0)

		# Capture and prediction time.
		prediction_time = round(time.time() - start_time, 6)
		average_prediction_times.append(prediction_time)
		print(f'Prediction time: {prediction_time}s')

		# if the `q` key was pressed, break from the loop
		if key == ord('q'):
			camera.close()
			break
		
finally:
    average_prediction_time = round(np.sum(average_prediction_times) / len(average_prediction_times), 6)
    print('Quitting.')
    print(f'Average prediction time: {average_prediction_time}')
    camera.close()


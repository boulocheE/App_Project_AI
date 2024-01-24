import numpy             as np
import matplotlib.pyplot as plt

from keras.models import load_model

import matplotlib
import cv2

CATEGORIES = ['no turn', 'speed limit', 'access forbiden', 'no way', 'no parking', 'other']

pwd = '/home/antoine/Prog/AI_Portfolio/'

model_dir   = pwd + 'models/'
img_dir     = pwd + '../../img/'
upload_path = pwd + 'static/images/'
image_path  = pwd + 'uploads/in.png'


# Load shape model for locating circular shapes
shape_recognizer = load_model( model_dir + 'shape-recognizerv3-30eh.h5' )

# Load sign model for classifying signs
sign_model       = load_model( model_dir + 'best_model.h5'         )


MIN_PANNEL_RATIO  = .5
MAX_PANNEL_RATIO  =  2
MIN_AREA_OCCUPIED = .0004
MIN_AREA_PIXEL    =  6**2


# Chat-GPT
def edge_detection(img_bgr):
	"""
	Applies edge detection to an image using Sobel filters.

	Parameters:
	img_bgr: numpy.ndarray
		The input image in BGR format.

	Returns:
	numpy.ndarray
		The edge-detected image.
	"""

	img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


	# Apply the Sobel filters for horizontal and vertical edge detection
	sobel_x = cv2.Sobel(img[:,:,1], cv2.CV_64F, 1, 0, ksize=3)
	sobel_y = cv2.Sobel(img[:,:,1], cv2.CV_64F, 0, 1, ksize=3)


	# Calculate the magnitude of the gradient
	edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

	# Normalize the result to an 8-bit scale
	edge_magnitude = cv2.normalize(edge_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


	return edge_magnitude

def zoom_image(image, zoom_factor=1.0):
    """
    Zooms an image using the specified zoom factor.
    
    :param image: A numpy array representing the image to be zoomed.
    :param zoom_factor: A float representing the zoom factor. Values > 1 will zoom in, values < 1 will zoom out.
    :return: The zoomed image as a numpy array.
    """
    height, width = image.shape[:2]

    # Center of the image
    center_x, center_y = width / 2, height / 2

    # The dimensions of the zoomed image
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)

    # The coordinates of the top-left corner of the zoomed image
    top_left_x = int(center_x - new_width / 2)
    top_left_y = int(center_y - new_height / 2)

    # Ensuring the coordinates are within the bounds of the original image
    top_left_x = max(top_left_x, 0)
    top_left_y = max(top_left_y, 0)
    new_width = min(new_width, width - top_left_x)
    new_height = min(new_height, height - top_left_y)

    # Cropping and resizing the image
    cropped_img = image[top_left_y: top_left_y + new_height, top_left_x: top_left_x + new_width]
    resized_img = cv2.resize(cropped_img, (width, height), interpolation=cv2.INTER_LINEAR)

    return resized_img

def binary_filter(img_bgr):
	"""
	A third method for converting an image to binary format, emphasizing red regions.

	Parameters:
	img_bgr: numpy.ndarray
		The input image in BGR format.

	Returns:
	numpy.ndarray
		The binary image with red regions highlighted.
	"""
	
	img     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	lower_red = np.array([100, 100, 100])
	upper_red = np.array([255, 255, 255])


	# Create a binary mask for the red color within the specified range
	mask = cv2.inRange(img_hsv, lower_red, upper_red)

	# Apply the mask to the original image to segment the red regions
	red_segmented    = cv2.bitwise_and(img, img, mask=mask)
	_, binary_image = cv2.threshold   (red_segmented, red_segmented.mean(), 255, cv2.THRESH_BINARY)


	return binary_image

def blur(img_bgr, kernel_size=5):
	"""
	Applies Gaussian blur to an image.

	Parameters:
	img_bgr: numpy.ndarray
		The input image in BGR format.
	kernel_size: int, optional
		The size of the Gaussian kernel.

	Returns:
	numpy.ndarray
		The blurred image.
	"""
	return cv2.GaussianBlur(img_bgr, (kernel_size, kernel_size), 0)
	
def filter_c(contours, image):
	"""
	Filters contours based on certain criteria like area and aspect ratio.

	Parameters:
	contours: list
		A list of detected contours.
	image: numpy.ndarray
		The original image.

	Returns:
	list
		A list of filtered contours.
	"""

	height, width = image.shape[:2]

	image_area = width * height

	n_list = []


	for cont in contours:

		if len(cont) < 6:
			continue

		ellipse = cv2.fitEllipse (cont)
		c_area  = cv2.contourArea(cont)

		# Extract the major and minor axes of the fitted ellipse
		major_axis, minor_axis = ellipse[1]

		# Calculate the aspect ratio (ratio of major axis to minor axis)
		ratio = major_axis / minor_axis

		if( c_area > MIN_AREA_PIXEL and
			c_area > image_area*MIN_AREA_OCCUPIED and 
			MIN_PANNEL_RATIO < ratio < MAX_PANNEL_RATIO):
			
			n_list.append(cont)


	return n_list

def pred_circle(img_bgr):
	"""
	Predicts whether a given image contains a circular shape.

	Parameters:
	img_bgr: numpy.ndarray
		The input image in BGR format.

	Returns:
	float
		The probability of the image containing a circle.
	"""
	# Load and preprocess the image
	img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
	img = cv2.resize  (img, (224, 224))
	img = np.expand_dims(img, axis=0)

	# Make a prediction
	prediction = shape_recognizer.predict(img, verbose=0)

	# Interpret the prediction
	return prediction[0][0]

def get_min_x_max_x_min_y_max_y(contour):
	"""
	Finds the bounding coordinates of a contour.

	Parameters:
	contour: numpy.ndarray
		The contour to be analyzed.

	Returns:
	tuple
		A tuple containing minimum and maximum X, Y coordinates (min_x, max_x, min_y, max_y).
	"""
	min_x, min_y =  float('inf'),  float('inf')
	max_x, max_y = -float('inf'), -float('inf')


	# Iterate through the points in the contour
	for point in contour:
		x, y  = point[0]
		min_x = min(min_x, x)
		max_x = max(max_x, x)
		min_y = min(min_y, y)
		max_y = max(max_y, y)


	return min_x, max_x, min_y, max_y

def predict_contour(contour):
	"""
	Predicts if a contour is likely to be a traffic sign based on its shape.

	Parameters:
	contour: numpy.ndarray
		The contour to be analyzed.

	Returns:
	tuple
		A tuple containing the prediction score and
	the processed image for the contour.
	"""

	min_x, max_x, min_y, max_y = get_min_x_max_x_min_y_max_y(contour)

	width  = max_x - min_x + 2*3
	height = max_y - min_y + 2*3

	offset_contour = contour.copy()


	for point in offset_contour:
		point[0][0] -= min_x - 3
		point[0][1] -= min_y - 3


	# Create a blank binary image
	new_image = np.zeros((height, width), dtype=np.uint8)

	cv2.drawContours(new_image, [offset_contour], 0, 255, thickness=2)

	pred = 100*(1 - pred_circle(new_image))


	return pred, new_image
	
def get_pannels(contours, img_rgb, threshold=80):
	"""
	Identifies and filters contours that are likely to be traffic signs.

	Parameters:
	contours: list
		A list of detected contours.
	img: nd.array
		Image working on
	threshold: float, optional
		The threshold score to consider a contour as a traffic sign.

	Returns:
	list
		A list of contours that are likely traffic signs.
	"""
	# TODO: improve the cross sign issue

	res = []
	
	for contour in filter_c(contours, img_rgb):

		min_x, max_x, min_y, max_y = get_min_x_max_x_min_y_max_y(contour)
		
		pred, im = predict_contour(contour)
		
		if(pred < threshold):
			continue
		
		res.append(img(min_x, max_x, min_y, max_y, pred, im))
	
	if len(res) < 2:
		return res
	
	# remove the regions that are in common
	n_res = []

	for r in range(len(res)):

		included = False
		
		for k in range(len(res)):

			if k != r and res[r].is_in(res[k]):				
				included = True
				break

		if not included: n_res.append(res[r])


	return n_res

class img:

	def __init__(self, min_x, max_x, min_y, max_y, pred, image):
		self.min_x = min_x
		self.max_x = max_x
		self.min_y = min_y
		self.max_y = max_y
		self.pred  = pred
		self.image = image

	def is_in(self, other_image):
		return  (self.min_x >= other_image.min_x) and \
				(self.max_x <= other_image.max_x) and \
				(self.min_y >= other_image.min_y) and \
				(self.max_y <= other_image.max_y)

	def __str__(self):
		return f"min_x: {self.min_x} max_x: {self.max_x} min_y: {self.min_y} max_y: {self.max_y}"

def predict_sign(photo):
	"""
	Predicts the type of traffic sign from an image.

	Parameters:
	photo: numpy.ndarray
		The image of the traffic sign.

	Returns:
	numpy.ndarray
		The prediction of the traffic sign type.
	"""

	photo = cv2.resize(photo, (224, 224))

	photo = np.expand_dims(photo, axis=0)

	photo = photo / 255.0 # normalise the photo

	predictions = sign_model.predict(photo, verbose=0)

	return predictions

def predict_pannel_sign(pannels, background_image):
	"""
	Predicts the type of traffic signs for multiple image regions.

	Parameters:
	pannels: list
		A list of image regions (traffic signs).
	background_image: numpy.ndarray
		The
	original image from which the regions are extracted.

	This function modifies each element in `pannels` by adding a `sign_prediction` attribute that contains the predicted type of the traffic sign.
	"""

	for i in pannels:
		im = background_image[i.min_y : i.max_y, i.min_x : i.max_x]
		prediction = predict_sign(im)
		i.sign_prediction = prediction.argmax()

def disply_im(imgs, im, save_path=''):
	"""
	Display an image with annotated rectangles around detected objects.

	This function visualizes an image and overlays rectangles around detected objects. 
	Each rectangle is annotated with the category of the detected object. It's designed 
	to be used in object detection tasks.

	Parameters:
	imgs (list of DetectedObject): A list of detected objects. Each object in the list 
	should have properties `min_x`, `min_y`, `max_x, max_y, and sign_prediction, which are used to draw and annotate rectangles on the image.
	im (ndarray): The image to be displayed. This should be a NumPy array, typically
	loaded through an image processing library like OpenCV or PIL.
	save_path (str): Optional parameter: Path for saving the photo. If the path is '', the photo won't be saved

	The function uses matplotlib for rendering the image and annotations.
	"""
	predict_pannel_sign(imgs, im)
	
	_, ax1 = plt.subplots(1, figsize=(20, 20))
	plt.xticks([]), plt.yticks([])
	ax1.imshow(im)

	for imm in imgs:
		ax1.add_patch(
			matplotlib.patches.Polygon([(imm.min_x, imm.min_y),
										(imm.max_x, imm.min_y),
										(imm.max_x, imm.max_y),
										(imm.min_x, imm.max_y)],
										edgecolor="green",
										linewidth=3,
										fill=False))
		
		plt.annotate(CATEGORIES[imm.sign_prediction], (imm.min_x + 2, imm.min_y - 2), color='yellow', size=16)

	# Save or display the image
	if save_path != '':
		plt.savefig(save_path, bbox_inches='tight')
	else:
		plt.show()

	# Close the plot to free up resources
	plt.close()
"""

for i in range(1, 100):
	fn = f"{img_dir}IMG_0{i:03d}.png"

	img_bgr = cv2.imread(fn)
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

	edged = edge_detection(blur(binary_filter(img_bgr)))
	contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	pannels = get_pannels(contours, 90)

	disply_im(pannels, img_rgb) #f'./aa{i}.png')

"""

def main_r(file_name:str = 'Nothing'):
	if file_name == 'Nothing':
		print('use: main(\'file\') with a file in the uploads directory')
	if file_name == 'file':
		print(image_path)
		img_bgr = cv2.imread(image_path)
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		edged = edge_detection(blur(binary_filter(img_bgr)))
		contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		pannels = get_pannels(contours, img_rgb, 90)

		disply_im(pannels, img_rgb, upload_path + "out.png")
		print("saving image to", upload_path,  "in.png")
		cv2.imwrite(upload_path + "in.png", img_bgr)

# if __name__ == '__main__':
# 	main('file')

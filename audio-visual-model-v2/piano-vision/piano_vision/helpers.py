import cv2
import numpy as np
import time
from math import inf


# def rotate_image(image, angle):
# 	image_center = tuple(np.array(image.shape[1::-1]) / 2)
# 	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
# 	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_AREA if angle % 360 > 0 else cv2.INTER_LINEAR)
# 	return result

def rotate_image(image, angle):
	# Image dimensions and center
    (height, width) = image.shape[:2]
    image_center = (width / 2, height / 2)
    
    # Caluclate rotation
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    
    # Calculate rotation angle (radians)
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])
    
    # Work out new width and height
    new_w = int(height * abs_sin + width * abs_cos)
    new_h = int(height * abs_cos + width * abs_sin)
    
    # Rotation matrix considering translation
    rot_mat[0, 2] += new_w / 2 - image_center[0]
    rot_mat[1, 2] += new_h / 2 - image_center[1]

    result = cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_AREA if angle % 360 > 0 else cv2.INTER_LINEAR)
    
    return result


def apply_mask(frame, mask):
    # Ensure mask is of the correct type
    if mask.dtype != np.uint8:
        mask = cv2.convertScaleAbs(mask)
    
    # Ensure mask and frame have the same size
    if mask.shape[:2] != frame.shape[:2]:
        raise ValueError("The mask and frame must have the same size.")
    
    # Apply mask
    return cv2.bitwise_and(frame, frame, mask=mask)


def mean_and_standard_dev(values, key=None):
	if key:
		values = tuple(map(key, values))

	return np.mean(values), np.std(values)


# def dist(p1, p2):
#     try:
#         dx = p1[0][0] - p2[0][0]
#         dy = p1[0][1] - p2[0][1]
#         return np.sqrt(dx ** 2 + dy ** 2)
#     except IndexError as e:
#         print(f"Error in dist function with p1: {p1}, p2: {p2}")
#         raise e

def dist(p1, p2):
    try:
        if isinstance(p1[0], (list, np.ndarray)) and isinstance(p2[0], (list, np.ndarray)):
            dx = p1[0][0] - p2[0][0]
            dy = p1[0][1] - p2[0][1]
        else:
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
        return np.sqrt(dx ** 2 + dy ** 2)
    except IndexError as e:
        print(f"Error in dist function with p1: {p1}, p2: {p2}")
        raise e



def group(data, radius, dist_func=dist):
	clustered = [[data[0]]]
	for val in data[1:]:
		if dist_func(val, clustered[-1][0]) < radius:
			clustered[-1].append(val)
		else:
			clustered.append([val])

	return clustered


def avg_of_groups(point_groups):
	point_avgs = []

	for group in point_groups:
		point_avg = [[0, 0]]

		for point in group:
			point_avg[0][0] += point[0][0]
			point_avg[0][1] += point[0][1]

		point_avg[0][0] = int(round(point_avg[0][0] / len(group)))
		point_avg[0][1] = int(round(point_avg[0][1] / len(group)))

		point_avgs += [point_avg]

	return point_avgs


def index_of_closest(data, points, dist_func=dist):
	ioc = []

	for point in points:
		dist_curr = float(inf)
		indx = -1
		for i, dpoint in enumerate(data):
			disti = dist_func(point, dpoint)
			if disti < dist_curr:
				dist_curr = disti
				indx = i
		ioc += [indx]

	return ioc


def centre_of_contour(contour):
	moments = cv2.moments(contour)
	centre_x = int(moments['m10'] / moments['m00'])
	centre_y = int(moments['m01'] / moments['m00'])
	return centre_x, centre_y


def epochtime_ms():
	return int(round(time.time() * 1000))

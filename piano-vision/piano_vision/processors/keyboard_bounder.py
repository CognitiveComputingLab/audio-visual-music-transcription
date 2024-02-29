import cv2
import numpy as np
from math import atan, degrees

class KeyboardBounder:
    def __init__(self, offset=25, min_contour_area=1000, min_aspect_ratio=1.0):
        self.OFFSET = offset
        self.MIN_CONTOUR_AREA = min_contour_area
        self.MIN_ASPECT_RATIO = min_aspect_ratio


	# Find the rotation 
    def find_rotation(self, frame) -> float:
        frame_copy = frame.copy()
        grey = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        # Apply canny edge detection
        edges = cv2.Canny(grey, 100, 200)
		# Apply hough line transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=50)

        angles = []
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x2 - x1 == 0:
                        angle = 90.0 if y2 > y1 else -90.0
                    else:
                        angle = degrees(atan((y2 - y1) / (x2 - x1)))
                    angles.append(angle)
                    cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if not angles:
            print("No angles found. Returning default rotation angle 0.")
            return 0

        angles.sort()
        median_angle = angles[len(angles) // 2]
        return median_angle


	# Identify the keyboard region
    def find_bounds(self, frame):
        frame_copy = frame.copy()
        # Convert to HSV colour
        hsv = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)
        # Find the white keys
        white = cv2.inRange(hsv, np.array([0, 0, 240]), np.array([255, 30, 255]))

		# Enhance the white regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        white = cv2.dilate(white, kernel, iterations=3)
        white = cv2.erode(white, kernel, iterations=5)
        white = cv2.dilate(white, kernel, iterations=2)

		# Find contours
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            raise ValueError("No contours found.")

        # Filter contours by area and aspect ratio
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= self.MIN_CONTOUR_AREA
                             and self.is_keyboard_aspect_ratio(contour)]

        if not filtered_contours:
            raise ValueError("No keyboard-sized contours found.")

        # largest_contour = max(filtered_contours, key=cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(largest_contour)
        # bounds = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        # print("Bounds:", bounds)
		
        candidate_bounds = [((x, y), (x + w, y), (x, y + h), (x + w, y + h)) for x, y, w, h in
                        (cv2.boundingRect(contour) for contour in filtered_contours if self.is_keyboard_aspect_ratio(contour))]

        min_aspect_ratio = 0.1
        max_aspect_ratio = 1000.0

        candidate_bounds = []

        for contour in contours:
            # Aspect ratio of the current contour
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            # Check if in range
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                candidate_bounds.append(((x, y), (x + w, y), (x + w, y + h), (x, y + h)))

        # Minimum keyboard contour area
        min_size_threshold = 10 

        candidate_bounds = []

        for contour in contours:
            # Area of the current contour
            area = cv2.contourArea(contour)

            # Check if in range
            if area >= min_size_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                candidate_bounds.append(((x, y), (x + w, y), (x + w, y + h), (x, y + h)))

        return candidate_bounds


    def is_keyboard_aspect_ratio(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        return aspect_ratio >= self.MIN_ASPECT_RATIO

    def get_bounded_section(self, frame, bounds):
        if len(bounds) != 4 or any(len(point) != 2 for point in bounds):
            raise ValueError("Bounds must be a list of 4 points (x, y).")
        
        min_x, max_x, min_y, max_y = bounds[0][0], bounds[1][0], bounds[0][1], bounds[2][1]
        
        corners_pre = np.float32([[min_x + self.OFFSET, min_y], [max_x - self.OFFSET, min_y], [min_x, max_y], [max_x, max_y]])
        corners_post = np.float32([[0, 0], [max_x - min_x, 0], [0, max_y - min_y], [max_x - min_x, max_y - min_y]])

        matrix = cv2.getPerspectiveTransform(corners_pre, corners_post)
        return cv2.warpPerspective(frame, matrix, (max_x - min_x, max_y - min_y))
    
	# Finds the corners of the keyboard
    # def find_keyboard_corners(self, frame):
    #     frame_copy = frame.copy()
    #     grey = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    #     edges = cv2.Canny(grey, 100, 200)

    #     # Detect lines using Hough line transform
    #     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=50)

    #     if lines is None or len(lines) < 4:
    #         raise ValueError("Not enough lines found to detect the keyboard.")

    #     intersections = []
    #     for i in range(len(lines)):
    #         for j in range(i+1, len(lines)):
    #             line1 = lines[i][0]
    #             line2 = lines[j][0]
    #             x1, y1, x2, y2 = line1
    #             x3, y3, x4, y4 = line2

    #             det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    #             if det != 0:  # Check for parallel lines
    #                 intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    #                 intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det
    #                 intersections.append((intersection_x, intersection_y))

    #     if len(intersections) < 4:
    #         raise ValueError("Not enough intersection points found to detect the keyboard quadrilateral.")
        
    #     for intersection in intersections:
    #         x, y = intersection
    #     cv2.circle(frame_copy, (int(x), int(y)), 5, (0, 255, 0), -1)

    #     cv2.imshow("Lines and Intersections", frame_copy)
    #     cv2.waitKey(0)  # Press any key to continue

    #     intersections.sort(key=lambda point: point[0])

    #     keyboard_corners = np.float32([
    #         intersections[0], intersections[1], intersections[2], intersections[3]
    #     ])

    #     return keyboard_corners


    # def transform_to_rectangular(self, frame, keyboard_corners):
    #     if len(keyboard_corners) != 4:
    #         raise ValueError("keyboard_corners must have 4 points")

    #     frame_copy = frame.copy()
    #     min_x, max_x = min(keyboard_corners[:, 0]), max(keyboard_corners[:, 0])
    #     min_y, max_y = min(keyboard_corners[:, 1]), max(keyboard_corners[:, 1])
		
    #     width = max_x - min_x
    #     height = max_y - min_y
    #     print(f"Calculated width: {width}, Calculated height: {height}")

    #     if width <= 0 or height <= 0 or width > frame.shape[1] * 2 or height > frame.shape[0] * 2:
    #         print(f"Unusual dimensions found: Width={width}, Height={height}")
    #         for pt in keyboard_corners:
    #             x, y = pt
    #             cv2.circle(frame_copy, (int(x), int(y)), 5, (0, 255, 0), -1)
    #         cv2.imshow("Unusual Dimensions", frame_copy)
    #         cv2.waitKey(0)
    #         raise ValueError("Calculated dimensions are not reasonable")

    #     corners_pre = keyboard_corners
    #     corners_post = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    #     matrix = cv2.getPerspectiveTransform(corners_pre, corners_post)
    
    #     if width <= 0 or height <= 0 or width > frame.shape[1] * 2 or height > frame.shape[0] * 2:
    #         print(f"Unusual dimensions found: Width={width}, Height={height}")
    #         for pt in keyboard_corners:
    #             x, y = pt
    #             cv2.circle(frame_copy, (int(x), int(y)), 5, (0, 255, 0), -1)
    #         cv2.imshow("Unusual Dimensions", frame_copy)
    #         cv2.waitKey(0)
    #         raise ValueError("Calculated dimensions are not reasonable")

    #     return cv2.warpPerspective(frame, matrix, (int(width), int(height)))

    
    def is_potential_keyboard(self, bounds):
        bounds = np.array(bounds, dtype=np.int32).reshape((-1, 1, 2))
        x, y, w, h = cv2.boundingRect(bounds)
        
        aspect_ratio = w / float(h)
        # Keyboard aspect ratio
        min_aspect_ratio = self.MIN_ASPECT_RATIO
        # Adjust maximum aspect ratio
        max_aspect_ratio = 10 #1000
        
        return min_aspect_ratio <= aspect_ratio <= max_aspect_ratio

    def get_brightness_lower_third(self, image):
        lower_third = image[2 * image.shape[0] // 3:, :]
        mean_brightness = cv2.mean(cv2.cvtColor(lower_third, cv2.COLOR_BGR2GRAY))[0]
        return mean_brightness

    def count_black_keys_upper_two_thirds(self, image):
        upper_two_thirds = image[:2 * image.shape[0] // 3, :]
        gray = cv2.cvtColor(upper_two_thirds, cv2.COLOR_BGR2GRAY)
		# Otsu thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		# Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
		# Assume the black keys are smaller components
        black_key_size_threshold = 200
        black_key_count = sum(1 for stat in stats[1:] if stat[cv2.CC_STAT_AREA] < black_key_size_threshold)
        return black_key_count
import cv2
import numpy as np
from math import atan, degrees
from itertools import combinations
import numpy as np
from scipy.spatial import distance as dist
import itertools
from math import atan2, degrees
from scipy.stats import circmean
import torch
from torchvision import transforms
from PIL import Image
from .keyboard_segmentation import KeyboardSegmentationModel
import torch.nn.functional as F

class KeyboardBounder:
    def __init__(self, offset=25, min_contour_area=1000, min_aspect_ratio=1.0):
        self.OFFSET = offset
        self.MIN_CONTOUR_AREA = min_contour_area
        self.MIN_ASPECT_RATIO = min_aspect_ratio

    def keyboard_segmentation(self, frame, original_image):
        # Perform the segmentation
        # Only keep the segmentated areas of the frame
        original_size = original_image.shape[:2]

        model = KeyboardSegmentationModel("FPN", "resnet50", in_channels=3, out_classes=1)
        model.load_state_dict(torch.load("../PSPNet/keyboardSegmentation_state_dict.pth"))
        model.eval()

        # Generate the segmentation mask
        with torch.no_grad():
            logits = model(frame)
            predicted_mask = logits.sigmoid()
            # prediction = model(frame)
            # predicted_mask = prediction.sigmoid()  # Convert to probability
            # predicted_mask = (predicted_mask > 0.5).float()

        # predicted_mask = torch.nn.functional.interpolate(predicted_mask, size=(frame.shape[2], frame.shape[3]), mode='nearest')

        predicted_mask = (predicted_mask > 0.5).float()

        # frame_masked = frame * predicted_mask
        # frame_masked = frame * predicted_mask.expand_as(frame)

        # frame_masked = frame_masked.squeeze(0).permute(1, 2, 0).byte().numpy()
        
        # predicted_mask = predicted_mask[0].squeeze().numpy()

        # # Convert both mask and the frame back to the original size
        # predicted_mask = predicted_mask.squeeze().numpy()
        # frame = frame.numpy().transpose(1, 2, 0)

        predicted_mask = F.interpolate(predicted_mask, size=original_size, mode='bilinear', align_corners=False)
        # frame = F.interpolate(frame, size=original_size, mode='bilinear', align_corners=False)

        # # Convert mask to binary (0 or 1)
        predicted_mask = predicted_mask.squeeze().cpu().numpy()
        # predicted_mask = predicted_mask.expand(-1, 3, -1, -1)
        
        if predicted_mask.ndim == 2:
            predicted_mask = np.stack([predicted_mask]*3, axis=-1) 
        
        if original_image.dtype == np.uint8:
            original_image = original_image.astype(np.float32) / 255
        
        if original_image.shape[:2] != predicted_mask.shape[:2]:
            predicted_mask = np.transpose(predicted_mask, (1, 0, 2)) 

        masked_image = original_image * predicted_mask

        # if frame.shape != predicted_mask.shape:
        #     raise ValueError(f"Frame and mask must have the same shape, got {frame.shape} and {predicted_mask.shape}")
        # frame_masked = frame * predicted_mask
        # # Apply mask to frame
        # # Ensure that the batch dimension is handled properly
        # if frame.dim() == 4 and frame.size(0) == 1:
        #     frame = frame.squeeze(0)  # Remove batch dimension if present

        # frame = frame.permute(1, 2, 0).numpy()
        if np.all(masked_image == 0):
            print("Masked image is all black, returning original image.")
            return original_image, predicted_mask

        if masked_image.dtype != np.uint8:
            # Scale to [0, 255] and convert to uint8
            masked_image = np.clip(masked_image * 255, 0, 255).astype(np.uint8)

        return masked_image, predicted_mask
    

    def apply_segmentation_mask(self, mask, original_image):
        if mask.ndim == 2:
            mask = np.stack([mask]*3, axis=-1) 
        
        if original_image.dtype == np.uint8:
            original_image = original_image.astype(np.float32) / 255
        
        if original_image.shape[:2] != mask.shape[:2]:
            mask = np.transpose(mask, (1, 0, 2)) 

        masked_image = original_image * mask

        # if frame.shape != predicted_mask.shape:
        #     raise ValueError(f"Frame and mask must have the same shape, got {frame.shape} and {predicted_mask.shape}")
        # frame_masked = frame * predicted_mask
        # # Apply mask to frame
        # # Ensure that the batch dimension is handled properly
        # if frame.dim() == 4 and frame.size(0) == 1:
        #     frame = frame.squeeze(0)  # Remove batch dimension if present

        # frame = frame.permute(1, 2, 0).numpy()
        if np.all(masked_image == 0):
            print("Masked image is all black, returning original image.")
            return original_image

        if masked_image.dtype != np.uint8:
            # Scale to [0, 255] and convert to uint8
            masked_image = np.clip(masked_image * 255, 0, 255).astype(np.uint8)

        return masked_image

    # Find the initial keyboard rotation
    def find_rotation(self, frame) -> float:
        # frame_copy = frame.copy()
        # Convert to grayscale and use Canny edge to find lines
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grey, 50, 150)
        
        # Use HoughLinesP to find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=60, maxLineGap=5)
            
        if lines is None:
            print("No lines found. Returning default rotation angle 0.")
            return 0.0
        
        # angles = []
        # if lines is not None:
        #     for line in lines:
        #         for x1, y1, x2, y2 in line:
        #             if x2 - x1 == 0:
        #                 angle = 90.0 if y2 > y1 else -90.0
        #             else:
        #                 angle = degrees(atan((y2 - y1) / (x2 - x1)))
        #             angles.append(angle)
        #             cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # if not angles:
        #     print("No angles found. Returning default rotation angle 0.")
        #     return 0

        # angles.sort()
        # median_angle = angles[len(angles) // 2]

        # angles = [atan2(y2 - y1, x2 - x1) for x1, y1, x2, y2 in lines[:, 0]]
        # normalized_angles = [degrees(angle) % 180 - 90 for angle in angles]
        # # Remove the most common orientation
        # main_orientation = np.median(normalized_angles)
        # filtered_angles = [angle for angle in normalized_angles if abs(angle - main_orientation) < 10]

        # if not filtered_angles:
        #     print("No main orientation found. Using median of all angles.")
        #     average_angle = main_orientation
        # else:
        #     average_angle = sum(filtered_angles) / len(filtered_angles)

        # Calculate angles of the lines
        angles = [atan2(y2 - y1, x2 - x1) for x1, y1, x2, y2 in lines[:, 0]]
        
        # Convert angles to degrees and normalize them to the range [-90, 90)
        normalized_angles = [degrees(angle) % 180 - 90 for angle in angles]
        
        # Determine if the majority of lines are vertical
        vertical_count = sum(1 for angle in normalized_angles if abs(angle) > 45)

        check = False
        
        # If more than half of the lines are vertical, decide on 90-degree rotation
        if vertical_count > len(normalized_angles) / 2:
            check = True
            # Determine direction based on the median of the angles
            return (90 if np.median(normalized_angles) < 0 else -90), check
        
        # If conditions for rotation are not met, don't rotate
        return 0.0, check

	# Find the possible keyboard bounding boxes
    def find_bounds(self, frame, attempt=1):
        # Order the four corners
        def order_corners(points):
            centroid = np.mean(points, axis=0)
            
            # Calculate angles to the center and sort
            angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
            sorted_idx = np.argsort(angles)
            sorted_points = points[sorted_idx]
            
            # Reorder the points cylindrically by checking the cross-product between the points
            vec = np.diff(sorted_points[[0, 1, -1], :], axis=0)
            # Correct order
            if np.cross(vec[0], vec[1]) > 0:
                ordered_points = sorted_points
            else:
                # Move all the points and check again
                ordered_points = np.roll(sorted_points, -1, axis=0)
            
            # Make sure the order is correct from bottom left
            sum_coords = np.sum(ordered_points, axis=1)
            bl_index = np.argmin(sum_coords)
            ordered_points = np.roll(ordered_points, -bl_index, axis=0)
            
            return tuple(map(tuple, ordered_points))

        frame_copy = frame.copy()
        # Convert colour to HSV
        hsv = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        # hsv = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        
        if attempt == 1:
            # Ensure the brighter pixels are viewable and set everything else to black
            white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([255, 30, 255]))

            # cv2.imshow("Detected Keyboard Corners", white)

            # Enhance the white regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            white = cv2.dilate(white, kernel, iterations=3)
            white = cv2.erode(white, kernel, iterations=5)
            white = cv2.dilate(white, kernel, iterations=2)
            # white = cv2.Canny(hsv, 50, 150)

            # cv2.imshow("Detected Keyboard Corners", white)

            # Find the contours with cv2 in-built function
            contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            if gray.dtype != np.uint8:
                gray = gray.astype(np.uint8)
            edges = cv2.Canny(gray, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            if attempt == 1:
                print("No contours found on first attempt, retrying with adaptive thresholding.")
                return self.find_bounds(frame, attempt + 1)
            else:
                raise ValueError("No contours found after retrying with adaptive thresholding.")

        # # Filter contours by area and aspect ratio
        # filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= self.MIN_CONTOUR_AREA
        #                      and self.is_keyboard_aspect_ratio(contour)]

        # if not filtered_contours:
        #     raise ValueError("No keyboard-sized contours found.")

        # Calculate the minimum keyboard contour area
        height, width = frame.shape[:2]
        total_pixels = height * width
        min_size_threshold = round(total_pixels * 0.05)

        candidate_bounds = []
        candidate_true_bounds = []

        # For each detected area, draw the two bounding boxes
        for contour in contours:
            # Area of the current contour
            area = cv2.contourArea(contour)

            # Check if the area of the current contour is larger than the threashold
            if area >= min_size_threshold:
                # Find the bounding box
                x, y, w, h = cv2.boundingRect(contour)
                candidate_bounds.append(((x, y), (x + w, y), (x + w, y + h), (x, y + h)))

                # Find the exact bounding box
                min_rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(min_rect)
                box = np.int0(box)

                cv2.drawContours(white, [box], 0, (100, 100, 100), 2)

                # Save the contour regions
                box = tuple(tuple(point) for point in box)
                box = order_corners(np.array(box))
                candidate_true_bounds.append(box)

                # print(candidate_bounds, candidate_true_bounds)
            
            # cv2.imshow("RECTANGE", white)
            # print(candidate_bounds, box)

        return candidate_bounds, candidate_true_bounds

    # def is_keyboard_aspect_ratio(self, contour):
    #     x, y, w, h = cv2.boundingRect(contour)
    #     aspect_ratio = float(w) / h
    #     return aspect_ratio >= self.MIN_ASPECT_RATIO
    


    # Identify the real keyboard bounding box from the selection before
    def get_bounded_section(self, frame, bounds):
        # Validate input bounds
        if len(bounds) != 4 or any(len(point) != 2 for point in bounds):
            raise ValueError("Bounds must be a list of 4 points (x, y).")
        
        # Define the input and destination bounds in the correct order
        # corners_pre = np.float32(bounds)
        
        # # Calculate width and height of the bounding box
        # width_a = np.sqrt(((bounds[2][0] - bounds[3][0]) ** 2) + ((bounds[2][1] - bounds[3][1]) ** 2))
        # width_b = np.sqrt(((bounds[1][0] - bounds[0][0]) ** 2) + ((bounds[1][1] - bounds[0][1]) ** 2))
        # height_a = np.sqrt(((bounds[1][0] - bounds[2][0]) ** 2) + ((bounds[1][1] - bounds[2][1]) ** 2))
        # height_b = np.sqrt(((bounds[0][0] - bounds[3][0]) ** 2) + ((bounds[0][1] - bounds[3][1]) ** 2))
        
        # max_width = max(int(width_a), int(width_b))
        # max_height = max(int(height_a), int(height_b))
        
        # corners_post = np.float32([
        #     [0, 0],
        #     [max_width - 1, 0],
        #     [max_width - 1, max_height - 1],
        #     [0, max_height - 1]
        # ])
        
        # # Perform the perspective transform
        # matrix = cv2.getPerspectiveTransform(corners_pre, corners_post)
        # print(corners_pre, corners_post)
        # warped = cv2.warpPerspective(frame, matrix, (max_width, max_height))
        
        # Directly extract min and max x, y from bounds
        min_x = min(bounds, key=lambda x: x[0])[0]
        max_x = max(bounds, key=lambda x: x[0])[0]
        min_y = min(bounds, key=lambda x: x[1])[1]
        max_y = max(bounds, key=lambda x: x[1])[1]
        
        # Crop the image to the bounding rectangle
        cropped = frame[min_y:max_y, min_x:max_x]
        
        return cropped
    


    # def bound_transform_and_flip(self, frame, bounds):
    #     frame_copy = frame.copy()
    #     (x_min, y_min), (_, _), (x_max, y_max), (_, _) = bounds
    #     frame_copy = frame_copy[int(y_min):int(y_max), int(x_min):int(x_max)]
    #     cv2.imshow("Detected Keyboard Corners", frame_copy)

    #     hsv = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)
    #     # hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
    #     white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([255, 30, 255]))

    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #     white = cv2.dilate(white, kernel, iterations=3)
    #     white = cv2.erode(white, kernel, iterations=5)
    #     white = cv2.dilate(white, kernel, iterations=2)

    #     def detect_keyboard_corners(image):
    #         # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         blurred = cv2.GaussianBlur(image, (5, 5), 0)

    #         _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)  # Adjust the threshold value as needed
    #         edges = cv2.Canny(thresh, 50, 150)
    #         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #         corners = []

    #         for cnt in contours:
    #             epsilon = 0.01 * cv2.arcLength(cnt, True)
    #             approx = cv2.approxPolyDP(cnt, epsilon, True)

    #             for p in approx:
    #                 corners.append(tuple(p[0]))

    #         corners.sort(key=lambda x: (x[1], x[0]))

    #         top_left = corners[0]
    #         top_right = corners[-1]
    #         bottom_left = corners[len(corners)//2]
    #         bottom_right = corners[(len(corners)//2) - 1]

    #         print([top_left, top_right, bottom_right, bottom_left])

    #         return np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
        
    #     bounds = detect_keyboard_corners(white)
        
    #     if len(bounds) != 4:
    #         raise ValueError("Did not find 4 corner points for perspective transform.")
        
    #     # width = max(np.linalg.norm(bounds[0] - bounds[1]), np.linalg.norm(bounds[2] - bounds[3]))
    #     # height = max(np.linalg.norm(bounds[0] - bounds[3]), np.linalg.norm(bounds[1] - bounds[2]))
    #     height, width = white.shape[:2]

    #     corners_post = np.float32([
    #         [0, 0],
    #         [width, 0],
    #         [width, height],
    #         [0, height]
    #     ])

    #     print(corners_post)

    #     matrix = cv2.getPerspectiveTransform(bounds, corners_post)
    #     warped = cv2.warpPerspective(frame, matrix, (int(width), int(height)))
        
    #     return warped

    def bound_transform_and_flip(self, frame, bounds, true_bounds):
        frame_copy = frame.copy()
        # cv2.imshow("NEW", frame)
        # print(bounds, true_bounds)

        # Find the correct size of the destination points to transform to
        x, y, x_plus_w, y_plus_h = bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1]
        width = x_plus_w - x
        height = y_plus_h - y

        # frame_copy = frame[bounds[0][1]:bounds[1][1], bounds[0][0]:bounds[1][0]]

        if len(true_bounds) != 4:
            raise ValueError("Did not find 4 corner points for perspective transform.")

        # Correctly define the destination points in the correct order
        output = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])

        true_bounds = np.float32(true_bounds)
        bounds = np.float32(bounds)

        # The perspective transform matrix
        # From the true exact keyboard region to the rectangular region
        matrix = cv2.getPerspectiveTransform(true_bounds, output)
        # Apply the perspective transformation
        warped = cv2.warpPerspective(frame_copy, matrix, (width, height))
        
        # cv2.imshow("Warped Section", warped)
        
        # frame_with_true_bounds = frame.copy()
        # true_bounds_int = np.int32(bounds)
        # cv2.drawContours(frame_with_true_bounds, [true_bounds_int], -1, (0, 255, 0), 3)
        # cv2.imshow("Frame with True Bounds", frame_with_true_bounds)

        # Determine whether the image needs to be flipped
        # Split the image into top and bottom halves
        top_half = warped[:height // 2, :]
        bottom_half = warped[height // 2:, :]

        # Count the number of black (< 50 RGB) pixels in each half
        black_thresh = 50
        top_black = np.sum(np.all(top_half < black_thresh, axis=2))
        bottom_black = np.sum(np.all(bottom_half < black_thresh, axis=2))

        # check2 = False

        # The half which has more black pixels will be the black notes
        # Flip the image if the bottom half has more black pixels than the top half
        if bottom_black > top_black:
            warped = cv2.flip(warped, 0)  # 0 indicates a flip around the x-axis (vertical flip)
            # check2 = True

        return warped

    # def check_image_rotation(self, warped):
    #     # cv2.imshow("NEW", frame)
    #     # print(bounds, true_bounds)

    #     height, width = warped.shape[:2]

    #     top_half = warped[:height // 2, :]
    #     bottom_half = warped[height // 2:, :]

    #     black_thresh = 50
    #     top_black = np.sum(np.all(top_half < black_thresh, axis=2))
    #     bottom_black = np.sum(np.all(bottom_half < black_thresh, axis=2))

    #     check2 = False

    #     if bottom_black > top_black:
    #         warped = cv2.flip(warped, 0)  # 0 indicates a flip around the x-axis (vertical flip)
    #         check2 = True

    #     return warped, check2



    ####### Combined method to find keyboard corners ########
    # def find_keyboard_corners(self, frame):
    #     def cross_product(a, b, c):
    #         ab = [b[0] - a[0], b[1] - a[1]]
    #         bc = [c[0] - b[0], c[1] - b[1]]
    #         return ab[0] * bc[1] - ab[1] * bc[0]

    #     def is_convex(quadrilateral):
    #         num_positive, num_negative = 0, 0
    #         for i in range(len(quadrilateral)):
    #             turn = cross_product(quadrilateral[i], quadrilateral[(i + 1) % len(quadrilateral)], quadrilateral[(i + 2) % len(quadrilateral)])
    #             if turn > 0:
    #                 num_positive += 1
    #             elif turn < 0:
    #                 num_negative += 1
    #         return num_positive == 0 or num_negative == 0

    #     def order_points(pts):
    #         rect = np.zeros((4, 2), dtype="float32")
    #         s = pts.sum(axis=1)
    #         rect[0] = pts[np.argmin(s)]
    #         rect[2] = pts[np.argmax(s)]
    #         diff = np.diff(pts, axis=1)
    #         rect[1] = pts[np.argmin(diff)]
    #         rect[3] = pts[np.argmax(diff)]
    #         return rect

    #     def is_valid_quadrilateral(intersections):
    #         if len(intersections) != 4:
    #             return False
    #         quadrilateral = order_points(np.array(intersections))
    #         return is_convex(quadrilateral)
        
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

    #     def find_intersections(lines):
    #         intersections = []
    #         for line1, line2 in itertools.combinations(lines, 2):
    #             x1, y1, x2, y2 = line1[0]
    #             x3, y3, x4, y4 = line2[0]
    #             det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    #             if det:
    #                 x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    #                 y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det
    #                 intersections.append((x, y))
    #         return intersections

    #     frame_copy = frame.copy()
    #     grey = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    #     # edges = cv2.Canny(grey, 100, 200)
    #     edges = cv2.Canny(grey, 50, 150, apertureSize=3)

    #     # Detect lines using Hough line transform
    #     # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=50)
    #     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    #     if lines is None or len(lines) < 4:
    #         raise ValueError("Not enough lines found to detect the keyboard.")
        
    #     intersections = find_intersections(lines)
    #     print(intersections)

    #     # for intersection in intersections:
    #     #     x, y = intersection
    #     # cv2.circle(frame_copy, (int(x), int(y)), 5, (0, 255, 0), -1)

    #     # cv2.imshow("Lines and Intersections", frame_copy)
    #     # cv2.waitKey(0)  # Press any key to continue

    #     # intersections.sort(key=lambda point: point[0])

    #     # keyboard_corners = np.float32([
    #     #     intersections[0], intersections[1], intersections[2], intersections[3]
    #     # ])

    #     for combination in itertools.combinations(intersections, 4):
    #         if is_valid_quadrilateral(combination):
    #             print(combination)
    #             keyboard_corners = order_points(np.array(combination))
    #             # Visualization for debugging
    #             for (x, y) in keyboard_corners:
    #                 cv2.circle(frame_copy, (int(x), int(y)), 5, (0, 255, 0), -1)
    #             cv2.imshow("Detected Keyboard Corners", frame_copy)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
    #             return keyboard_corners
        


    ####### Methods for a perspective transform and find exact key position #######
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

    #     return cv2.warpPerspective(frame, matrix, (int(width), int(height)))

    # # Find the brightness in the lower third
    # def get_brightness_lower_third(self, image):
    #     lower_third = image[2 * image.shape[0] // 3:, :]
    #     mean_brightness = cv2.mean(cv2.cvtColor(lower_third, cv2.COLOR_BGR2GRAY))[0]
    #     return mean_brightness

    # # Check the number of black keys in the upper two thrids of the image
    # def count_black_keys_upper_two_thirds(self, image):
    #     upper_two_thirds = image[:2 * image.shape[0] // 3, :]
    #     gray = cv2.cvtColor(upper_two_thirds, cv2.COLOR_BGR2GRAY)
    # # Otsu threasholding
    #     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)

    #     return black_key_count



    ####### Method with Hough line and a longer method for keyboard location detection #######
    # Find lines with Hough transform
    # def find_lines(self, frame):
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    #     lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    #     if lines is not None:
    #         return lines
    #     else:
    #         return []
    
    # Find the intersection point of lines
    # def line_intersection(self, line1, line2):
    #     rho1, theta1 = line1[0]
    #     rho2, theta2 = line2[0]
    #     A = np.array([
    #         [np.cos(theta1), np.sin(theta1)],
    #         [np.cos(theta2), np.sin(theta2)]
    #     ])
    #     b = np.array([[rho1], [rho2]])
    #     x0, y0 = np.linalg.solve(A, b)
    #     return [int(np.round(x0)), int(np.round(y0))]

    # def find_intersections(self, lines):
    #     intersections = []
    #     for line1, line2 in combinations(lines, 2):
    #         try:
    #             intersection = self.line_intersection(line1, line2)
    #             intersections.append(intersection)
    #         except np.linalg.LinAlgError:
    #             continue
    #     return intersections
    
    # Check whether points form a convex quadrilateral
    # def is_convex_quadrilateral(self, pts):
    #     pts = np.array(pts)
    #     pts = np.roll(pts, -pts.argmin(axis=0)[0], axis=0)
    #     cross_product = np.cross(pts[1] - pts[0], pts[2] - pts[1]) * np.cross(pts[2] - pts[1], pts[3] - pts[2])
    #     return cross_product > 0

    # def calculate_aspect_ratio(self, pts):
    #     pts = np.array(pts)
    #     _, _, w, h = cv2.boundingRect(pts)
    #     return w / h

    # def calculate_area(self, pts):
    #     pts = np.array(pts)
    #     return cv2.contourArea(pts)

    # Filter the four points to find the rectnagular quadrilateral
    # def filter_quadrilaterals(self, intersections, frame_shape):
    #     valid_quads = []
    #     for quad in combinations(intersections, 4):
    #         if self.is_convex_quadrilateral(quad):
    #             aspect_ratio = self.calculate_aspect_ratio(quad)
    #             area = self.calculate_area(quad)
    #             if 1.0 < aspect_ratio < 4.0 and 10000 < area < (frame_shape[0] * frame_shape[1]) / 2:
    #                 valid_quads.append(quad)
    #     return valid_quads

    # Order the points correctly
    # def order_points(pts):
    #     xSorted = pts[np.argsort(pts[:, 0]), :]

    #     leftMost = xSorted[:2, :]
    #     rightMost = xSorted[2:, :]

    #     leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    #     (tl, bl) = leftMost

    #     D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    #     (br, tr) = rightMost[np.argsort(D)[::-1], :]

    #     return np.array([tl, tr, br, bl], dtype="float32")

    # Transform the quadrilateral to a parallel recntagular region
    # def apply_perspective_transform(frame, src_points):
    #     width, height = 600, 200

    #     dst_points = np.array([
    #         [0, 0],
    #         [width - 1, 0],
    #         [width - 1, height - 1],
    #         [0, height - 1]],
    #         dtype="float32"
    #     )

    #     matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    #     warped = cv2.warpPerspective(frame, matrix, (width, height))

    #     return warped
from pathlib import Path

import cv2
import numpy as np
import mido

from .helpers import rotate_image
from .processors import KeysManager, KeyboardBounder, HandFinder, PressedKeyDetector
from .video_reader import VideoReader


class PianoVision:
	# Delay between reading frames
	DELAY = 15  
	# Frames between snapshots (30fps)
	SNAPSHOT_INTERVAL = 30  
	# Number of snapshots for list
	NUM_SNAPSHOTS = 200

	def __init__(self, video_name):
		self.video_name = video_name
		self.video_file = '{}.mp4'.format(video_name)
		self.ref_frame_file = '{}-f00.png'.format(video_name)

		output_dir = Path("")
		output_dir.mkdir(exist_ok=True)
		self.output_dir = output_dir

		self.reference_frame = None

		self.bounder = KeyboardBounder()
		self.bounds = [0, 0, 0, 0]
		self.true_bounds = [0, 0, 0, 0]

		self.hand_finder = HandFinder()
		self.keys_manager = None
		self.pressed_key_detector = None

		self.frame_counter = 0

		self.candidate_notes = []

	def main_loop(self):
		# open('output/{}.log'.format(self.video_name), 'w').close()

		with VideoReader(self.video_file) as video_reader:
			paused = False
			# Read the frame
			frame = video_reader.read_frame()

			# Use the first frame
			if Path(self.ref_frame_file).exists():
				initial_frame = cv2.imread(self.ref_frame_file)
			else:
				initial_frame = frame

			handled, bound_type = self.handle_reference_frame(initial_frame)

			if not handled:
				print("Failed to handle reference frame.")
				return

			# Find rotation and apply it to all the frames
			rotation, rotation_required = self.bounder.find_rotation(initial_frame)
			# frame, rotation2_required = self.bounder.check_image_rotation(initial_frame)
			# print(rotation_required, rotation2_required)

			initial_image = True

			# Loop through remaining frames
			while frame is not None:
				if bound_type == 2:
					frame = cv2.copyMakeBorder(frame, 100, 100, 100, 100,
												cv2.BORDER_CONSTANT, value=[0, 0, 0])
				# cv2.imshow('frame', frame)
				# print(f"self.bounds before get_bounded_section: {self.bounds}")
	
				# print('rotation: {}'.format(rotation))
				if rotation_required:
					rotation, _ = self.bounder.find_rotation(frame)
					frame = rotate_image(frame, rotation)
					# cv2.imshow("FRAME", frame)
					# cv2.waitKey(0)

				if bound_type == 1:
					keyboard = self.bounder.bound_transform_and_flip(frame, self.bounds, self.true_bounds)
				else:
					keyboard, _ = self.bounder.bound_transform_and_flip_2(frame, self.true_bounds)
				cv2.imshow('post_warp', keyboard)
	
				# if rotation2_required:
				# 	frame = self.bounder.check_image_rotation(frame)

				skin_mask = self.hand_finder.get_skin_mask(keyboard)
				if skin_mask.dtype != np.uint8:
					skin_mask = cv2.convertScaleAbs(skin_mask)


				# # Perform the background update if hands are detected covering the initial image
				# # Check for hand coverage
				# coverage = self.calculate_hand_coverage(hand_contours, keyboard)
				# if coverage < your_defined_threshold:
				# 	# Re-run handle_reference_frame only when hands are not covering the keyboard
				# 	if not self.handle_reference_frame(frame):
				# 		print("Failed to handle reference frame.")
				# 		return


				# Use morphological closing to join up hand segments
				kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
				skin_mask_closed = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
				if skin_mask_closed.dtype != np.uint8:
					skin_mask_closed = cv2.convertScaleAbs(skin_mask_closed)
				# cv2.imshow('skin_mask_closed', skin_mask_closed)
				hand_contours = self.hand_finder.get_hand_contours(skin_mask_closed)

				fingertips = self.hand_finder.find_fingertips(hand_contours, keyboard)
				flat_fingertips = []
				for hand in fingertips:
					flat_fingertips.extend(hand)

				pressed_keys = self.pressed_key_detector.detect_pressed_keys(keyboard, skin_mask, flat_fingertips)

				# cv2.imshow('keyboard vs. ref', np.vstack([keyboard, self.reference_frame]))

				# Show frame with keys overlayed
				for key in self.keys_manager.white_keys:
					x, y, w, h = key.x, key.y, key.width, key.height
					cv2.rectangle(keyboard, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=key in pressed_keys and cv2.FILLED or 1)
					cv2.putText(keyboard, str(key), (x + 3, y + h - 10), cv2.FONT_HERSHEY_PLAIN, 0.75, color=(0, 0, 255))
				for key in self.keys_manager.black_keys:
					x, y, w, h = key.x, key.y, key.width, key.height
					cv2.rectangle(keyboard, (x, y), (x + w, y + h), color=(255, 150, 75), thickness=key in pressed_keys and cv2.FILLED or 1)
					cv2.putText(keyboard, str(key), (x, y + h - 10), cv2.FONT_HERSHEY_PLAIN, 0.75, color=(255, 150, 75))

				if hand_contours:
					cv2.drawContours(keyboard, tuple(hand_contours), -1, color=(0, 255, 0), thickness=1)

				# Highlight detected fingertips
				for hand in fingertips:
					for finger in hand:
						if finger:
							cv2.circle(keyboard, finger, radius=5, color=(0, 255, 0), thickness=2)

				if initial_image == True:
					cv2.imwrite(str(self.output_dir / f'{self.video_name}-keys.png'), keyboard)
				# cv2.imshow('keyboard', keyboard)
				initial_image = False

				# Wait for 30ms then get next frame unless quit
				pressed_key = cv2.waitKey(self.DELAY) & 0xFF
				if pressed_key == 32:  # spacebar
					paused = not paused
				elif pressed_key == ord('r'):
					self.handle_reference_frame(frame)
				elif pressed_key == ord('q'):
					break
				if not paused:
					if self.frame_counter % self.SNAPSHOT_INTERVAL == 0:
						snapshot_index = self.frame_counter // self.SNAPSHOT_INTERVAL
						self.take_snapshot(snapshot_index, frame, keyboard, pressed_keys)
					self.frame_counter += 1
					frame = video_reader.read_frame()
			cv2.destroyAllWindows()
		return self.candidate_notes


	def handle_reference_frame(self, reference_frame):
		original = reference_frame.copy()
		# Define the border size
		border_size = 100  # You can adjust the size of the border as needed
		
		# Create a new image by adding a border around the original
		border_original = cv2.copyMakeBorder(original, border_size, border_size, border_size, border_size,
											cv2.BORDER_CONSTANT, value=[0, 0, 0])

		bound_method = 1
		# Find whether the reference frame needs to be rotated 90 degrees
		rotation, _ = self.bounder.find_rotation(reference_frame)
		# print('rotation: {}'.format(rotation))
		# Rotate the reference frame
		reference_frame = rotate_image(reference_frame, rotation)

		# keyboard_corners = self.bounder.find_keyboard_corners(reference_frame)
		# transformed_keyboard = self.bounder.transform_to_rectangular(reference_frame, keyboard_corners)
		# self.keys_manager = KeysManager(transformed_keyboard)
		# self.reference_frame = transformed_keyboard
		# self.pressed_key_detector = PressedKeyDetector(self.reference_frame, self.keys_manager)

		# Find all the candiate bounds for rectangular shapes
		# Find their exact corners additionally
		candidate_bounds, candidate_true_bounds = self.bounder.find_bounds(reference_frame)
		# print("DONE Candidate Bounds")

    # # Find and select the best candidate bound
	# 	best_candidate = None
	# 	best_brightness = -1
	# 	best_black_key_count = -1
	#   # Loop through each bound
	# 	for bounds in candidate_bounds:
	# 		transformed_keyboard = self.bounder.get_bounded_section(reference_frame, bounds)
	# 		# print("DONE Bounded Keys Section")
	# 		brightness = self.bounder.get_brightness_lower_third(transformed_keyboard)
	# 		# print("DONE White Keys")
	# 		black_key_count = self.bounder.count_black_keys_upper_two_thirds(transformed_keyboard)
	# 		# print("DONE Black Keys")
	# 		print("CURRENT: " + str(brightness) + " " + str(black_key_count) + "PREVIOUS BEST: " + str(best_brightness) + " " + str(best_black_key_count))
	# 		# Select the best candidate based on the number of black keys detected in each 
	# 		if black_key_count > best_black_key_count:
	# 			best_candidate = bounds
	# 			best_brightness = brightness
	# 			best_black_key_count = black_key_count
	# 			print(best_candidate, best_brightness, best_black_key_count)

		best_candidate = None
		highest_black_key_count = -1
		i = -1

		for bounds in candidate_bounds:
			try:
				i = i + 1

				# Crop the image based on the general bounds
				transformed_keyboard = self.bounder.get_bounded_section(reference_frame, bounds)
				
				# Initialize the KeysManager
				temp_keys_manager = KeysManager(transformed_keyboard)
				
				# Count the number of black keys
				current_black_key_count = len(temp_keys_manager.black_keys)
				# print(current_black_key_count)
				
				# Find the region with the most black keys to focus on
				if current_black_key_count > highest_black_key_count:
					best_candidate = bounds
					highest_black_key_count = current_black_key_count
					bounds_option = i
					self.keys_manager = temp_keys_manager
			except Exception as e:
				print(f"Error processing bounds {bounds}: {e}")

		if best_candidate is None:
			raise ValueError("No suitable keyboard area found.")

		# Perspective transform the detected region (keyboard)
  		# Rotate if the keyboard is upside-down
		self.bounds = best_candidate
		self.true_bounds = candidate_true_bounds[bounds_option]

		transformed_keyboard = self.bounder.bound_transform_and_flip(reference_frame, best_candidate, candidate_true_bounds[bounds_option])

		cv2.imshow("Candidate Bounds", transformed_keyboard)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		user_input = ""
		while user_input not in ("y", "n"):
			user_input = input("Do the bounds look correct? (y/n): ").lower().strip()
			if user_input not in ("y", "n"):
				print("Invalid input. Please enter 'y' for yes or 'n' for no.")

		if user_input == "n":
			bound_method = 2
			cv2.imshow('Original Frame', border_original)
			print("Please click on the four corners of the keyboard in the original frame. Press ESC when done.")

			# Setting up data for collecting clicks
			points = []
			# Define the click event within this scope so it uses the `points` variable correctly
			def click_event(event, x, y, flags, param):
				if event == cv2.EVENT_LBUTTONDOWN:
					if len(points) < 4:  # Ensure only 4 points are collected
						cross_size = 10  # Size of the cross arms
						cv2.line(param, (x - cross_size, y), (x + cross_size, y), (0, 255, 0), 2)
						cv2.line(param, (x, y - cross_size), (x, y + cross_size), (0, 255, 0), 2)
						points.append((x, y))
						print(f"Point selected: {x}, {y}")
						cv2.imshow('Original Frame', param)  # Update the display with the circle

			# Attach the click event function
			cv2.namedWindow('Original Frame')
			cv2.setMouseCallback('Original Frame', click_event, border_original)
			while True:
				cv2.imshow('Original Frame', border_original)
				key = cv2.waitKey(1) & 0xFF
				if key == 27 or len(points) == 4:  # Exit on ESC or after 4 points
					break

			cv2.destroyAllWindows()

			transform_original = cv2.copyMakeBorder(original, border_size, border_size, border_size, border_size,
											cv2.BORDER_CONSTANT, value=[0, 0, 0])

			points = tuple(points)
			# print(points)
			transformed_keyboard, rectangular_bounds = self.bounder.bound_transform_and_flip_2(transform_original, points)
			cv2.imshow("Candidate Bounds", transformed_keyboard)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

			self.bounds = rectangular_bounds
			# self.true_bounds = [(x + border_size, y + border_size) for x, y in points]
			self.true_bounds = points

			user_input_2 = ""
			while user_input_2 not in ("y", "n"):
				user_input_2 = input("Do the bounds look correct? (y/n): ").lower().strip()
				if user_input_2 not in ("y", "n"):
					print("Invalid input. Please enter 'y' (for yes) or 'n' (for no)")

			if user_input_2 == "n":
				raise ValueError("No visual model suitable.")
			
		else:
			print("Keyboard accepted as correctly transformed.")
		

		# transformed_keyboard, _ = self.bounder.check_image_rotation(transformed_keyboard)
		self.keys_manager = KeysManager(transformed_keyboard)
		# print(f"self.bounds updated to: {self.bounds}")

		# transformed_keyboard = self.bounder.get_bounded_section(reference_frame, bounds)

		self.keys_manager = KeysManager(transformed_keyboard)

		# Set the new image as the reference frame
		self.reference_frame = transformed_keyboard
		# Detect the location of keys
		self.pressed_key_detector = PressedKeyDetector(self.reference_frame, self.keys_manager)

		print('{} black keys found'.format(len(self.keys_manager.black_keys)))
		print('{} white keys found'.format(len(self.keys_manager.white_keys)))
		return True, bound_method


	def take_snapshot(self, snapshot_index, frame, keyboard, pressed_keys):
		# if snapshot_index < self.NUM_SNAPSHOTS:
		# 	# Pad the keyboard image to match the width of the frame
		# 	pad_width = frame.shape[1] - keyboard.shape[1]
		# 	# Pad the height vertically
		# 	pad_height = frame.shape[0] - keyboard.shape[0]
		# 	keyboard_padded = np.pad(keyboard, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
			
			# # Array shapes
			# print(f"Frame shape: {frame.shape}")
			# print(f"Keyboard padded shape: {keyboard_padded.shape}")

			# # Stack the frame and the padded keyboard image
			# combined_image = np.vstack([frame, keyboard_padded])
			
			# cv2.imwrite(
			# 	'output/{}-snapshot{:02d}.png'.format(self.video_name, snapshot_index),
			# 	combined_image
			# )
		
		# if snapshot_index < self.NUM_SNAPSHOTS:
			# cv2.imwrite(
			# 	'output/{}-snapshot{:02d}.png'.format(self.video_name, snapshot_index),
			# 	np.vstack([frame, keyboard_padded])
			# )
			# with open('output/{}.log'.format(self.video_name), 'a+') as log:
			# 	line = '{}: [{}]\n'.format(snapshot_index, ', '.join([str(key) for key in pressed_keys]))
			# 	log.write(line)
			# 	print(line, end='')

		note_to_midi_map = {
			'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
			'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
		}

		def note_name_to_midi(note_name):
			# print(note_name)
			# Skip empty strings
			if not note_name:
				print("NONE")
				return None

			# Last character is the octave part
			octave = note_name[-1]
			# The rest is the note part
			note_part = note_name[:-1]

			if not note_part or not octave.isdigit() or note_part not in note_to_midi_map:
				return None

			# Calculate the MIDI number
			midi_number = note_to_midi_map[note_part] + (int(octave) + 1) * 12
			return midi_number

		def notes_to_midi_numbers(note_lists):
			# Process each note in the list of lists
			return [[note_name_to_midi(note) for note in note_lists]]
		
		note_values = list([str(key) for key in pressed_keys])
		midi_note_values = notes_to_midi_numbers(note_values)
		self.candidate_notes.extend(midi_note_values)
		# print(len(self.candidate_notes.extend(midi_note_values)))
		# Close all cv2 windows
		cv2.destroyAllWindows


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
		self.video_file = 'data/{}.mp4'.format(video_name)
		self.ref_frame_file = 'data/{}-f00.png'.format(video_name)

		self.reference_frame = None

		self.bounder = KeyboardBounder()
		self.bounds = [0, 0, 0, 0]

		self.hand_finder = HandFinder()
		self.keys_manager = None
		self.pressed_key_detector = None

		self.frame_counter = 0

		self.candidate_notes = []

	def main_loop(self):
		open('output/{}.log'.format(self.video_name), 'w').close()

		with VideoReader(self.video_file) as video_reader:
			paused = False
			frame = video_reader.read_frame()

			# Use the first frame
			if Path(self.ref_frame_file).exists():
				initial_frame = cv2.imread(self.ref_frame_file)
			else:
				initial_frame = frame

			if not self.handle_reference_frame(initial_frame):
				print("Failed to handle reference frame.")
				return

			# Loop through remaining frames
			while frame is not None:
				cv2.imshow('frame', frame)
				# print(f"self.bounds before get_bounded_section: {self.bounds}")
				
				keyboard = self.bounder.get_bounded_section(frame, self.bounds)
				# cv2.imshow('post_warp', keyboard)

				skin_mask = self.hand_finder.get_skin_mask(keyboard)
				if skin_mask.dtype != np.uint8:
					skin_mask = cv2.convertScaleAbs(skin_mask)

				# Use morphological closing to join up hand segments
				# TODO maybe replace this with joining nearby contours?
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

				# Show frame with keys overlaid
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

				cv2.imshow('keyboard', keyboard)

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
		# # Find rotation and apply it to the reference frame
		# rotation = self.bounder.find_rotation(reference_frame)
		# print('rotation: {}'.format(rotation))
		# reference_frame = rotate_image(reference_frame, rotation)

		# # Find rotation and apply it to the reference frame
		# rotation = self.bounder.find_rotation(reference_frame)
		# print('rotation: {}'.format(rotation))
		# reference_frame = rotate_image(reference_frame, rotation)

		# # Hough Line Transform
		# keyboard_corners = self.bounder.find_keyboard_corners(reference_frame)
		
		# # Transform the keyboard area to a rectangular shape
		# transformed_keyboard = self.bounder.transform_to_rectangular(reference_frame, keyboard_corners)
		
		# self.keys_manager = KeysManager(transformed_keyboard)

		# self.reference_frame = transformed_keyboard
		# self.pressed_key_detector = PressedKeyDetector(self.reference_frame, self.keys_manager)


		# Get all candidate bounds
		candidate_bounds = self.bounder.find_bounds(reference_frame)

    # Compute metrics for each candidate and select the best one
		best_candidate = None
		best_brightness = -1
		best_black_key_count = -1
		for bounds in candidate_bounds:
			# bounds should be in format: ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
			transformed_keyboard = self.bounder.get_bounded_section(reference_frame, bounds)
			brightness = self.bounder.get_brightness_lower_third(transformed_keyboard)
			black_key_count = self.bounder.count_black_keys_upper_two_thirds(transformed_keyboard)

			# Select the best candidate (colud add here)
			if brightness > best_brightness and black_key_count > best_black_key_count:
				best_candidate = bounds
				best_brightness = brightness
				best_black_key_count = black_key_count

		if best_candidate is None:
			raise ValueError("No suitable keyboard area found.")

		# Convert best_candidate to the correct format
		self.bounds = best_candidate
		transformed_keyboard = self.bounder.get_bounded_section(reference_frame, best_candidate)
		self.keys_manager = KeysManager(transformed_keyboard)
		# print(f"self.bounds updated to: {self.bounds}")

		# The bounded section of the frame / the transformed keyboard
		transformed_keyboard = self.bounder.get_bounded_section(reference_frame, bounds)

		self.keys_manager = KeysManager(transformed_keyboard)

		# Proceed if the keyboard is valid
		self.reference_frame = transformed_keyboard
		self.pressed_key_detector = PressedKeyDetector(self.reference_frame, self.keys_manager)

		# Number of detected black and white notes found
		print('{} black keys found'.format(len(self.keys_manager.black_keys)))
		print('{} white keys found'.format(len(self.keys_manager.white_keys)))
		return True


	def take_snapshot(self, snapshot_index, frame, keyboard, pressed_keys):
		if snapshot_index < self.NUM_SNAPSHOTS:
			# Pad the keyboard image to match the width of the frame
			pad_width = frame.shape[1] - keyboard.shape[1]
			# Pad the height vertically
			pad_height = frame.shape[0] - keyboard.shape[0]
			keyboard_padded = np.pad(keyboard, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
			
			# Array shapes
			print(f"Frame shape: {frame.shape}")
			print(f"Keyboard padded shape: {keyboard_padded.shape}")

			# Stack the frame and the padded keyboard image
			combined_image = np.vstack([frame, keyboard_padded])
			
			cv2.imwrite(
				'output/{}-snapshot{:02d}.png'.format(self.video_name, snapshot_index),
				combined_image
			)
		
		if snapshot_index < self.NUM_SNAPSHOTS:
			cv2.imwrite(
				'output/{}-snapshot{:02d}.png'.format(self.video_name, snapshot_index),
				np.vstack([frame, keyboard_padded])
			)
			with open('output/{}.log'.format(self.video_name), 'a+') as log:
				line = '{}: [{}]\n'.format(snapshot_index, ', '.join([str(key) for key in pressed_keys]))
				log.write(line)
				print(line, end='')

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

import pickle
import Data

pickle_file_path = "transkun/test.pickle"

with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

text_file_path = "../OMAPS2/pickleCheck1.txt"

with open(text_file_path, 'w') as text_file:
    text_data = str(data)
    text_file.write(text_data)

print(f"Saved to {text_file_path}")

import glob
import os

# Specify the path to the input folder
input_folder = '../../OMAPS2/complete/text'
output_folder = '../../OMAPS2/complete/MV2H_text'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Find all text files in the input folder
input_files = glob.glob(os.path.join(input_folder, '*.txt'))

for input_file in input_files:
    # Determine the base name and extension of the current input file
    base_name, ext = os.path.splitext(os.path.basename(input_file))
    # Construct the output file name by appending "_converted" before the file extension
    # and specifying the output folder
    output_file = os.path.join(output_folder, f"{base_name}_converted{ext}")
    
    # Initialize a variable to keep track of the latest end time across all notes
    latest_end_time_ms = 0
    
    # Read the input file to extract note timings and velocities
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    with open(output_file, 'w') as outfile:
        for line in lines:
            start, end, note, velocity = line.strip().split('\t')
            start_ms = int(float(start) * 1000)
            end_ms = int(float(end) * 1000)
            # Update the latest end time if the current note ends later than any previous note
            latest_end_time_ms = max(latest_end_time_ms, end_ms)
            # Write the formatted note information to the output file, omitting velocity
            outfile.write(f"Note {note} {start_ms} {start_ms} {end_ms} 0\n")

        # Append fixed Key and Hierarchy information
        outfile.write("Key 0 Maj 0\n")
        outfile.write("Hierarchy 4,2 1 a=0 0\n")

        # Calculate the ending tatum as the multiple of 250 milliseconds just before the last note's end time
        end_tatum = (latest_end_time_ms // 250) * 250

        # Generate and append Tatum entries up to the calculated ending time
        for tatum_time in range(0, end_tatum + 1, 250):
            outfile.write(f"Tatum {tatum_time}\n")

print("Conversion completed for all files.")

import glob
import os

# Specify the path to the input folder
input_folder = '../../OMAPS/complete/text'
output_folder = '../../OMAPS/complete/MV2H_text'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Find all files in the input folder (assuming .txt extension for ground truth files)
input_files = glob.glob(os.path.join(input_folder, '*.txt'))

for input_file in input_files:
    # Determine the output file name by appending "_converted" before the file extension
    base_name, ext = os.path.splitext(os.path.basename(input_file))
    output_file = os.path.join(output_folder, f"{base_name}_converted{ext}")
    
    # Initialize variable to keep track of the latest end time
    latest_end_time_ms = 0
    
    # Read the input file and extract note timings
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    with open(output_file, 'w') as outfile:
        for line in lines:
            start, end, note = line.strip().split('\t')
            start_ms = int(float(start) * 1000)
            end_ms = int(float(end) * 1000)
            # Update the latest end time if this note ends later
            latest_end_time_ms = max(latest_end_time_ms, end_ms)
            # Write the formatted note information to the output file
            outfile.write(f"Note {note} {start_ms} {start_ms} {end_ms} 0\n")

        # Append the fixed Key and Hierarchy information
        outfile.write("Key 0 Maj 0\n")
        outfile.write("Hierarchy 4,2 1 a=0 0\n")

        # Determine the ending tatum as the multiple of 250 just before the last note's end time
        end_tatum = (latest_end_time_ms // 250) * 250

        # Generate Tatum entries up to the calculated ending time
        for tatum_time in range(0, end_tatum + 1, 250):
            outfile.write(f"Tatum {tatum_time}\n")

print("Conversion completed for all files.")

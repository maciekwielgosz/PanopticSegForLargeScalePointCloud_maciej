import os
import argparse

def rename_files(input_folder):
    # List all files in the given directory
    for filename in os.listdir(input_folder):
        # Construct the file path
        filepath = os.path.join(input_folder, filename)
        # Check if it's a file
        if os.path.isfile(filepath):
            # Replace '-' with '_' in the filename
            new_filename = filename.replace('-', '_')
            # Construct the new file path
            new_filepath = os.path.join(input_folder, new_filename)
            # Rename the file
            os.rename(filepath, new_filepath)
            print(f"Renamed: {filepath} to {new_filepath}")

def main():
    parser = argparse.ArgumentParser(description='Replace "-" with "_" in filenames within a directory.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder')

    args = parser.parse_args()
    rename_files(args.input_folder)

if __name__ == '__main__':
    main()

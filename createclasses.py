import os

def create_classes():
    # Folder containing the text files
    text_files_folder = "Marathon/train_gt3"

    # Create an empty list to store the text labels
    text_list = []
    # Iterate over all the text files in the folder
    for filename in os.listdir(text_files_folder):
        filepath = os.path.join(text_files_folder, filename)

        # Read the contents of the text file
        with open(filepath, "r") as file:
            line = file.readline().strip()
            # Extract the last element (the text label) from the line
            text_label = line.split()[-1]
            text_list.append(text_label)

    return text_list
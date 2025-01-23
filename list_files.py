import os

def list_files_and_folders(start_directory, indent=''):
    # Get the list of all entries in the directory
    entries = os.listdir(start_directory)
    # Filter out hidden files and directories
    entries = [entry for entry in entries if not entry.startswith('.')]
    
    for index, entry in enumerate(entries):
        # Create a full path to the entry
        entry_path = os.path.join(start_directory, entry)
        
        # Check if the entry is a directory
        if os.path.isdir(entry_path):
            # Print the directory name
            print(f'{indent}{entry}/')
            # Recursively list the contents of the directory
            new_indent = indent + '   '  # Increase indentation for nested items
            list_files_and_folders(entry_path, new_indent)
        else:
            # Print the file name
            print(f'{indent}{entry}')


directory_to_list = r"C:\Users\dell\AppData\Roaming\NextIAS"
list_files_and_folders(directory_to_list)

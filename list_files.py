import os

def list_files_and_folders(start_directory, indent='', ignore_list=None):
    print(f"{os.path.basename(start_directory)}/")
    if ignore_list is None:
        ignore_list = []

    try:
        entries = [entry for entry in os.listdir(start_directory)
                   if not entry.startswith(('.', '__')) and entry not in ignore_list]
    except PermissionError:
        print(f"{indent}|- [Permission Denied]")
        return

    for entry in entries:
        entry_path = os.path.join(start_directory, entry)
        if os.path.isdir(entry_path):
            print(f'{indent}|- {entry}/')
            list_files_and_folders(entry_path, indent + '   ', ignore_list)
        else:
            print(f'{indent}|- {entry}')


# Define the directory and list of folders/files to ignore
directory_to_list = r"C:\Users\dell\Desktop\personal rag"
ignore_these = ['node_modules', 'venv', 'build', '__pycache__','query_index.py']

list_files_and_folders(directory_to_list, ignore_list=ignore_these)

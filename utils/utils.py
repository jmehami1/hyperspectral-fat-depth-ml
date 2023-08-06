import os

def get_folder_names(path):
    """Get all folder names in a given directory

    Args:
        path (_type_): absolute path

    Returns:
        list of strings : folder names in path
    """
    folder_names = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    return folder_names
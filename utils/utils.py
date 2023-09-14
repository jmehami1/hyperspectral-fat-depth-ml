import os

def get_folder_names(path):
    """Get all folder names in a given directory

    Args:
        path (str): absolute path

    Returns:
        list of str : folder names in path
    """
    folder_names = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    return folder_names



def has_folder(path, folder_name):
    """Checks if path has the given folder

    Args:
        path (str): absolute path
        folder_name (str): folder to check if exists

    Returns:
        bool: is folder exists in path
    """    
    return os.path.exists(os.path.join(path, folder_name))
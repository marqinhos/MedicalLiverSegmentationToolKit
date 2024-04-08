import os 


def load_from_file(path):
    """Function to load the data from a file.

    Args:
        path (str): Path to the file.
    """
            
    path = os.path.normpath(os.path.join(os.path.dirname(__file__), path))  
    return path
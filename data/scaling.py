import numpy as np

class DataScaler:
    def __init__(self, method='normalize', min_value=0.0, max_value=1.0):
        """
        Initialize the DataScaler.
        
        Args:
            method (str): The scaling method to use. Options: 'normalize' or 'standardize'.
        """
        self.method = method
        self.min_value=min_value 
        self.max_value=max_value

    def scale(self, data):
        """
        Scale the input data based on the chosen method.
        
        Args:
            data (numpy.ndarray): The input data to be scaled.
            
        Returns:
            numpy.ndarray: The scaled data.
        """
        if self.method == 'normalize':
            return self.normalize(data, self.min_value, self.max_value)
        elif self.method == 'standardize':
            return self.standardize(data)
        else:
            raise ValueError("Invalid scaling method. Options are 'normalize' or 'standardize'.")

    def normalize(self, data, min_value=0.0, max_value=1.0):
        """
        Normalize the input data given range. Default between 0 and 1
        
        Args:
            data (numpy.ndarray): The input data to be normalized.
            min_value (float): The minimum value of the desired range.
            max_value (float): The maximum value of the desired range.
        Returns:
            numpy.ndarray: The normalized data.
        """
        normalized_data = min_value + (data - np.min(data)) * (max_value - min_value) / (np.max(data) - np.min(data))
        return normalized_data

    def standardize(self, data):
        """
        Standardize the input data using the mean and standard deviation.
        
        Args:
            data (numpy.ndarray): The input data to be standardized.
            
        Returns:
            numpy.ndarray: The standardized data.
        """
        standardized_data = (data - np.mean(data)) / np.std(data)
        return standardized_data

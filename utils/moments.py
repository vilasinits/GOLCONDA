from utils.config import *

def get_moments_from_map(map_values):
    """
    Calculates the moments (mean, variance, skewness, kurtosis) from a 2D map.

    Parameters:
        map_values (numpy.ndarray): A 2D array representing the map values (e.g., convergence, density field).

    Returns:
        tuple: Contains mean, variance, skewness, kurtosis of the map values.
    """
    # Flatten the map to treat it as a distribution of values
    flattened_values = map_values.flatten()

    # Calculate the mean
    mean_value = np.mean(flattened_values)

    # Calculate the variance (second central moment)
    variance = np.var(flattened_values)

    # Calculate the third moment (skewness numerator)
    third_moment = np.mean((flattened_values - mean_value) ** 3)

    # Calculate the fourth moment (kurtosis numerator)
    fourth_moment = np.mean((flattened_values - mean_value) ** 4)

    # Skewness: third moment divided by variance squared
    skewness = third_moment / (variance ** 2)

    # Kurtosis: fourth moment divided by variance squared, minus 3 for excess kurtosis
    kurtosis = (fourth_moment / (variance ** 2)) - 3.0

    return mean_value, variance, skewness, kurtosis
# from utils.config import *
import numpy as np
from astropy.io import fits
import astropy.units as u
import os

class SimulationDataLoader:
    """
    Class to load simulation data from various sources based on simulation type and file format.
    
    Attributes:
    -----------
    user_defined_angle : float, optional
        A default angle to use if no angle is found in the data file.
    """

    def __init__(self, user_defined_angle=None):
        """
        Initialize the SimulationDataLoader with an optional default angle.
        
        Parameters:
        -----------
        user_defined_angle : float, optional
            A default angle to use if no angle is found in the file.
        """
        self.user_defined_angle = user_defined_angle

    def load_data(self, file_path, angle=None):
        """
        Load data based on the provided file path, automatically determining the simulation type and file format.
        
        Parameters:
        -----------
        file_path : str
            The path to the data file.
        angle : float, optional
            A user-defined angle to use if the file does not contain angle information.
        
        Returns:
        --------
        tuple : (data, angle)
            The data array and corresponding angular resolution.
        """
        simulation_type = self._detect_simulation_type(file_path)
        
        if simulation_type == 'massivenus':
            return self._load_massivenus_data(file_path, angle)
        elif simulation_type == 'slics':
            return self._load_slics_data(file_path, angle)
        elif simulation_type == 'howls':
            return self._load_howls_data(file_path, angle)
        else:
            raise ValueError(f"Unsupported simulation type: {simulation_type}")

    def _detect_simulation_type(self, file_path):
        """
        Detect the simulation type based on the file name or directory structure.
        
        Parameters:
        -----------
        file_path : str
            The path to the data file.
        
        Returns:
        --------
        str
            The detected simulation type ('massivenus', 'slics', 'howls', etc.).
        """
        file_name = os.path.basename(file_path).lower()

        if 'massivenus' in file_name:
            return 'massivenus'
        elif 'slics' in file_name or '.dat_los' in file_name:
            return 'slics'
        elif 'howls' in file_name:
            return 'howls'
        else:
            raise ValueError("Unable to detect simulation type from the file name.")

    def _load_massivenus_data(self, file_path, angle=None):
        """
        Load data for the MassiveNuS simulation (typically from a FITS file).
        
        Parameters:
        -----------
        file_path : str
            Path to the MassiveNuS data file.
        angle : float, optional
            A user-defined angle if the file does not contain angle information.
        
        Returns:
        --------
        tuple : (data, angle)
            The data array and corresponding angular resolution.
        """
        with fits.open(file_path) as hdu:
            data = hdu[0].data
            header_angle = hdu[0].header.get("ANGLE", None)

        if header_angle is not None:
            return data, header_angle * u.deg
        elif angle is not None:
            return data, angle * u.deg
        elif self.user_defined_angle is not None:
            return data, self.user_defined_angle * u.deg
        else:
            raise ValueError("No angle information found or provided for MassiveNuS data.")

    def _load_slics_data(self, file_path, angle=None):
        """
        Load data for the SLICS simulation (from binary or .dat_LOS files).
        
        Parameters:
        -----------
        file_path : str
            Path to the SLICS data file.
        angle : float, optional
            A user-defined angle for the data.
        
        Returns:
        --------
        tuple : (data, angle)
            The data array and corresponding angular resolution.
        """
        npix = 7745 
        with open(file_path, 'rb') as f:
            data_bin = np.fromfile(f, dtype=np.float32)
            data = np.reshape(data_bin, [npix, npix])
        data *= 64.0

        if angle is not None:
            return data, angle * u.deg
        elif self.user_defined_angle is not None:
            return data, self.user_defined_angle * u.deg
        else:
            raise ValueError("No angle information provided for SLICS data.")

    def _load_howls_data(self, file_path, angle=None):
        """
        Load data for the HOWLS simulation (typically from FITS files).
        
        Parameters:
        -----------
        file_path : str
            Path to the HOWLS data file.
        angle : float, optional
            A user-defined angle for the data.
        
        Returns:
        --------
        tuple : (data, angle)
            The data array and corresponding angular resolution.
        """
        with fits.open(file_path) as hdu:
            data = hdu[0].data
            header_angle = hdu[0].header.get("ANGLE", None)

        if header_angle is not None:
            return data, header_angle * u.deg
        elif angle is not None:
            return data, angle * u.deg
        elif self.user_defined_angle is not None:
            return data, self.user_defined_angle * u.deg
        else:
            raise ValueError("No angle information found or provided for HOWLS data.")

# Example usage
if __name__ == "__main__":
    loader = SimulationDataLoader(user_defined_angle=5)  # Default angle of 5 degrees

    # Load MassiveNuS data from a FITS file
    massivenus_data, massivenus_angle = loader.load_data("/path/to/massivenus_file.fits")
    print(f"MassiveNuS Data Shape: {massivenus_data.shape}, Angle: {massivenus_angle}")

    # Load SLICS data from a binary or .dat_LOS file
    slics_data, slics_angle = loader.load_data("/path/to/slics_file.dat_LOS")
    print(f"SLICS Data Shape: {slics_data.shape}, Angle: {slics_angle}")

    # Load HOWLS data from a FITS file
    howls_data, howls_angle = loader.load_data("/path/to/howls_file.fits")
    print(f"HOWLS Data Shape: {howls_data.shape}, Angle: {howls_angle}")

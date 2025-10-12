from __future__ import annotations
import csv
import os
from typing import Optional
import numpy as np

def read_eeg(path: str) -> np.ndarray:
    """
    Summary:
    Reads EEG data from a supported file format into a standardized 2D NumPy array.

    Description:
    WHY: This function provides a unified interface for loading EEG data, abstracting away the specifics of the underlying file format. It ensures that data loaded from different sources is consistently formatted.

    WHEN: Use this function as the primary entry point for loading a single EEG recording from a file into memory for subsequent processing or analysis.

    WHERE: It acts as a dispatcher in the data loading pipeline. It identifies the file type and delegates the reading task to a format-specific helper function (`_read_csv` or `_read_npy`) and then ensures the output format is consistent.

    HOW: The function inspects the file extension of the provided path. Based on whether the extension is `.csv` or `.npy`, it calls the appropriate internal reader. Finally, it passes the loaded data through a validation step (`_ensure_channels_time`) to guarantee the output array has a standard `(channels, time_points)` shape.

    Args:
    path (str): The file path to the EEG data. Supported extensions are `.csv` and `.npy`.

    Returns:
    np.ndarray: A 2D NumPy array of `float32` type representing the EEG signal, with dimensions corresponding to `(channels, time_points)`.

    Raises:
    ValueError: If the file extension is not one of the supported formats (`.csv`, `.npy`). This can also be raised by underlying functions if the data inside the file is malformed.
    FileNotFoundError: If the file at the specified `path` does not exist.

    Examples:
    ```python
    # Assume 'eeg_recording.csv' and 'eeg_recording.npy' are valid data files.

    # Load data from a CSV file
    try:
        eeg_from_csv = read_eeg('path/to/eeg_recording.csv')
        print(f"Loaded from CSV, shape: {eeg_from_csv.shape}")
    except FileNotFoundError:
        print("CSV file not found.")

    # Load data from a NumPy binary file
    try:
        eeg_from_npy = read_eeg('path/to/eeg_recording.npy')
        print(f"Loaded from NPY, shape: {eeg_from_npy.shape}")
    except FileNotFoundError:
        print("NPY file not found.")

    # Example of an unsupported file type
    try:
        read_eeg('path/to/unsupported_file.txt')
    except ValueError as e:
        print(f"Error handling unsupported file: {e}")
    # Expected output:
    # Error handling unsupported file: Unsupported EEG file extension: .txt for path: path/to/unsupported_file.txt
    ```
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        arr = _read_csv(path)
    elif ext == '.npy':
        arr = _read_npy(path)
    else:
        raise ValueError(f'Unsupported EEG file extension: {ext} for path: {path}')
    return _ensure_channels_time(arr, path)

def _read_csv(path: str) -> np.ndarray:
    """
    Summary:
    Reads EEG data from a CSV file and converts it to a NumPy array.

    Description:
    WHY: Provides a standardized method for loading EEG signal data from CSV files, ensuring consistent data type conversion.

    WHEN: Used internally during EEG data loading when a CSV file is detected as the input format.

    WHERE: Acts as a helper function within the EEG data loading pipeline, specifically handling CSV file parsing.

    HOW: Opens the CSV file, reads all rows using Python's csv module, and converts the data to a 32-bit floating-point NumPy array.

    Args:
        path (str): Full file path to the CSV containing EEG data. File should be a valid CSV with numeric data.

    Returns:
        np.ndarray: A 2D NumPy array of float32 type representing the raw EEG signal data, with rows potentially representing channels or time points.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        csv.Error: If there are issues parsing the CSV file structure.
        ValueError: If the CSV contains non-numeric data that cannot be converted to float32.

    Examples:
        ```python
        # Typical usage within read_eeg function
        try:
            eeg_data = _read_csv('recording.csv')
            print(f"Loaded CSV data with shape: {eeg_data.shape}")
        except FileNotFoundError:
            print("CSV file not found")
        except ValueError as e:
            print(f"Data conversion error: {e}")
        ```
    """
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return np.asarray(data, dtype=np.float32)

def _read_npy(path: str) -> np.ndarray:
    """
    Summary:
    Loads EEG data from a NumPy binary file, ensuring a consistent 2D array representation.

    Description:
    WHY: Provides a robust method for loading NumPy-stored EEG data with automatic dimensionality handling and type conversion.

    WHEN: Used internally during EEG data loading when a NumPy (.npy) file is detected as the input format.

    WHERE: Acts as a specialized helper function in the EEG data loading pipeline for processing NumPy binary files.

    HOW: Loads the NumPy array, handles various input dimensionalities (1D or multi-dimensional), ensures 2D representation, and converts to float32 data type.

    Args:
        path (str): Full file path to the NumPy binary file containing EEG data.

    Returns:
        np.ndarray: A 2D NumPy array of float32 type representing the EEG signal data, with guaranteed 2D shape.

    Raises:
        ValueError: If the loaded array cannot be transformed into a 2D array.
        FileNotFoundError: If the specified NumPy file does not exist.

    Examples:
        ```python
        # Typical usage within read_eeg function
        try:
            eeg_data = _read_npy('recording.npy')
            print(f"Loaded NumPy data with shape: {eeg_data.shape}")
        except FileNotFoundError:
            print("NumPy file not found")
        except ValueError as e:
            print(f"Data loading error: {e}")
        ```
    """
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    else:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D array for EEG data, got shape {arr.shape} from {path}')
    return arr.astype(np.float32, copy=False)

def _ensure_channels_time(arr: np.ndarray, src: str | None=None) -> np.ndarray:
    """
    Summary:
    Validates and normalizes EEG data array dimensions to a standard (channels, time_points) format.

    Description:
    WHY: Ensures consistent data representation for downstream EEG signal processing by enforcing a standard array layout.

    WHEN: Used internally during EEG data loading to guarantee uniform array structure before further analysis.

    WHERE: Acts as a preprocessing validation step in the EEG data loading pipeline, ensuring data consistency.

    HOW: Checks array dimensionality, transposes if channels and time points are incorrectly oriented, and validates that the number of time points significantly exceeds the number of channels.

    Args:
        arr (np.ndarray): Input 2D NumPy array representing EEG signal data.
        src (str | None, optional): Source identifier for more informative error messages. Defaults to None.

    Returns:
        np.ndarray: A 2D NumPy array with dimensions (channels, time_points), potentially transposed from the input.

    Raises:
        ValueError: If the input array is not 2D or cannot be transformed into a valid (channels, time_points) format.
            - Triggered when array dimensionality is incorrect
            - Raised if channels dimension is not smaller than time points dimension

    Examples:
        ```python
        # Correctly oriented array (no change)
        arr = np.random.rand(8, 1000)  # 8 channels, 1000 time points
        normalized = _ensure_channels_time(arr)  # Returns arr as-is

        # Transposed array
        arr = np.random.rand(1000, 8)  # 1000 time points, 8 channels
        normalized = _ensure_channels_time(arr)  # Returns transposed array

        # Invalid array (raises ValueError)
        arr = np.random.rand(10, 10)  # Equal channels and time points
        _ensure_channels_time(arr)  # Raises ValueError
        ```
    """
    if arr.ndim != 2:
        raise ValueError(f"EEG array must be 2D, got shape {arr.shape} from {src or 'array'}")
    (c, t) = arr.shape
    if c > t:
        arr = arr.T
        (c, t) = arr.shape
    if c >= t:
        raise ValueError(f"Cannot determine (C, T) with C < T for {src or 'array'}; got shape {arr.shape}")
    return arr

def gen_eeg(C: int=32, T: int=1024, *, sample_rate: float=256.0, mode: str='mixed', noise_std: float=0.1, num_components: int=3, seed: Optional[int]=None) -> np.ndarray:
    """
    Summary:
    Generates synthetic electroencephalogram (EEG)-like data with customizable channel count, time points, signal characteristics, and noise levels.

    Description:
    WHY: This function provides a versatile way to create simulated EEG data, which is invaluable for testing algorithms, prototyping models, demonstrating concepts, or generating synthetic datasets when real data is unavailable or restricted. It allows control over the underlying signal complexity (sine waves) and the addition of random noise.
    WHEN: Use this function when you need to generate a controlled, reproducible, multi-channel time-series signal that mimics the general characteristics of EEG data for development, debugging, or educational purposes. It's particularly useful for verifying that data processing pipelines and models handle various signal types and noise conditions correctly.
    WHERE: It typically fits into data generation scripts, unit tests, or examples for EEG signal processing libraries and machine learning projects, serving as a flexible source of input data.
    HOW: It initializes a 2D NumPy array of shape `(C, T)` representing `C` channels and `T` time samples. Depending on the `mode` parameter:
    - For 'sine' or 'mixed' modes, each channel is populated by summing a specified number of randomly generated sine waves, each with unique frequency, amplitude, and phase.
    - For 'noise' or 'mixed' modes, Gaussian noise with a specified standard deviation is added across all channels and time points.
    The `sample_rate` is used to define the time vector for sine wave generation. A `seed` can be provided for reproducible data generation.

    Args:
    C: int, default 32. The number of channels to generate in the synthetic EEG data.
        Constraints: Must be a positive integer.
    T: int, default 1024. The number of time samples (data points) for each channel.
        Constraints: Must be a positive integer.
    sample_rate: float, default 256.0. The simulated sampling rate in Hz. This affects the time vector and thus the interpretation of frequencies in the sine waves.
        Constraints: Must be a positive float.
    mode: str, default 'mixed'. Specifies the type of signal to generate.
        Valid values:
        - 'sine': Generates only sine wave components, without additional Gaussian noise.
        - 'noise': Generates only Gaussian noise, without sine wave components.
        - 'mixed': Generates both sine wave components and Gaussian noise.
    noise_std: float, default 0.1. The standard deviation of the Gaussian noise added to the signal when `mode` is 'noise' or 'mixed'. A value of 0.0 effectively disables noise.
        Constraints: Must be a non-negative float.
    num_components: int, default 3. The number of random sine wave components to sum for each channel when `mode` is 'sine' or 'mixed'. Each component has a randomly chosen frequency, amplitude, and phase.
        Constraints: Must be a positive integer.
    seed: Optional[int], default None. An integer seed for the random number generator. Providing a seed ensures that the generated EEG data is reproducible across multiple calls.

    Returns:
    np.ndarray: A 2-dimensional NumPy array of shape `(C, T)` and `np.float32` dtype, containing the synthetic EEG-like data.

    Raises:
    ValueError: If an unsupported `mode` string is provided (i.e., not 'sine', 'noise', or 'mixed').

    Examples:
    ```python
    import numpy as np

    # 1. Generate default mixed EEG-like data (32 channels, 1024 samples, mixed sine and noise)
    eeg_data_default = gen_eeg()
    print(f"Default EEG data shape: {eeg_data_default.shape}")
    print(f"Default EEG data dtype: {eeg_data_default.dtype}")
    print(f"Default EEG data values range: [{eeg_data_default.min():.2f}, {eeg_data_default.max():.2f}]")

    # 2. Generate pure sine wave data with specific dimensions and reproducibility
    eeg_sine_reproducible = gen_eeg(C=8, T=512, mode='sine', num_components=5, seed=42)
    print(f"\\nSine wave EEG data shape: {eeg_sine_reproducible.shape}")
    print(f"Sine wave EEG data values range: [{eeg_sine_reproducible.min():.2f}, {eeg_sine_reproducible.max():.2f}]")

    # Verify reproducibility
    eeg_sine_reproducible_2 = gen_eeg(C=8, T=512, mode='sine', num_components=5, seed=42)
    assert np.array_equal(eeg_sine_reproducible, eeg_sine_reproducible_2)
    print("Sine wave data with same seed is reproducible.")

    # 3. Generate pure noise data with higher standard deviation
    eeg_noise = gen_eeg(C=16, T=2048, mode='noise', noise_std=0.5)
    print(f"\\nNoise-only EEG data shape: {eeg_noise.shape}")
    print(f"Noise-only EEG data values range: [{eeg_noise.min():.2f}, {eeg_noise.max():.2f}]")

    # 4. Generate mixed data with adjusted parameters
    eeg_custom_mixed = gen_eeg(C=4, T=200, sample_rate=128.0, mode='mixed', noise_std=0.2, num_components=2)
    print(f"\\nCustom mixed EEG data shape: {eeg_custom_mixed.shape}")

    # 5. Demonstrate error handling for an invalid mode
    try:
        gen_eeg(mode='unsupported_mode')
    except ValueError as e:
        print(f"\\nCaught expected error: {e}")
    ```
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32) / float(sample_rate)
    x = np.zeros((C, T), dtype=np.float32)
    if mode not in {'sine', 'noise', 'mixed'}:
        raise ValueError(f'Unsupported mode: {mode}')
    if mode in {'sine', 'mixed'}:
        for c in range(C):
            freqs = rng.uniform(1.0, 40.0, size=(num_components,)).astype(np.float32)
            amps = rng.uniform(0.1, 1.0, size=(num_components,)).astype(np.float32)
            phases = rng.uniform(0.0, 2.0 * np.pi, size=(num_components,)).astype(np.float32)
            s = np.zeros_like(t)
            for (f, a, p) in zip(freqs, amps, phases):
                s += a * np.sin(2.0 * np.pi * f * t + p)
            x[c] += s.astype(np.float32)
    if mode in {'noise', 'mixed'}:
        x += rng.normal(loc=0.0, scale=noise_std, size=(C, T)).astype(np.float32)
    return x
__all__ = ['read_eeg', 'gen_eeg']
from __future__ import annotations
import os
import random
from typing import Callable, Optional, Tuple, Dict, List
import torch
from torch.utils.data import Dataset
from IO.reader import read_eeg, gen_eeg

class EEGDataset(Dataset):
    """
    Summary:
    A PyTorch dataset for loading pairs of EEG signals, comprising a potentially noisy attribute signal and a clean brain target signal, along with associated metadata.

    Description:
    WHY: This class facilitates machine learning tasks involving EEG signal processing, such as denoising or source separation. It provides a structured and efficient way to access raw EEG data, potentially augmented with various noise types, and their corresponding clean versions, preparing them for model input.
    WHEN: Use this class when you need to load EEG data from a structured directory (e.g., `root/split/Brain` for targets and `root/split/Category` for attributes) and prepare it for use with PyTorch's `DataLoader`. It is suitable for scenarios where a clear distinction between a "target" clean signal and a potentially "corrupted" attribute signal is required.
    WHERE: This class serves as the fundamental data loading component in a PyTorch-based machine learning pipeline. It acts as the interface between the raw EEG data files stored on disk and the model's training, validation, or evaluation loops.
    HOW: Upon initialization, it configures the dataset based on the provided `config` and `mode` (e.g., 'train', 'test'). It discovers available EEG filenames for the specified split and sets up paths to directories containing clean brain signals and various noise categories. For each requested sample, it loads a clean brain signal as the target. It then randomly selects a noise category and attempts to load a corresponding noisy signal as the attribute. If no specific noise file exists for the chosen category and sample, the clean brain signal is used as the attribute, and its category is marked 'Brain'. Both signals are loaded using an assumed `read_eeg` function (which handles formats like `.npy` or `.csv`), converted to PyTorch tensors, and then passed through optional `transform` and `target_transform` functions before being returned with relevant metadata.

    Parameters:
    config: Dict. A dictionary containing configuration parameters for the dataset.
        Constraints: Must include a `data.root` key specifying the base directory of the EEG dataset.
        Sub-keys:
        - `data.root` (required): A string path to the base directory where EEG data for all splits and categories is stored. Example: `/path/to/eeg_data/`.
        - `data.splits` (optional): A dictionary mapping `mode` strings ('train', 'val', 'test') to lists of filenames (e.g., `{'train': ['sample_0.npy', 'sample_1.npy']}`). If provided, these lists explicitly define which files belong to each split. If not provided, the dataset will discover files by listing the contents of the `Brain` directory for the given `mode`.
    mode: str, default 'train'. Specifies the dataset split to load, typically 'train', 'val', or 'test'. This determines which subdirectory within `root` (e.g., `root/train`) and which entry in `data.splits` (if provided) is used.
    transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]], default None. An optional callable function that accepts a tuple `(attr, target)` of `torch.Tensor`s and returns a transformed `(attr, target)` tuple. This transformation is applied jointly to both the attribute and target signals after they are loaded from disk and converted to tensors, but before `target_transform`.
    target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]], default None. An optional callable function that accepts a single `torch.Tensor` (the `target` signal) and returns a transformed `torch.Tensor`. This transformation is applied exclusively to the target signal after `transform`.
    seed: Optional[int], default None. An integer seed used to initialize the random number generator (`self.rng`). Providing a seed ensures reproducibility when randomly selecting noise categories for attribute signals.

    Attributes:
    root: str. The absolute path to the base directory where all EEG data is located.
    split: str. The current dataset split (e.g., 'train', 'val', 'test') that this instance represents.
    transform: Optional[Callable]. The transformation function applied to both attribute and target tensors, if specified during initialization.
    target_transform: Optional[Callable]. The transformation function applied only to the target tensor, if specified during initialization.
    rng: `random.Random`. An instance of Python's `random.Random` class, initialized with the provided `seed`, used for reproducible selection of noise categories.
    categories: List[str]. A list of strings representing the recognized noise categories (e.g., 'Brain', 'ChannelNoise', 'Eye', 'Heart', 'LineNoise', 'Muscle', 'Other'). 'Brain' is included as a category when the clean signal is used as the attribute.
    base_dir: str. The full path to the directory for the current split (e.g., `/path/to/eeg_data/train/`).
    brain_dir: str. The full path to the directory containing clean 'Brain' EEG signals for the current split (e.g., `/path/to/eeg_data/train/Brain/`).
    files: List[str]. A sorted list of filenames (e.g., `['sample_0.npy', 'sample_1.npy']`) that correspond to the EEG samples included in the current split of the dataset.

    Example:
    ```python
    import torch
    import numpy as np
    import os
    import shutil
    from typing import Dict, List, Optional, Callable, Tuple
    from torch.utils.data import Dataset, DataLoader

    # Assume read_eeg is a global or imported function that can load .npy files
    # (Mock implementation for demonstration purposes)
    def read_eeg(path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mock: File not found {path}")
        # Simulate loading an EEG array (e.g., 128 channels, 500 time steps)
        return np.load(path).astype(np.float32)

    # Define the EEGDataset class (as provided in FOCAL_CODE_COMPONENT)
    # --- Start of EEGDataset class definition ---
    class EEGDataset(Dataset):
        def __init__(self, *, config: Dict, mode: str='train', transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]=None, target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]]=None, seed: Optional[int]=None) -> None:
            super().__init__()
            cfg = config
            data_cfg = cfg.get('data', cfg)
            root = data_cfg.get('root')
            if not root:
                raise ValueError('Config must specify data.root')
            self.root = root
            self.split = mode
            self.transform = transform
            self.target_transform = target_transform
            self.rng = random.Random(seed)
            self.categories = ['Brain', 'ChannelNoise', 'Eye', 'Heart', 'LineNoise', 'Muscle', 'Other']
            self.base_dir = os.path.join(self.root, self.split)
            self.brain_dir = os.path.join(self.base_dir, 'Brain')
            if not os.path.isdir(self.brain_dir):
                raise FileNotFoundError(f'Brain directory not found: {self.brain_dir}')
            inline = data_cfg.get('splits') or cfg.get('splits')
            if isinstance(inline, dict) and mode in inline:
                self.files: List[str] = list(inline[mode])
            else:
                self.files = sorted(os.listdir(self.brain_dir))

        def __len__(self) -> int:
            return len(self.files)

        def __getitem__(self, index: int):
            fname = self.files[index]
            brain_path = os.path.join(self.brain_dir, fname)
            category = self.rng.choice(self.categories)
            noise_path = os.path.join(self.base_dir, category, fname)
            target_np = read_eeg(brain_path) # Assumes read_eeg is available
            if os.path.isfile(noise_path):
                attr_np = read_eeg(noise_path)
            else:
                attr_np = target_np
                category = 'Brain'
            target = torch.from_numpy(target_np).to(torch.float32)
            attr = torch.from_numpy(attr_np).to(torch.float32)
            if self.transform is not None:
                (attr, target) = self.transform(attr, target)
            if self.target_transform is not None:
                target = self.target_transform(target)
            meta: Dict[str, str] = {'filename': fname, 'category': category, 'split': self.split}
            return (attr, target, meta)
    # --- End of EEGDataset class definition ---


    # 1. Setup dummy data directories and files for the example
    dataset_root = './mock_eeg_data'
    train_dir = os.path.join(dataset_root, 'train')
    brain_train_dir = os.path.join(train_dir, 'Brain')
    noise_eye_train_dir = os.path.join(train_dir, 'Eye')
    noise_muscle_train_dir = os.path.join(train_dir, 'Muscle')

    # Ensure directories exist
    os.makedirs(brain_train_dir, exist_ok=True)
    os.makedirs(noise_eye_train_dir, exist_ok=True)
    os.makedirs(noise_muscle_train_dir, exist_ok=True)

    # Create dummy .npy files
    # 3 clean brain samples
    for i in range(3):
        dummy_eeg_data = np.random.rand(128, 500) * 10 # Simulate some EEG data
        np.save(os.path.join(brain_train_dir, f'sample_{i}.npy'), dummy_eeg_data)

    # Add some noise files for 'Eye' and 'Muscle'
    np.save(os.path.join(noise_eye_train_dir, 'sample_0.npy'), np.random.rand(128, 500) * 5)
    np.save(os.path.join(noise_muscle_train_dir, 'sample_1.npy'), np.random.rand(128, 500) * 7)

    # 2. Configuration for the dataset
    config = {
        'data': {
            'root': dataset_root
        }
    }

    # 3. Define a simple transformation (optional)
    def normalize_and_scale(attr: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Simple min-max normalization
        attr_min, attr_max = attr.min(), attr.max()
        target_min, target_max = target.min(), target.max()
        norm_attr = (attr - attr_min) / (attr_max - attr_min + 1e-6)
        norm_target = (target - target_min) / (target_max - target_min + 1e-6)
        return norm_attr * 2 - 1, norm_target * 2 - 1 # Scale to -1 to 1

    # 4. Initialize the dataset
    try:
        # Initialize with a transform and a seed for reproducibility of noise selection
        dataset = EEGDataset(config=config, mode='train', transform=normalize_and_scale, seed=42)
        print(f"Dataset initialized with {len(dataset)} samples.")

        # 5. Access a sample using __getitem__
        attr_tensor, target_tensor, meta_data = dataset[0] # Access the first sample
        print(f"\\nSample 0 details:")
        print(f"  Attribute tensor shape: {attr_tensor.shape}, dtype: {attr_tensor.dtype}")
        print(f"  Target tensor shape: {target_tensor.shape}, dtype: {target_tensor.dtype}")
        print(f"  Metadata: {meta_data}")
        # Check if transform was applied (e.g., values should be in [-1, 1])
        print(f"  Attribute range: [{attr_tensor.min():.2f}, {attr_tensor.max():.2f}]")
        print(f"  Target range: [{target_tensor.min():.2f}, {target_tensor.max():.2f}]")

        # Access another sample to see different noise categories (due to seed=42)
        attr_tensor_1, target_tensor_1, meta_data_1 = dataset[1]
        print(f"\\nSample 1 details:")
        print(f"  Metadata: {meta_data_1}")


        # 6. Use with PyTorch DataLoader for batching
        data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        print("\\nIterating through DataLoader for one batch:")
        for i, (attrs_batch, targets_batch, metas_batch) in enumerate(data_loader):
            print(f"Batch {i+1}:")
            print(f"  Attrs batch shape: {attrs_batch.shape}")    # (batch_size, channels, time_steps)
            print(f"  Targets batch shape: {targets_batch.shape}")
            print(f"  First filename in batch: {metas_batch['filename'][0]}")
            print(f"  Categories in batch: {metas_batch['category']}")
            if i == 0: # Process only the first batch for brevity
                break

    except (FileNotFoundError, ValueError) as e:
        print(f"Error during dataset initialization or access: {e}")
    finally:
        # 7. Clean up dummy data after example
        if os.path.exists(dataset_root):
            shutil.rmtree(dataset_root)
            print(f"\\nCleaned up dummy data at {dataset_root}")
    ```
    """

    def __init__(self, *, config: Dict, mode: str='train', transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]=None, target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]]=None, seed: Optional[int]=None) -> None:
        super().__init__()
        cfg = config
        data_cfg = cfg.get('data', cfg)
        root = data_cfg.get('root')
        if not root:
            raise ValueError('Config must specify data.root')
        self.root = root
        self.split = mode
        self.transform = transform
        self.target_transform = target_transform
        self.rng = random.Random(seed)
        self.categories = ['Brain', 'ChannelNoise', 'Eye', 'Heart', 'LineNoise', 'Muscle', 'Other']
        self.base_dir = os.path.join(self.root, self.split)
        self.brain_dir = os.path.join(self.base_dir, 'Brain')
        if not os.path.isdir(self.brain_dir):
            raise FileNotFoundError(f'Brain directory not found: {self.brain_dir}')
        inline = data_cfg.get('splits') or cfg.get('splits')
        if isinstance(inline, dict) and mode in inline:
            self.files: List[str] = list(inline[mode])
        else:
            self.files = sorted(os.listdir(self.brain_dir))

    def __len__(self) -> int:
        """
        Summary:
        Returns the total number of samples in the dataset.

        Description:
        WHY: This method is essential for making the dataset class compatible with Python's built-in `len()` function and PyTorch's `DataLoader`. It allows other parts of the framework to query the total size of the dataset.

        WHEN: This method is invoked implicitly whenever `len()` is called on an instance of the dataset, for example, `len(my_dataset)`. The `DataLoader` also uses it to determine how many batches to create and how to shuffle indices.

        WHERE: As a required special method for map-style PyTorch `Dataset` subclasses, it works in conjunction with `__getitem__` to define the dataset's size and accessibility.

        HOW: It calculates the dataset size by returning the length of the internal list `self.files`, where each entry corresponds to a single data sample.

        Returns:
        int: The total number of data samples available in the dataset.

        Examples:
        ```python
        # Assuming `dataset` is an initialized instance of this class
        # that has found 1000 data files.
        num_samples = len(dataset)
        print(f"Total samples in the dataset: {num_samples}")
        # Expected output:
        # Total samples in the dataset: 1000
        ```
        """
        return len(self.files)

    def __getitem__(self, index: int):
        """
        Summary:
        Fetches a data sample, including an attribute EEG, a target EEG, and corresponding metadata.

        Description:
        WHY: This method implements the core logic for accessing a single data point from the dataset. It is essential for compatibility with PyTorch's `DataLoader`, which uses it to create batches for training or evaluation.

        WHEN: This method is automatically called by a `DataLoader` when iterating over the dataset. It is generally not called directly by the user.

        WHERE: As the standard item accessor in a PyTorch `Dataset` subclass, it defines how a single sample is constructed from source files and prepared for a model.

        HOW: For a given index, the method first identifies the filename of the target "brain" EEG. It then randomly selects a noise category and attempts to find a corresponding noise file. If a noise file exists, it's loaded as the "attribute" signal. If not, the original "brain" signal is used as the attribute, and the category is set to 'Brain'. Both signals are loaded, converted to PyTorch tensors, and optionally passed through transformation functions before being returned along with metadata.

        Args:
        index (int): The index of the data sample to retrieve from the dataset.

        Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, str]]: A tuple containing three elements:
        - attr (torch.Tensor): The attribute tensor, representing either a noise-corrupted version of the signal or the clean signal itself.
        - target (torch.Tensor): The target tensor, representing the clean "brain" EEG signal.
        - meta (dict): A dictionary of metadata about the sample, including 'filename', 'category' (of the attribute), and 'split' (e.g., 'train', 'test').

        Raises:
        FileNotFoundError: If the target EEG file corresponding to the given index does not exist.
        IndexError: If the `index` is out of the valid range for the dataset.
        ValueError: If an underlying `read_eeg` call encounters an unsupported file format or malformed data.
        """
        fname = self.files[index]
        brain_path = os.path.join(self.brain_dir, fname)
        category = self.rng.choice(self.categories)
        noise_path = os.path.join(self.base_dir, category, fname)
        target_np = read_eeg(brain_path)
        if os.path.isfile(noise_path):
            attr_np = read_eeg(noise_path)
        else:
            attr_np = target_np
            category = 'Brain'
        target = torch.from_numpy(target_np).to(torch.float32)
        attr = torch.from_numpy(attr_np).to(torch.float32)
        if self.transform is not None:
            (attr, target) = self.transform(attr, target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        meta: Dict[str, str] = {'filename': fname, 'category': category, 'split': self.split}
        return (attr, target, meta)

class GenEEGDataset(Dataset):
    """
    Summary:
    A PyTorch dataset that dynamically generates pairs of synthetic EEG-like signals (an attribute and a target) on-the-fly, based on configurable parameters.

    Description:
    WHY: This class is designed to provide a flexible and reproducible source of synthetic EEG data for machine learning tasks, particularly for scenarios like denoising, signal transformation, or anomaly detection. It eliminates the need for large pre-generated datasets stored on disk, making it ideal for rapid prototyping, testing new algorithms, or scenarios where real EEG data is scarce or proprietary.
    WHEN: Use this dataset when you need a controlled and customizable stream of EEG-like data. It is particularly useful for debugging models, developing data augmentation techniques, or generating training/validation/test splits with well-defined signal characteristics and noise profiles without managing physical data files.
    WHERE: It serves as a data provider in a PyTorch machine learning pipeline, directly feeding `torch.utils.data.DataLoader` instances. It fits into the data preparation phase, preceding model definition and training loops.
    HOW: Upon initialization, it parses configuration parameters to determine the shape of the generated signals (channels `C`, time points `T`), the simulated `sample_rate`, and the total `length` of the dataset. Crucially, it defines two sets of generation specifications: `spec_attr` for the attribute signal and `spec_target` for the target signal. For each requested index, it internally calls an external `gen_eeg` function with these specifications and index-dependent seeds (for reproducibility) to create the attribute and target NumPy arrays. These are then converted to `torch.Tensor`s, passed through optional transformation functions, and returned along with relevant metadata.

    Parameters:
    config: Dict. A dictionary containing configuration parameters for the dataset.
        Constraints: Must be a dictionary.
        Sub-keys:
        - `data.splits` (or `cfg.splits`): A dictionary where keys are `mode` strings (e.g., 'train', 'val', 'test') and values are dictionaries defining generation parameters for that split.
            - `C` (int, default 30): Number of channels for the generated EEG signals.
            - `T` (int, default 1024): Number of time points for the generated EEG signals.
            - `sample_rate` (float, default 256.0): The simulated sampling rate in Hz.
            - `length` (int, default 1000): The total number of synthetic samples available in this dataset split.
            - `target` (Dict): Configuration for generating the target signal.
                - `mode` (str, default 'sine'): Generation mode for the target signal (e.g., 'sine', 'noise', 'mixed').
                - `noise_std` (float, default 0.0): Standard deviation of noise for the target signal.
                - `num_components` (int, default 3): Number of sine components for the target signal.
            - `attr` (Dict): Configuration for generating the attribute signal.
                - `mode` (str, default 'mixed'): Generation mode for the attribute signal (e.g., 'sine', 'noise', 'mixed').
                - `noise_std` (float, default 0.1): Standard deviation of noise for the attribute signal.
                - `num_components` (int, default 3): Number of sine components for the attribute signal.
    mode: str, default 'train'. Specifies the current dataset split (e.g., 'train', 'val', 'test'). This string is used to retrieve specific generation parameters from the `config`.
    transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]], default None. An optional callable that takes a tuple `(attr, target)` of `torch.Tensor`s and returns a transformed `(attr, target)` tuple. This is applied jointly to both signals after generation.
    target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]], default None. An optional callable that takes a single `torch.Tensor` (the `target` signal) and returns a transformed `torch.Tensor`. This is applied only to the target signal after `transform`.
    seed: Optional[int], default None. An integer seed used to initialize the base random number generation. When provided, this ensures reproducible synthetic data generation for each specific sample index.

    Attributes:
    C: int. The number of channels for the generated EEG signals.
    T: int. The number of time points for the generated EEG signals.
    sample_rate: float. The simulated sampling rate for the EEG signals.
    length: int. The total number of synthetic samples this dataset will yield.
    spec_target: Dict. A dictionary containing the generation parameters specific to the target signal.
    spec_attr: Dict. A dictionary containing the generation parameters specific to the attribute signal.
    transform: Optional[Callable]. The joint transformation function applied to `(attr, target)`, if provided.
    target_transform: Optional[Callable]. The transformation function applied only to `target`, if provided.
    mode: str. The current operational mode/split of the dataset (e.g., 'train', 'val', 'test').
    seed: Optional[int]. The base seed used for reproducible data generation.

    Example:
    ```python
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, Dataset
    from typing import Dict, Optional, Callable, Tuple
    import random

    # Assume gen_eeg is available (as defined in previous contexts)
    def gen_eeg(C: int=32, T: int=1024, *, sample_rate: float=256.0, mode: str='mixed', noise_std: float=0.1, num_components: int=3, seed: Optional[int]=None) -> np.ndarray:
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

    # Define a simple transform for demonstration
    def double_amplitude(attr: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return attr * 2.0, target * 2.0

    # 1. Configuration dictionary for train and validation splits
    config_data = {
        'data': {
            'splits': {
                'train': {
                    'C': 30, 'T': 512, 'sample_rate': 250.0, 'length': 1000,
                    'target': {'mode': 'sine', 'noise_std': 0.0, 'num_components': 2}, # Clean target
                    'attr': {'mode': 'mixed', 'noise_std': 0.2, 'num_components': 3}    # Noisy attribute
                },
                'val': {
                    'C': 30, 'T': 512, 'sample_rate': 250.0, 'length': 200,
                    'target': {'mode': 'sine', 'noise_std': 0.0, 'num_components': 2},
                    'attr': {'mode': 'mixed', 'noise_std': 0.2, 'num_components': 3}
                }
            }
        }
    }

    # 2. Initialize the training dataset
    train_dataset = GenEEGDataset(config=config_data, mode='train', transform=double_amplitude, seed=42)
    print(f"Train Dataset initialized with {len(train_dataset)} samples.")
    print(f"Train Dataset (C, T): ({train_dataset.C}, {train_dataset.T})")
    print(f"Train Dataset Target Spec: {train_dataset.spec_target}")
    print(f"Train Dataset Attr Spec: {train_dataset.spec_attr}")

    # 3. Access a sample and check its properties
    attr_train, target_train, meta_train = train_dataset[0]
    print(f"\\nFirst train sample:")
    print(f"  Attribute shape: {attr_train.shape}, Target shape: {target_train.shape}")
    print(f"  Attribute range: [{attr_train.min():.2f}, {attr_train.max():.2f}] (after transform)")
    print(f"  Target range: [{target_train.min():.2f}, {target_train.max():.2f}] (after transform)")
    print(f"  Metadata: {meta_train}")
    assert meta_train['generated'] is True
    assert meta_train['split'] == 'train'
    assert meta_train['index'] == 0

    # 4. Initialize the validation dataset (without transform)
    val_dataset = GenEEGDataset(config=config_data, mode='val', seed=42)
    print(f"\\nValidation Dataset initialized with {len(val_dataset)} samples.")
    attr_val, target_val, meta_val = val_dataset[0]
    print(f"First validation sample:")
    print(f"  Attribute range: [{attr_val.min():.2f}, {attr_val.max():.2f}] (no transform)")

    # 5. Use with PyTorch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    print(f"\\nTrain DataLoader has {len(train_loader)} batches.")

    for batch_idx, (attrs, targets, metas) in enumerate(train_loader):
        print(f"  Batch {batch_idx+1}: attrs_batch_shape={attrs.shape}, targets_batch_shape={targets.shape}")
        print(f"  Batch {batch_idx+1} meta (first index): {metas['index'][0]}, (first split): {metas['split'][0]}")
        if batch_idx == 0:
            break # Just show the first batch
    ```
    """

    def __init__(self, *, config: Dict, mode: str='train', transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]=None, target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]]=None, seed: Optional[int]=None) -> None:
        super().__init__()
        cfg = config
        data_cfg = cfg.get('data', cfg)
        gen_cfg = data_cfg.get('splits') or cfg.get('splits') or {}
        split_cfg: Dict = gen_cfg.get(mode, {}) if isinstance(gen_cfg, dict) else {}
        self.C = int(split_cfg.get('C', 30))
        self.T = int(split_cfg.get('T', 1024))
        self.sample_rate = float(split_cfg.get('sample_rate', 256.0))
        self.length = int(split_cfg.get('length', 1000))
        self.spec_target: Dict = {'mode': split_cfg.get('target', {}).get('mode', 'sine'), 'noise_std': float(split_cfg.get('target', {}).get('noise_std', 0.0)), 'num_components': int(split_cfg.get('target', {}).get('num_components', 3))}
        self.spec_attr: Dict = {'mode': split_cfg.get('attr', {}).get('mode', 'mixed'), 'noise_std': float(split_cfg.get('attr', {}).get('noise_std', 0.1)), 'num_components': int(split_cfg.get('attr', {}).get('num_components', 3))}
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.seed = seed

    def __len__(self) -> int:
        """
        Summary:
        Returns the total number of synthetic samples available in the dataset.

        Description:
        WHY: This method is a standard requirement for PyTorch `Dataset` subclasses, enabling compatibility with Python's built-in `len()` function and facilitating proper operation with `torch.utils.data.DataLoader`. It allows external components to query the size of the dataset.
        WHEN: This method is implicitly invoked whenever `len()` is called on an instance of this dataset class (e.g., `len(my_dataset)`). It is also used by `DataLoader` to determine iteration counts and for batching strategies.
        WHERE: As a mandatory special method for map-style PyTorch `Dataset` subclasses, it works in conjunction with `__getitem__` to define the dataset's overall structure and accessibility.
        HOW: It directly returns the value of the `self.length` attribute, which is pre-configured during the dataset's initialization to represent the desired total number of synthetic samples.

        Returns:
        int: The total count of synthetic data samples that can be generated by this dataset.

        Examples:
        ```python
        import torch
        # Assuming SyntheticEEGDataset is properly defined and imported
        # and `dataset` is an initialized instance of SyntheticEEGDataset.

        # Example: Initialize a dummy dataset for demonstration
        from typing import Dict, Optional, Callable, Tuple
        class SyntheticEEGDataset(torch.utils.data.Dataset):
            def __init__(self, length: int):
                self.length = length
            def __len__(self) -> int:
                return self.length
            def __getitem__(self, index: int):
                # Placeholder for __getitem__ implementation
                return torch.randn(1, 100), torch.randn(1, 100), {'index': index}

        dataset = SyntheticEEGDataset(length=500)

        # Get the length of the dataset
        num_samples = len(dataset)
        print(f"Total synthetic samples in the dataset: {num_samples}")
        # Expected output:
        # Total synthetic samples in the dataset: 500

        # This is also used by DataLoader:
        from torch.utils.data import DataLoader
        data_loader = DataLoader(dataset, batch_size=32)
        print(f"Number of batches for batch_size=32: {len(data_loader)}")
        # Expected output for 500 samples, batch_size=32:
        # Number of batches for batch_size=32: 16 (since 500/32 = 15.625, ceil is 16)
        ```
        """
        return self.length

    def __getitem__(self, index: int):
        '''
        Summary:
        Generates a pair of synthetic EEG-like signals (an attribute and a target) and associated metadata for a given index.

        Description:
        WHY: This method is the core access point for retrieving individual data samples from the synthetic dataset. It enables the creation of reproducible pairs of EEG signals with distinct characteristics (e.g., clean target and noisy attribute) on-the-fly, which is crucial for training and evaluating models that learn to process or transform EEG data.
        WHEN: This method is automatically invoked by PyTorch's `DataLoader` when iterating over the dataset to fetch individual samples or batches. It should generally not be called directly by users.
        WHERE: As a fundamental method of a `torch.utils.data.Dataset` subclass, it forms the data provision layer, preparing synthetic EEG signals for consumption by a PyTorch model in a training or evaluation loop.
        HOW: For a given `index`, it calculates unique random seeds for both the attribute and target signals to ensure reproducibility of each sample. It then calls the `gen_eeg` function twice, once for the target signal using `self.spec_target` parameters, and once for the attribute signal using `self.spec_attr` parameters. The resulting NumPy arrays are converted to `torch.Tensor`s, optionally passed through `self.transform` (for joint transformations) and `self.target_transform` (for target-specific transformations), and finally returned along with a metadata dictionary indicating its synthetic origin and index.

        Args:
        index: int. The integer index of the synthetic data sample to retrieve.
            Constraints: Must be a non-negative integer within the bounds defined by `__len__`.

        Returns:
        tuple[torch.Tensor, torch.Tensor, Dict[str, object]]: A tuple containing three elements:
        - attr (torch.Tensor): The generated synthetic attribute signal, a `torch.float32` tensor of shape `(self.C, self.T)`.
        - target (torch.Tensor): The generated synthetic target signal, a `torch.float32` tensor of shape `(self.C, self.T)`.
        - meta (Dict[str, object]): A dictionary containing metadata, including:
            - 'generated': True (indicates the data is synthetic).
            - 'split': The dataset's mode (e.g., 'train', 'val').
            - 'index': The index of the retrieved sample.

        Raises:
        ValueError: If the `mode` specified in either `self.spec_attr` or `self.spec_target` is not one of 'sine', 'noise', or 'mixed', as propagated from the `gen_eeg` function.
        IndexError: If the provided `index` is out of the valid range for the dataset (implicitly handled by `DataLoader` and `__len__`).

        Examples:
        ```python
        import torch
        import numpy as np
        import random
        from typing import Dict, Optional, Callable, Tuple

        # Mock gen_eeg function and a simplified SyntheticEEGDataset for demonstration
        # In a real scenario, gen_eeg and the full SyntheticEEGDataset would be imported.

        def gen_eeg(C: int=32, T: int=1024, *, sample_rate: float=256.0, mode: str='mixed', noise_std: float=0.1, num_components: int=3, seed: Optional[int]=None) -> np.ndarray:
            """Mock function: Generates simple synthetic EEG data."""
            rng = np.random.default_rng(seed)
            x = rng.rand(C, T).astype(np.float32)
            if mode == 'sine':
                x += np.sin(np.linspace(0, 2 * np.pi * num_components, T)) * rng.rand()
            if mode == 'noise':
                x += rng.normal(loc=0.0, scale=noise_std, size=(C, T))
            if mode == 'mixed':
                x += np.sin(np.linspace(0, 2 * np.pi * num_components, T)) * rng.rand()
                x += rng.normal(loc=0.0, scale=noise_std, size=(C, T))
            return x

        class SyntheticEEGDataset(torch.utils.data.Dataset):
            def __init__(self, C: int, T: int, num_samples: int, sample_rate: float,
                         spec_attr: Dict, spec_target: Dict,
                         mode: str = 'train', seed: Optional[int] = None,
                         transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
                         target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
                self.C = C
                self.T = T
                self.num_samples = num_samples
                self.sample_rate = sample_rate
                self.spec_attr = spec_attr
                self.spec_target = spec_target
                self.mode = mode
                self.seed = seed
                self.transform = transform
                self.target_transform = target_transform

            def __len__(self) -> int:
                return self.num_samples

            # FOCAL_CODE_COMPONENT would be here
            def __getitem__(self, index: int):
                s_attr = None if self.seed is None else self.seed * 100003 + index
                s_tgt = None if self.seed is None else self.seed * 100019 + index
                target_np = gen_eeg(C=self.C, T=self.T, sample_rate=self.sample_rate, mode=self.spec_target['mode'], noise_std=self.spec_target['noise_std'], num_components=self.spec_target['num_components'], seed=s_tgt)
                attr_np = gen_eeg(C=self.C, T=self.T, sample_rate=self.sample_rate, mode=self.spec_attr['mode'], noise_std=self.spec_attr['noise_std'], num_components=self.spec_attr['num_components'], seed=s_attr)
                target = torch.from_numpy(target_np).to(torch.float32)
                attr = torch.from_numpy(attr_np).to(torch.float32)
                if self.transform is not None:
                    (attr, target) = self.transform(attr, target)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                meta: Dict[str, object] = {'generated': True, 'split': self.mode, 'index': index}
                return (attr, target, meta)
            # End of FOCAL_CODE_COMPONENT

        # 1. Define specifications for attribute and target signals
        attribute_spec = {'mode': 'mixed', 'noise_std': 0.5, 'num_components': 3}
        target_spec = {'mode': 'sine', 'noise_std': 0.0, 'num_components': 3} # Clean sine waves

        # 2. Initialize the dataset
        dataset = SyntheticEEGDataset(
            C=16, T=512, num_samples=100, sample_rate=256.0,
            spec_attr=attribute_spec, spec_target=target_spec,
            mode='train', seed=123
        )

        print(f"Dataset initialized with {len(dataset)} samples.")

        # 3. Access a sample using __getitem__
        attr_signal, target_signal, metadata = dataset[0]

        print(f"\\nSample 0 details:")
        print(f"  Attribute signal shape: {attr_signal.shape}, dtype: {attr_signal.dtype}")
        print(f"  Target signal shape: {target_signal.shape}, dtype: {target_signal.dtype}")
        print(f"  Metadata: {metadata}")
        print(f"  Attribute signal (first 5 values of first channel): {attr_signal[0, :5].numpy()}")
        print(f"  Target signal (first 5 values of first channel): {target_signal[0, :5].numpy()}")


        # 4. Use with DataLoader (typical usage)
        from torch.utils.data import DataLoader

        data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

        print("\\nIterating through DataLoader (first batch):")
        for i, (attrs_batch, targets_batch, metas_batch) in enumerate(data_loader):
            print(f"  Batch {i+1} - Attributes shape: {attrs_batch.shape}, Targets shape: {targets_batch.shape}")
            print(f"  Batch {i+1} - Metadata (first sample's index): {metas_batch['index'][0]}")
            if i == 0: # Just show the first batch
                break

        # 5. Demonstrate reproducibility (if seed is set)
        # Get sample 5 twice
        attr_5_a, target_5_a, _ = dataset[5]
        attr_5_b, target_5_b, _ = dataset[5]
        assert torch.equal(attr_5_a, attr_5_b)
        assert torch.equal(target_5_a, target_5_b)
        print("\\nSample 5 generated twice with same seed is identical.")


        # 6. Example with a transform
        def custom_transform(attr: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # Simple scaling: scale both signals by 0.5
            return attr * 0.5, target * 0.5

        dataset_with_transform = SyntheticEEGDataset(
            C=8, T=256, num_samples=10, sample_rate=128.0,
            spec_attr=attribute_spec, spec_target=target_spec,
            transform=custom_transform, seed=42
        )
        attr_scaled, target_scaled, _ = dataset_with_transform[0]
        print(f"\\nSample 0 with transform applied:")
        print(f"  Attribute values (mean, max): {attr_scaled.mean():.2f}, {attr_scaled.max():.2f}")
        print(f"  Target values (mean, max): {target_scaled.mean():.2f}, {target_scaled.max():.2f}")
        ```
        '''
        s_attr = None if self.seed is None else self.seed * 100003 + index
        s_tgt = None if self.seed is None else self.seed * 100019 + index
        target_np = gen_eeg(C=self.C, T=self.T, sample_rate=self.sample_rate, mode=self.spec_target['mode'], noise_std=self.spec_target['noise_std'], num_components=self.spec_target['num_components'], seed=s_tgt)
        attr_np = gen_eeg(C=self.C, T=self.T, sample_rate=self.sample_rate, mode=self.spec_attr['mode'], noise_std=self.spec_attr['noise_std'], num_components=self.spec_attr['num_components'], seed=s_attr)
        target = torch.from_numpy(target_np).to(torch.float32)
        attr = torch.from_numpy(attr_np).to(torch.float32)
        if self.transform is not None:
            (attr, target) = self.transform(attr, target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        meta: Dict[str, object] = {'generated': True, 'split': self.mode, 'index': index}
        return (attr, target, meta)

def build_dataset_from_config(*, cfg: Dict, mode: str='train', transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]=None, target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]]=None, seed: Optional[int]=None):
    """
    <DOCSTRING>
    Summary:
    Constructs and returns an appropriate EEG dataset instance, dynamically choosing between a file-based or a synthetic generator based on configuration and data availability.

    Description:
    WHY: This function provides a unified entry point for obtaining an EEG dataset, abstracting away the underlying data source. It simplifies data pipeline setup by allowing seamless switching between loading real data from disk and generating synthetic data on-the-fly, which is valuable for development, testing, and handling scenarios with limited real data.
    WHEN: Use this function as the primary way to instantiate your EEG dataset. It's ideal when your application needs to adapt to different data availability scenarios (e.g., local files vs. no files, needing synthetic data for debugging) without changing the dataset instantiation logic.
    WHERE: This function typically resides in a data preparation or utility module. It serves as a factory function that produces a `torch.utils.data.Dataset` object, which is then passed to a `torch.utils.data.DataLoader` for batching in a machine learning training or evaluation loop.
    HOW: It first attempts to retrieve a `data.root` path from the provided `cfg`. If `data.root` is a string and the specific `Brain` subdirectory for the given `mode` (e.g., `root/train/Brain`) exists on the filesystem, it infers that real EEG data is available and returns an instance of `EEGDataset`. Otherwise (if `data.root` is not a string, or the specified directory does not exist), it falls back to generating synthetic data by returning an instance of `GenEEGDataset`. All other parameters like `mode`, `transform`, `target_transform`, and `seed` are consistently passed to the chosen dataset constructor.

    Args:
    cfg: Dict. The configuration dictionary containing all necessary parameters for dataset construction.
        Constraints: Must be a dictionary. Expected to contain a `data` key, which itself may contain a `root` key (string path) or `splits` key (dict of generation parameters for `GenEEGDataset`).
    mode: str, default 'train'. The operational mode (e.g., 'train', 'val', 'test') for which the dataset is being built. This affects the directory path lookup for file-based datasets and parameter selection for synthetic datasets.
    transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]], default None. An optional callable function that applies transformations jointly to both the attribute and target `torch.Tensor`s after they are loaded or generated.
    target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]], default None. An optional callable function that applies transformations specifically to the target `torch.Tensor` after `transform` (if any).
    seed: Optional[int], default None. An integer seed used for reproducibility. This seed is passed directly to the chosen dataset constructor to ensure consistent data loading/generation.

    Returns:
    torch.utils.data.Dataset: An initialized dataset object, which will be either an `EEGDataset` (for file-based data) or a `GenEEGDataset` (for synthetic data), ready for use with a `DataLoader`.

    Raises:
    This function itself does not directly raise exceptions. However, the instantiated `EEGDataset` or `GenEEGDataset` constructors may raise:
    - ValueError: If critical configuration keys are missing or invalid within the `cfg` for the selected dataset type.
    - FileNotFoundError: If an `EEGDataset` is selected but the specified `data.root` path or its required subdirectories do not exist or are inaccessible.
    These exceptions will propagate from the dataset constructor calls.

    Examples:
    ```python
    import torch
    import os
    import shutil
    import numpy as np
    import random
    from typing import Dict, Optional, Callable, Tuple
    from torch.utils.data import Dataset, DataLoader

    # --- Mock Dataset Implementations for Example ---
    # In a real scenario, EEGDataset and GenEEGDataset would be imported.

    # Mock EEGDataset (simplified for example)
    class EEGDataset(Dataset):
        def __init__(self, *, config: Dict, mode: str='train', transform=None, target_transform=None, seed=None) -> None:
            _cfg = config.get('data', config)
            self.root = _cfg.get('root')
            self.mode = mode
            self.transform = transform
            self.target_transform = target_transform
            self.seed = seed
            self.files = sorted(os.listdir(os.path.join(self.root, mode, 'Brain')))
            print(f"Mock EEGDataset initialized for '{mode}' with {len(self.files)} files from {self.root}")

        def __len__(self) -> int:
            return len(self.files)

        def __getitem__(self, index: int):
            # Simulate loading real data
            dummy_attr = torch.randn(32, 1024, dtype=torch.float32) + (index * 0.01)
            dummy_target = torch.randn(32, 1024, dtype=torch.float32)
            if self.transform:
                dummy_attr, dummy_target = self.transform(dummy_attr, dummy_target)
            if self.target_transform:
                dummy_target = self.target_transform(dummy_target)
            return dummy_attr, dummy_target, {'filename': self.files[index], 'split': self.mode, 'generated': False}

    # Mock gen_eeg function (as provided in previous contexts)
    def gen_eeg(C: int=32, T: int=1024, *, sample_rate: float=256.0, mode: str='mixed', noise_std: float=0.1, num_components: int=3, seed: Optional[int]=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.rand(C, T).astype(np.float32) # Simplified for example
        if 'sine' in mode:
            x += np.sin(np.linspace(0, 2 * np.pi * num_components, T)) * rng.rand()
        if 'noise' in mode:
            x += rng.normal(loc=0.0, scale=noise_std, size=(C, T))
        return x

    # Mock GenEEGDataset (simplified for example)
    class GenEEGDataset(Dataset):
        def __init__(self, *, config: Dict, mode: str='train', transform=None, target_transform=None, seed=None) -> None:
            _cfg = config.get('data', config)
            gen_cfg = _cfg.get('splits', {}).get(mode, {})
            self.C = int(gen_cfg.get('C', 30))
            self.T = int(gen_cfg.get('T', 1024))
            self.length = int(gen_cfg.get('length', 100))
            self.sample_rate = float(gen_cfg.get('sample_rate', 256.0))
            self.spec_attr = gen_cfg.get('attr', {'mode': 'mixed', 'noise_std': 0.1, 'num_components': 3})
            self.spec_target = gen_cfg.get('target', {'mode': 'sine', 'noise_std': 0.0, 'num_components': 3})
            self.transform = transform
            self.target_transform = target_transform
            self.mode = mode
            self.seed = seed
            print(f"Mock GenEEGDataset initialized for '{mode}' with {self.length} synthetic samples.")

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, index: int):
            # Simulate generating data
            s_attr = None if self.seed is None else self.seed * 100003 + index
            s_tgt = None if self.seed is None else self.seed * 100019 + index
            attr_np = gen_eeg(C=self.C, T=self.T, sample_rate=self.sample_rate, **self.spec_attr, seed=s_attr)
            target_np = gen_eeg(C=self.C, T=self.T, sample_rate=self.sample_rate, **self.spec_target, seed=s_tgt)

            attr = torch.from_numpy(attr_np).to(torch.float32)
            target = torch.from_numpy(target_np).to(torch.float32)

            if self.transform:
                attr, target = self.transform(attr, target)
            if self.target_transform:
                target = self.target_transform(target)
            return attr, target, {'generated': True, 'split': self.mode, 'index': index}

    # --- Focal function definition ---
    def build_dataset_from_config(*, cfg: Dict, mode: str='train', transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]=None, target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]]=None, seed: Optional[int]=None):
        data_cfg = cfg.get('data', cfg)
        root = data_cfg.get('root')
        if isinstance(root, str):
            base_dir = os.path.join(root, mode, 'Brain')
            if os.path.isdir(base_dir):
                return EEGDataset(config=cfg, mode=mode, transform=transform, target_transform=target_transform, seed=seed)
        return GenEEGDataset(config=cfg, mode=mode, transform=transform, target_transform=target_transform, seed=seed)
    # --- End of Focal function definition ---

    # Setup for Examples
    DUMMY_DATA_ROOT = './temp_eeg_data'
    TRAIN_BRAIN_DIR = os.path.join(DUMMY_DATA_ROOT, 'train', 'Brain')

    def setup_dummy_eeg_files():
        os.makedirs(TRAIN_BRAIN_DIR, exist_ok=True)
        for i in range(5): # Create 5 dummy files
            np.save(os.path.join(TRAIN_BRAIN_DIR, f'sample_{i}.npy'), np.random.rand(32, 1024))

    def cleanup_dummy_eeg_files():
        if os.path.exists(DUMMY_DATA_ROOT):
            shutil.rmtree(DUMMY_DATA_ROOT)

    # Example Transform
    def simple_scaling_transform(attr: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return attr * 0.1, target * 0.1

    print("--- Scenario 1: File-based EEGDataset is chosen (data.root exists) ---")
    cleanup_dummy_eeg_files()
    setup_dummy_eeg_files()

    config_file_based = {
        'data': {
            'root': DUMMY_DATA_ROOT,
            'some_other_setting': True
        }
    }
    dataset_file = build_dataset_from_config(cfg=config_file_based, mode='train', transform=simple_scaling_transform, seed=123)
    print(f"Type of dataset created: {type(dataset_file).__name__}")
    print(f"Number of samples: {len(dataset_file)}")

    # Verify a sample from the file-based dataset
    attr_f, target_f, meta_f = dataset_file[0]
    print(f"Sample 0 from file-based dataset: generated={meta_f['generated']}, filename={meta_f['filename']}, attr_shape={attr_f.shape}")
    print(f"  Attr range after transform: [{attr_f.min():.2f}, {attr_f.max():.2f}]")

    cleanup_dummy_eeg_files() # Clean up for the next scenario

    print("\\n--- Scenario 2: Synthetic GenEEGDataset is chosen (data.root missing or invalid) ---")
    config_synthetic = {
        'data': {
            # 'root' is intentionally missing or points to a non-existent dir
            'splits': {
                'train': {
                    'C': 16, 'T': 256, 'length': 20,
                    'target': {'mode': 'sine', 'noise_std': 0.0},
                    'attr': {'mode': 'mixed', 'noise_std': 0.3}
                }
            }
        }
    }
    dataset_synth = build_dataset_from_config(cfg=config_synthetic, mode='train', seed=456)
    print(f"Type of dataset created: {type(dataset_synth).__name__}")
    print(f"Number of samples: {len(dataset_synth)}")

    # Verify a sample from the synthetic dataset
    attr_s, target_s, meta_s = dataset_synth[0]
    print(f"Sample 0 from synthetic dataset: generated={meta_s['generated']}, index={meta_s['index']}, attr_shape={attr_s.shape}")

    print("\\n--- Scenario 3: Using with DataLoader ---")
    # Using the synthetic dataset for DataLoader
    data_loader = DataLoader(dataset_synth, batch_size=4, shuffle=True)
    for i, (attrs_batch, targets_batch, metas_batch) in enumerate(data_loader):
        print(f"  Batch {i+1}: Attrs shape={attrs_batch.shape}, Targets shape={targets_batch.shape}")
        print(f"  Batch {i+1} Metadata (first index): {metas_batch['index'][0]}, (first 'generated'): {metas_batch['generated'][0]}")
        if i == 0:
            break # Only show the first batch

    ```
    """
    data_cfg = cfg.get('data', cfg)
    root = data_cfg.get('root')
    if isinstance(root, str):
        base_dir = os.path.join(root, mode, 'Brain')
        if os.path.isdir(base_dir):
            return EEGDataset(config=cfg, mode=mode, transform=transform, target_transform=target_transform, seed=seed)
    return GenEEGDataset(config=cfg, mode=mode, transform=transform, target_transform=target_transform, seed=seed)
__all__ = ['EEGDataset', 'GenEEGDataset', 'build_dataset_from_config']
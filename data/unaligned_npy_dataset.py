import os.path
import numpy as np
import torch
from data.base_dataset import BaseDataset # Removed get_transform as it's unused here
# from data.image_folder import make_dataset # No longer using this
import random
import glob # <-- Add this import


class UnalignedNpyDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets stored as .npy files.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB'.
    Domain A and B identities are assigned based on the sorting order of the filenames.

    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if not os.path.exists(self.dir_A):
             raise FileNotFoundError(f"Dataset directory not found: {self.dir_A}")
        if not os.path.exists(self.dir_B):
             raise FileNotFoundError(f"Dataset directory not found: {self.dir_B}")

        # --- Modification Start ---
        # Use glob to find all .npy files directly
        self.A_paths = sorted(glob.glob(os.path.join(self.dir_A, '*.npy')))
        self.B_paths = sorted(glob.glob(os.path.join(self.dir_B, '*.npy')))

        # Apply max_dataset_size limit if specified
        if opt.max_dataset_size != float('inf'):
             self.A_paths = self.A_paths[:int(opt.max_dataset_size)] # Ensure index is integer
             self.B_paths = self.B_paths[:int(opt.max_dataset_size)]
        # --- Modification End ---


        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        if self.A_size == 0:
            print(f"Warning: Found 0 *.npy files in {self.dir_A}")
        if self.B_size == 0:
             print(f"Warning: Found 0 *.npy files in {self.dir_B}")

        print(f"Found {self.A_size} files for A and {self.B_size} files for B.")

        # No image transforms like color jitter, resize, or flip needed for preprocessed .npy
        # We only need to ensure the data is a tensor.
        # Input channels should match --input_nc and --output_nc (which we set to 2)
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        if input_nc != 2 or output_nc != 2:
            print(f"Warning: input_nc ({input_nc}) or output_nc ({output_nc}) is not 2. Ensure model config matches data.")

    # __getitem__ and __len__ methods remain the same as before...
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image path related to A
            B_paths (str)    -- image path related to B
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # Load .npy files
        try:
             A_npy = np.load(A_path)
             B_npy = np.load(B_path)
        except Exception as e:
             print(f"Error loading .npy file:")
             print(f"  A_path: {A_path}")
             print(f"  B_path: {B_path}")
             print(f"  Error: {e}")
             # Return dummy data or raise error? For now, let's raise it.
             raise e


        # Convert to PyTorch tensors
        # Input shape should be [C, H, W] = [2, Freq, Time]
        A = torch.from_numpy(A_npy).float()
        B = torch.from_numpy(B_npy).float()

        # Ensure they have the correct shape (C=2)
        if A.shape[0] != 2:
             raise ValueError(f"Tensor A from {A_path} has shape {A.shape}, expected C=2")
        if B.shape[0] != 2:
            raise ValueError(f"Tensor B from {B_path} has shape {B.shape}, expected C=2")
        #assert A.shape[0] == 2, f"Tensor A from {A_path} has shape {A.shape}, expected C=2" # Use ValueError for clarity
        #assert B.shape[0] == 2, f"Tensor B from {B_path} has shape {B.shape}, expected C=2" # Use ValueError for clarity


        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different sizes,
        we take the maximum of the two sets.
        """
        # Ensure we don't return 0 if one dataset is empty but we want unpaired items
        if self.A_size == 0 or self.B_size == 0:
             # If one is empty, the effective size for unaligned is 0 unless testing/single image mode
             # However, CycleGAN needs samples from both. Let's warn earlier.
             # For training, return max might still be okay if user ignores warnings,
             # but it will likely fail later. Returning max is consistent with original logic.
             pass # Warning printed in __init__
        return max(self.A_size, self.B_size)
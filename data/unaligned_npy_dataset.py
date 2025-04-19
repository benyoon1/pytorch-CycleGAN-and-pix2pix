import os.path
import numpy as np
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import random


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

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size, extensions=['.npy']))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size, extensions=['.npy']))   # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        print(f"Found {self.A_size} files for A and {self.B_size} files for B.")

        # No image transforms like color jitter, resize, or flip needed for preprocessed .npy
        # We only need to ensure the data is a tensor.
        # Input channels should match --input_nc and --output_nc (which we set to 2)
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        if input_nc != 2 or output_nc != 2:
            print(f"Warning: input_nc ({input_nc}) or output_nc ({output_nc}) is not 2. Ensure model config matches data.")

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
        A_npy = np.load(A_path)
        B_npy = np.load(B_path)

        # Convert to PyTorch tensors
        # Input shape should be [C, H, W] = [2, Freq, Time]
        A = torch.from_numpy(A_npy).float()
        B = torch.from_numpy(B_npy).float()

        # Ensure they have the correct shape (C=2)
        assert A.shape[0] == 2, f"Tensor A from {A_path} has shape {A.shape}, expected C=2"
        assert B.shape[0] == 2, f"Tensor B from {B_path} has shape {B.shape}, expected C=2"

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different sizes,
        we take the maximum of the two sets.
        """
        return max(self.A_size, self.B_size)
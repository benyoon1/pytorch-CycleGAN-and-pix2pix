import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os
import random
from tqdm.auto import tqdm
import itertools
import soundfile as sf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0002
BETA1 = 0.5           # Adam optimizer beta1

# --- Data Loading ---
class AudioSegmentDataset(Dataset):
    def __init__(self, data_dir, segment_length, sample_rate):
        """
        Dataset for loading audio segments from a directory.
        Args:
            data_dir (str): Directory containing audio segment files.
            segment_length (int): Length of each audio segment in samples.
            sample_rate (int): Sample rate for loading audio files.
        """
        self.data_dir = data_dir
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        try:
            self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.npy', '.wav'))]
            if not self.file_list:
                 print(f"Warning: No '.npy' or '.wav' files found in {data_dir}")
            else:
                print(f"Found {len(self.file_list)} segments in {data_dir}")
        except FileNotFoundError:
            print(f"Error: Data directory not found: {data_dir}")
            self.file_list = [] # Initialize as empty list to avoid errors later

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        try:
            if filepath.endswith('.npy'):
                segment = np.load(filepath).astype(np.float32) # Ensure float32
            elif filepath.endswith('.wav'):
                segment, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
                if sr != self.sample_rate:
                     print(f"Warning: File {filepath} has sr {sr}, resampling to {self.sample_rate}")
                     segment = librosa.resample(segment, orig_sr=sr, target_sr=self.sample_rate)
                segment = segment.astype(np.float32) # Ensure float32

            # Ensure segment length consistency (padding or trimming) - should mostly be handled by prep script
            if len(segment) > self.segment_length:
                # Take a random crop if longer
                start = random.randint(0, len(segment) - self.segment_length)
                segment = segment[start : start + self.segment_length]
            elif len(segment) < self.segment_length:
                # Pad if shorter
                segment = np.pad(segment, (0, self.segment_length - len(segment)), 'constant')

            # Convert to torch tensor and add channel dimension
            segment_tensor = torch.from_numpy(segment).unsqueeze(0) # Shape: [1, segment_length]
            return segment_tensor

        except Exception as e:
            print(f"Error loading or processing file {filepath}: {e}")
            # Return a dummy tensor of zeros if loading fails
            return torch.zeros((1, self.segment_length), dtype=torch.float32)
        

# Buffers for fake samples (helps stabilize training)
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data: # Iterate over samples in the batch
            element = element.unsqueeze(0) # Add batch dim: [1, C, L]
            if len(self.data) < self.max_size:
                self.data.append(element.clone().cpu()) # Store on CPU to save GPU memory
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5: # 50% chance to replace existing
                    i = random.randint(0, self.max_size - 1)
                    # Need to move the popped element to the correct device
                    to_return.append(self.data[i].clone().to(data.device))
                    self.data[i] = element.clone().cpu()
                else: # 50% chance to just return the new element
                    to_return.append(element)
        return torch.cat(to_return) # Concatenate along batch dim


# --- Model Architectures ---

# Basic Residual Block for Generator
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm1d(channels), # affine=True by default
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm1d(channels) # affine=True by default
        )

    def forward(self, x):
        return x + self.conv_block(x)

# Generator (1D U-Net like structure)
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_n_filters=64, n_residual_blocks=6):
        super(Generator, self).__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, base_n_filters, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm1d(base_n_filters, affine=True), # Explicitly state affine=True
            nn.ReLU(inplace=True)
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv1d(base_n_filters, base_n_filters*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(base_n_filters*2, affine=True),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(base_n_filters*2, base_n_filters*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(base_n_filters*4, affine=True),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        res_blocks = [ResidualBlock(base_n_filters*4) for _ in range(n_residual_blocks)]
        self.residuals = nn.Sequential(*res_blocks)

        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(base_n_filters*4, base_n_filters*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(base_n_filters*2, affine=True),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(base_n_filters*2, base_n_filters, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(base_n_filters, affine=True),
            nn.ReLU(inplace=True)
        )

        # Output layer
        self.output = nn.Sequential(
            nn.Conv1d(base_n_filters, out_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Tanh() # Output range [-1, 1] matching normalized input
        )

    def forward(self, x):
        x = self.initial(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        x = self.residuals(d2)
        x = self.up1(x)
        x = self.up2(x)
        x = self.output(x)
        return x

# Discriminator (PatchGAN style)
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, base_n_filters=64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Layer 1: No normalization
            nn.Conv1d(in_channels, base_n_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv1d(base_n_filters, base_n_filters*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(base_n_filters*2, affine=True), # Explicitly state affine=True
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv1d(base_n_filters*2, base_n_filters*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(base_n_filters*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4
            nn.Conv1d(base_n_filters*4, base_n_filters*8, kernel_size=4, stride=1, padding=1), # Stride 1 here
            nn.InstanceNorm1d(base_n_filters*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Output Layer (predicts real/fake for patches)
            nn.Conv1d(base_n_filters*8, 1, kernel_size=4, stride=1, padding=1)
            # No sigmoid here if using LSGAN loss (MSELoss)
        )

    def forward(self, x):
        return self.model(x) # Output is a sequence of patch predictions

# --- Loss Functions ---
criterion_GAN = nn.MSELoss() # LSGAN loss
criterion_cycle = nn.L1Loss() # L1 loss for cycle consistency
criterion_identity = nn.L1Loss() # L1 loss for identity mapping

# --- Initialize Models ---
G_AB = Generator().to(DEVICE)
G_BA = Generator().to(DEVICE)
D_A = Discriminator().to(DEVICE)
D_B = Discriminator().to(DEVICE)

# --- CORRECTED Weight Initialization ---
def weights_init_normal(m):
    classname = m.__class__.__name__
    # Check for Convolutional layers
    if classname.find('Conv') != -1:
        # Initialize weights for Conv layers
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        # Initialize bias if it exists and is not None
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

    # Check for Normalization layers (BatchNorm or InstanceNorm)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        # Initialize gamma (weight) if it exists and is not None
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        # Initialize beta (bias) if it exists and is not None
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

# Apply the initialization AFTER model definition and BEFORE optimizer creation
print("Applying custom weight initialization...")
G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)
print("Weight initialization applied.")

# --- Optimizers ---
optimizer_G = optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()),
    lr=LEARNING_RATE, betas=(BETA1, 0.999)
)
optimizer_D_A = optim.Adam(D_A.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

# --- Main Training Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    SAMPLE_RATE = 16000
    SEGMENT_LENGTH = 16384  # ~1 second at 16kHz
    BATCH_SIZE = 4        # Keep low due to memory usage with raw audio
    NUM_EPOCHS = 50
    LAMBDA_CYCLE = 10     # Weight for cycle consistency loss
    LAMBDA_ID = 0.5 * LAMBDA_CYCLE # Weight for identity loss (0.1 to 0.5 of LAMBDA_CYCLE is common)

    DATA_DIR_A = 'bass_a_segments' # Directory containing Bass A segment files (.npy or .wav)
    DATA_DIR_B = 'bass_b_segments' # Directory containing Bass B segment files (.npy or .wav)
    OUTPUT_DIR = 'output/bass_cyclegan'
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    SAMPLE_DIR = os.path.join(OUTPUT_DIR, 'samples')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    print("Setting up datasets...")
    dataset_A = AudioSegmentDataset(DATA_DIR_A, SEGMENT_LENGTH, SAMPLE_RATE)
    dataset_B = AudioSegmentDataset(DATA_DIR_B, SEGMENT_LENGTH, SAMPLE_RATE)

    # Ensure datasets are not empty
    if len(dataset_A) == 0 or len(dataset_B) == 0:
        raise ValueError(f"One or both datasets are empty. Check paths:\n"
                         f"  Bass A: {os.path.abspath(DATA_DIR_A)}\n"
                         f"  Bass B: {os.path.abspath(DATA_DIR_B)}\n"
                         "Ensure these directories contain .npy or .wav segment files.")

    dataloader_A = DataLoader(dataset_A, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    dataloader_B = DataLoader(dataset_B, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    # drop_last=True is important if using InstanceNorm and batch size might become 1

    # Buffers for fake samples (helps stabilize training)
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # --- Training Loop ---
    print(f"Starting training on {DEVICE}...")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Segment length: {SEGMENT_LENGTH} samples")
    print(f"Adam LR: {LEARNING_RATE}, Beta1: {BETA1}")
    print(f"Lambda Cycle: {LAMBDA_CYCLE}, Lambda ID: {LAMBDA_ID}")
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")
    print(f"Sample audio will be saved to: {SAMPLE_DIR}")

    # Determine shorter dataset length to avoid index errors
    steps_per_epoch = min(len(dataloader_A), len(dataloader_B))

    for epoch in range(NUM_EPOCHS):
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()

        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        epoch_loss_GAN = 0.0
        epoch_loss_Cyc = 0.0
        epoch_loss_ID = 0.0

        # Use iterators to manually fetch batches, ensuring balanced iteration
        iter_A = iter(dataloader_A)
        iter_B = iter(dataloader_B)

        progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for i in progress_bar:
            try:
                real_A = next(iter_A).to(DEVICE)
                real_B = next(iter_B).to(DEVICE)
            except StopIteration:
                # Should not happen with steps_per_epoch logic, but as safety
                break

            # Ensure batch size is sufficient for InstanceNorm after drop_last
            if real_A.size(0) < 2 or real_B.size(0) < 2:
                 print(f"Warning: Skipping batch {i} due to insufficient size ({real_A.size(0)}, {real_B.size(0)}) after drop_last.")
                 continue

            # ---------------------
            #  Train Generators
            # ---------------------
            optimizer_G.zero_grad()

            # Identity loss
            if LAMBDA_ID > 0:
                # G_BA should not change real_A much
                loss_id_A = criterion_identity(G_BA(real_A), real_A)
                # G_AB should not change real_B much
                loss_id_B = criterion_identity(G_AB(real_B), real_B)
                loss_identity = (loss_id_A + loss_id_B) / 2
            else:
                loss_identity = torch.tensor(0.0, device=DEVICE) # Ensure it's a tensor on the correct device

            # GAN loss (Trick discriminators)
            # Generate fake B from real A
            fake_B = G_AB(real_A)
            pred_fake_B = D_B(fake_B)
            # Calculate target for real (usually 1) - Ensure shape matches discriminator output
            target_real = torch.ones_like(pred_fake_B, device=DEVICE)
            loss_GAN_AB = criterion_GAN(pred_fake_B, target_real) # G_AB wants D_B to think fake_B is real

            # Generate fake A from real B
            fake_A = G_BA(real_B)
            pred_fake_A = D_A(fake_A)
            # Calculate target for real (usually 1) - Ensure shape matches discriminator output
            target_real_A = torch.ones_like(pred_fake_A, device=DEVICE) # Use separate variable for clarity
            loss_GAN_BA = criterion_GAN(pred_fake_A, target_real_A) # G_BA wants D_A to think fake_A is real

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle consistency loss (L1 distance)
            # Reconstruct A: real A -> fake B -> cycled A
            cycled_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(cycled_A, real_A)

            # Reconstruct B: real B -> fake A -> cycled B
            cycled_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(cycled_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total generator loss
            loss_G = loss_GAN + LAMBDA_CYCLE * loss_cycle + LAMBDA_ID * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator A
            # ---------------------
            optimizer_D_A.zero_grad()

            # Real loss: How well D_A identifies real A
            pred_real_A = D_A(real_A)
            target_real_A = torch.ones_like(pred_real_A, device=DEVICE) # Target for real samples is 1
            loss_D_real_A = criterion_GAN(pred_real_A, target_real_A)

            # Fake loss: How well D_A identifies fake A (using buffer)
            fake_A_buffered = fake_A_buffer.push_and_pop(fake_A.detach()) # Detach G_BA's output
            pred_fake_A = D_A(fake_A_buffered)
            target_fake_A = torch.zeros_like(pred_fake_A, device=DEVICE) # Target for fake samples is 0
            loss_D_fake_A = criterion_GAN(pred_fake_A, target_fake_A)

            # Total loss D_A
            loss_D_A = (loss_D_real_A + loss_D_fake_A) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # ---------------------
            #  Train Discriminator B
            # ---------------------
            optimizer_D_B.zero_grad()

            # Real loss: How well D_B identifies real B
            pred_real_B = D_B(real_B)
            target_real_B = torch.ones_like(pred_real_B, device=DEVICE) # Target for real samples is 1
            loss_D_real_B = criterion_GAN(pred_real_B, target_real_B)

            # Fake loss: How well D_B identifies fake B (using buffer)
            fake_B_buffered = fake_B_buffer.push_and_pop(fake_B.detach()) # Detach G_AB's output
            pred_fake_B = D_B(fake_B_buffered)
            target_fake_B = torch.zeros_like(pred_fake_B, device=DEVICE) # Target for fake samples is 0
            loss_D_fake_B = criterion_GAN(pred_fake_B, target_fake_B)

            # Total loss D_B
            loss_D_B = (loss_D_real_B + loss_D_fake_B) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            # --- Accumulate Losses for Epoch Average ---
            current_loss_D = (loss_D_A + loss_D_B).item() / 2
            current_loss_G = loss_G.item()
            current_loss_GAN = loss_GAN.item()
            current_loss_Cyc = loss_cycle.item() * LAMBDA_CYCLE # Store weighted loss
            current_loss_ID = loss_identity.item() * LAMBDA_ID   # Store weighted loss

            epoch_loss_D += current_loss_D
            epoch_loss_G += current_loss_G
            epoch_loss_GAN += current_loss_GAN
            epoch_loss_Cyc += current_loss_Cyc
            epoch_loss_ID += current_loss_ID

            # --- Logging within Progress Bar ---
            progress_bar.set_postfix({
                'Loss_D': f"{current_loss_D:.4f}",
                'Loss_G': f"{current_loss_G:.4f}",
                'G_GAN': f"{current_loss_GAN:.4f}",
                'G_Cyc': f"{current_loss_Cyc:.4f}", # Show weighted cycle loss
                'G_ID': f"{current_loss_ID:.4f}"    # Show weighted identity loss
            })


        # --- End of Epoch ---
        avg_loss_D = epoch_loss_D / steps_per_epoch
        avg_loss_G = epoch_loss_G / steps_per_epoch
        avg_loss_GAN = epoch_loss_GAN / steps_per_epoch
        avg_loss_Cyc = epoch_loss_Cyc / steps_per_epoch
        avg_loss_ID = epoch_loss_ID / steps_per_epoch

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary: Loss_D={avg_loss_D:.4f}, Loss_G={avg_loss_G:.4f} "
              f"(GAN={avg_loss_GAN:.4f}, Cycle={avg_loss_Cyc:.4f}, ID={avg_loss_ID:.4f})")

        # --- Save Checkpoints  ---
        if (epoch + 1) % 50 == 0 or epoch == NUM_EPOCHS - 1: # Save every 50 epochs and at the end
            print(f"Saving models at epoch {epoch+1}...")
            torch.save(G_AB.state_dict(), os.path.join(CHECKPOINT_DIR, f'G_AB_epoch_{epoch+1}.pth'))
            torch.save(G_BA.state_dict(), os.path.join(CHECKPOINT_DIR, f'G_BA_epoch_{epoch+1}.pth'))
            torch.save(D_A.state_dict(), os.path.join(CHECKPOINT_DIR, f'D_A_epoch_{epoch+1}.pth'))
            torch.save(D_B.state_dict(), os.path.join(CHECKPOINT_DIR, f'D_B_epoch_{epoch+1}.pth'))

            # Set models back to train mode (already done at start of next epoch, but good practice)
            G_AB.train()
            G_BA.train()

    print("Training finished.")
#!/bin/bash

# --- Configuration ---
# Adjust these paths and parameters as needed
CYCLEGAN_REPO_DIR="./pytorch-CycleGAN-and-pix2pix" # Path to the cloned repo
DATASET_DIR="./datasets/bass_cyclegan"         # Path to the prepared dataset
EXP_NAME="bass_cyclegan_complex_v1"            # Experiment name
GPU_IDS="0"                                   # GPU ID(s) to use (e.g., 0 or 0,1)
BATCH_SIZE=1                                  # Batch size (increase if GPU memory allows)
N_EPOCHS=200                                  # Number of epochs at full learning rate
N_EPOCHS_DECAY=200                            # Number of epochs to linearly decay learning rate to zero
SAVE_EPOCH_FREQ=50                            # How often to save checkpoints
LOAD_SIZE=513                                 # Number of frequency bins (N_FFT/2 + 1) -> Calculated by 0_prepare_data.py (e.g., 1024/2 + 1 = 513)
CROP_SIZE=513                                 # Use same as load_size if no cropping desired
NET_G="resnet_9blocks"                        # Generator architecture
NET_D="basic"                                 # Discriminator architecture (PatchGAN)
# --- End Configuration ---

# Get calculated dimensions from data prep script output (replace if needed)
# You might need to manually set LOAD_SIZE based on the output of 0_prepare_data.py
# Example: If N_FFT = 1024, then LOAD_SIZE = 513

# Check if CycleGAN repo exists
if [ ! -d "$CYCLEGAN_REPO_DIR" ]; then
  echo "Error: CycleGAN repository not found at $CYCLEGAN_REPO_DIR"
  echo "Please clone it first: git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git"
  exit 1
fi

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
  echo "Error: Dataset directory not found at $DATASET_DIR"
  echo "Please run 0_prepare_data.py first."
  exit 1
fi

# Navigate to the repo directory
cd "$CYCLEGAN_REPO_DIR" || exit 1

echo "Starting CycleGAN Training..."
echo "Experiment Name: $EXP_NAME"
echo "Dataset Root: $DATASET_DIR"
echo "GPU IDs: $GPU_IDS"

# Run training
python train.py \
  --dataroot "$DATASET_DIR" \
  --name "$EXP_NAME" \
  --model cycle_gan \
  --input_nc 2 \
  --output_nc 2 \
  --preprocess none \
  --load_size $LOAD_SIZE \
  --crop_size $CROP_SIZE \
  --display_winsize $LOAD_SIZE \
  --no_flip \
  --netG "$NET_G" \
  --netD "$NET_D" \
  --pool_size 50 \
  --batch_size $BATCH_SIZE \
  --gpu_ids "$GPU_IDS" \
  --save_epoch_freq $SAVE_EPOCH_FREQ \
  --n_epochs $N_EPOCHS \
  --n_epochs_decay $N_EPOCHS_DECAY \
  --display_freq 200 \
  --print_freq 100 \
  --lambda_identity 0.5 # Optional: Add identity loss

echo "Training finished (or interrupted)."
# Navigate back to the original directory
cd ..
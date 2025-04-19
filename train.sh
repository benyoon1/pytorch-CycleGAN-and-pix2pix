#!/encs/bin/bash

# Example Training Command
# Adjust --gpu_ids, --batch_size, --n_epochs, --n_epochs_decay based on your hardware
# Check the exact time dimension (width) of your .npy files and set crop_size accordingly

# Example assuming .npy time dimension is 345
python train.py \
    --dataroot ./datasets/bass_transfer \
    --name bass_cyclegan_experiment \
    --model cycle_gan \
    --dataset_mode unaligned_npy \
    --input_nc 2 \
    --output_nc 2 \
    --preprocess none \
    --no_flip \
    --load_size 345 \
    --crop_size 345 \
    --batch_size 2 \
    --n_epochs 100 \
    --n_epochs_decay 100 \
    --display_id -1 \
    --save_epoch_freq 50 \
    --gpu_ids 0 # Or -1 for CPU

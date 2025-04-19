import numpy as np
import librosa
import soundfile as sf
import torch
import os
import argparse
from tqdm import tqdm
import scipy.signal # For hann window

# Import necessary components from the cloned CycleGAN repository
# Add the repository's directory to the Python path
import sys
# --- !!! Adjust this path if your repo is located elsewhere !!! ---
CYCLEGAN_REPO_PATH = './pytorch-CycleGAN-and-pix2pix'
# -------------------------------------------------------------------
if CYCLEGAN_REPO_PATH not in sys.path:
    sys.path.append(CYCLEGAN_REPO_PATH)

from models import create_model
from options.test_options import TestOptions # Use test options to setup model loading

# --- Configuration (Should match 0_prepare_data.py) ---
SAMPLE_RATE = 22050
SEGMENT_LEN_S = 2.0
SEGMENT_HOP_S = 1.0 # This determines the overlap during reconstruction
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = N_FFT
# ------------------------------------------------------

# Calculate segment lengths in samples (must match data prep)
SEGMENT_LEN_SAMPLES = int(SEGMENT_LEN_S * SAMPLE_RATE)
SEGMENT_HOP_SAMPLES = int(SEGMENT_HOP_S * SAMPLE_RATE)
N_FREQ_BINS = N_FFT // 2 + 1 # Should match LOAD_SIZE in train script
N_TIME_FRAMES = int(np.floor((SEGMENT_LEN_SAMPLES - WIN_LENGTH) / HOP_LENGTH)) + 1 # Should match width

print(f"Using parameters: SR={SAMPLE_RATE}, SegLen={SEGMENT_LEN_S}s, SegHop={SEGMENT_HOP_S}s")
print(f"STFT: n_fft={N_FFT}, hop={HOP_LENGTH}, win={WIN_LENGTH}")
print(f"Expected segment samples: {SEGMENT_LEN_SAMPLES}")
print(f"Expected frames per segment: {N_TIME_FRAMES}")

def load_generator_model(checkpoint_path, gpu_ids='0'):
    """Loads the generator model from a checkpoint."""
    # Create dummy options similar to test script to load the model
    opt = TestOptions().parse(load_eval_args=False) # Create base options
    opt.checkpoints_dir = os.path.dirname(os.path.dirname(checkpoint_path)) # ./checkpoints/
    opt.name = os.path.basename(os.path.dirname(checkpoint_path)) # experiment_name
    opt.epoch = os.path.basename(checkpoint_path).replace('_net_G_A.pth', '').replace('_net_G_B.pth','').replace('_net_G.pth','') # e.g. 'latest' or '100'
    opt.input_nc = 2  # Ensure these match training
    opt.output_nc = 2
    opt.ngf = 64      # Base number of filters, adjust if different during training
    opt.netG = 'resnet_9blocks' # Ensure this matches the trained generator
    opt.norm = 'instance'      # Norm layer used, ensure match
    opt.no_dropout = True      # Usually no dropout in resnet generator test
    opt.model = 'cycle_gan'    # Set model type
    opt.direction = 'AtoB'     # Specify which generator (A->B or B->A)
    opt.gpu_ids = [int(g) for g in gpu_ids.split(',')] if gpu_ids != '-1' else []
    opt.load_size = N_FREQ_BINS # Important: Match dimensions used in training/prep
    opt.crop_size = N_FREQ_BINS # Important: Match dimensions used in training/prep
    opt.preprocess = 'none'    # We handle preprocessing

    print("--- Model Options ---")
    print(f"Experiment Name: {opt.name}")
    print(f"Checkpoint Epoch: {opt.epoch}")
    print(f"Generator Type: {opt.netG}")
    print(f"Input Channels: {opt.input_nc}")
    print(f"Output Channels: {opt.output_nc}")
    print(f"Load Size: {opt.load_size}")
    print(f"GPU IDs: {opt.gpu_ids}")
    print("---------------------")


    model = create_model(opt)      # Create the model (loads G_A and G_B)
    model.setup(opt)               # Loads networks, sets eval mode

    # We usually want the G_A generator for A->B transformation
    if hasattr(model, 'netG_A'):
        generator = model.netG_A
        print("Loaded Generator G_A (for A -> B conversion).")
    else:
         # Fallback for single generator models if needed, though CycleGAN has two
        generator = model.netG
        print("Loaded single Generator G.")

    generator.eval() # Ensure model is in evaluation mode
    return generator, opt

def main():
    parser = argparse.ArgumentParser(description="Synthesize Bass Audio using trained CycleGAN")
    parser.add_argument('--input_wav', type=str, required=True, help='Path to the input Bass A .wav file')
    parser.add_argument('--output_wav', type=str, required=True, help='Path to save the output Bass B .wav file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the generator checkpoint file (e.g., ./pytorch-CycleGAN-and-pix2pix/checkpoints/bass_cyclegan_exp1/latest_net_G_A.pth)')
    parser.add_argument('--stats_dir', type=str, default="./datasets/bass_cyclegan", help='Directory containing statsA.npz and statsB.npz')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ID(s) to use (e.g., 0 or 0,1, -1 for CPU)')
    args = parser.parse_args()

    # --- Load Normalization Stats ---
    stats_path_A = os.path.join(args.stats_dir, "statsA.npz")
    stats_path_B = os.path.join(args.stats_dir, "statsB.npz")
    if not os.path.exists(stats_path_A) or not os.path.exists(stats_path_B):
        print(f"Error: Normalization stats not found in {args.stats_dir}")
        return

    stats_A = np.load(stats_path_A)
    stats_B = np.load(stats_path_B)
    mean_A, std_A = stats_A['mean'], stats_A['std']
    mean_B, std_B = stats_B['mean'], stats_B['std']
    print("Loaded normalization statistics.")

    # --- Load Model ---
    device = torch.device(f'cuda:{args.gpu_ids.split(",")[0]}' if torch.cuda.is_available() and args.gpu_ids != '-1' else 'cpu')
    print(f"Using device: {device}")

    try:
        generator, opt = load_generator_model(args.checkpoint, args.gpu_ids)
        generator.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the checkpoint path is correct and matches the model architecture used for training.")
        return

    # --- Load and Preprocess Input Audio ---
    print(f"Loading input audio: {args.input_wav}")
    try:
        audio, sr = librosa.load(args.input_wav, sr=SAMPLE_RATE, mono=True)
        original_length = len(audio) # Store original length for ISTFT reconstruction
        print(f"Original audio length: {original_length} samples ({original_length/SAMPLE_RATE:.2f} s)")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # Normalize audio (peak normalization) - Use the same method as in data prep
    max_amp = np.max(np.abs(audio))
    if max_amp > 1e-6:
        audio = audio / max_amp

    # --- Process in Segments ---
    generated_complex_specs = []
    print("Processing audio in segments...")
    for i in tqdm(range(0, len(audio) - SEGMENT_LEN_SAMPLES + 1, SEGMENT_HOP_SAMPLES)):
        segment = audio[i : i + SEGMENT_LEN_SAMPLES]

        if len(segment) != SEGMENT_LEN_SAMPLES: continue # Should not happen with loop bounds

        # 1. STFT
        stft_complex = librosa.stft(segment,
                                    n_fft=N_FFT,
                                    hop_length=HOP_LENGTH,
                                    win_length=WIN_LENGTH,
                                    window='hann',
                                    center=False)

        # Pad/Trim time frames if necessary (safety check)
        if stft_complex.shape[1] != N_TIME_FRAMES:
             if stft_complex.shape[1] < N_TIME_FRAMES:
                 pad_width = N_TIME_FRAMES - stft_complex.shape[1]
                 stft_complex = np.pad(stft_complex, ((0, 0), (0, pad_width)), mode='constant')
             elif stft_complex.shape[1] > N_TIME_FRAMES:
                 stft_complex = stft_complex[:, :N_TIME_FRAMES]

        # 2. Split Real/Imag
        real = np.real(stft_complex)
        imag = np.imag(stft_complex)
        spec_2ch = np.stack([real, imag], axis=0) # (2, F, T)

        # 3. Normalize (using Domain A stats)
        norm_spec_2ch = (spec_2ch - mean_A) / std_A

        # 4. Convert to Tensor and add Batch dim
        input_tensor = torch.from_numpy(norm_spec_2ch).unsqueeze(0).float().to(device) # (1, 2, F, T)

        # 5. Run Generator
        with torch.no_grad():
            generated_norm_spec_tensor = generator(input_tensor)

        # 6. Convert back to NumPy, remove Batch dim
        generated_norm_spec_2ch = generated_norm_spec_tensor.squeeze(0).cpu().numpy() # (2, F, T)

        # 7. Inverse Normalize (using Domain B stats)
        generated_spec_2ch = (generated_norm_spec_2ch * std_B) + mean_B

        # 8. Combine Real/Imag back to Complex
        generated_stft_complex = generated_spec_2ch[0, :, :] + 1j * generated_spec_2ch[1, :, :]

        generated_complex_specs.append(generated_stft_complex)

    if not generated_complex_specs:
        print("Error: No segments were processed. Input audio might be too short.")
        return

    # --- Overlap-Add Reconstruction ---
    print("Reconstructing audio using overlap-add...")
    output_audio = np.zeros(original_length)
    window_sum = np.zeros(original_length)
    # Use a Hann window for overlap-add
    window = scipy.signal.hann(SEGMENT_LEN_SAMPLES, sym=True)

    for i, complex_spec in enumerate(tqdm(generated_complex_specs)):
        # Perform ISTFT for the segment
        segment_audio = librosa.istft(complex_spec,
                                      hop_length=HOP_LENGTH,
                                      win_length=WIN_LENGTH,
                                      window='hann',
                                      length=SEGMENT_LEN_SAMPLES, # Crucial: ensure output length matches window
                                      center=False) # Match STFT center=False

        # Apply window and add to buffer
        start_sample = i * SEGMENT_HOP_SAMPLES
        end_sample = start_sample + SEGMENT_LEN_SAMPLES

        if end_sample > original_length: # Handle potential mismatch at the very end
             # This case ideally shouldn't happen if input slicing was correct
             segment_audio = segment_audio[:original_length - start_sample]
             current_window = window[:original_length - start_sample]
             end_sample = original_length
        else:
            current_window = window

        if start_sample < original_length:
            output_audio[start_sample:end_sample] += segment_audio * current_window
            window_sum[start_sample:end_sample] += current_window**2 # Use squared window for summing if using analysis window in istft

    # Avoid division by zero
    # Epsilon based on expected magnitudes after windowing
    epsilon = 1e-8
    window_sum[window_sum < epsilon] = epsilon

    # Normalize by window sum
    final_audio = output_audio / window_sum

    # Clip final audio to [-1, 1] range
    final_audio = np.clip(final_audio, -1.0, 1.0)

    # --- Save Output Audio ---
    print(f"Saving synthesized audio to: {args.output_wav}")
    try:
        sf.write(args.output_wav, final_audio, SAMPLE_RATE)
        print("Synthesis complete!")
    except Exception as e:
        print(f"Error saving output audio: {e}")


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm.auto import tqdm
import random
from cyclegan import Generator

# --- Configs ---
SAMPLE_RATE = 16000
SEGMENT_LENGTH = 16384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENERATOR_BASE_N_FILTERS = 64
GENERATOR_N_RESIDUAL_BLOCKS = 6


# --- Inference Function ---
def synthesize_audio(input_wav_path, output_wav_path, generator_checkpoint_path):
    """
    Loads a generator model, processes an input WAV file segment by segment,
    and saves the synthesized output WAV file.
    """
    print(f"Using device: {DEVICE}")
    print(f"Expected sample rate: {SAMPLE_RATE}")
    print(f"Expected segment length: {SEGMENT_LENGTH}")

    # --- Load Model ---
    print(f"Loading generator from {generator_checkpoint_path}")
    G_AB = Generator(base_n_filters=GENERATOR_BASE_N_FILTERS,
                     n_residual_blocks=GENERATOR_N_RESIDUAL_BLOCKS).to(DEVICE)
    try:
        G_AB.load_state_dict(torch.load(generator_checkpoint_path, map_location=DEVICE))
        print("State dictionary loaded successfully.")
    except RuntimeError as e:
        print("\n*** Error loading state dictionary. ***")
        print("This often means the Generator class definition here does not")
        print("match the architecture of the saved model checkpoint.")
        print("Verify 'GENERATOR_BASE_N_FILTERS' and 'GENERATOR_N_RESIDUAL_BLOCKS'")
        print(f"and the 'affine' settings in InstanceNorm1d. Original error:\n{e}\n")
        raise e
    except FileNotFoundError:
        print(f"\n*** Error: Checkpoint file not found at {generator_checkpoint_path} ***\n")
        raise

    G_AB.eval()

    # --- Load and Prepare Input Audio ---
    print(f"Loading input audio: {input_wav_path}")
    try:
        audio_in, sr = librosa.load(input_wav_path, sr=None, mono=True)
    except FileNotFoundError:
         print(f"\n*** Error: Input audio file not found at {input_wav_path} ***\n")
         raise

    if sr != SAMPLE_RATE:
        print(f"Resampling input from {sr} Hz to {SAMPLE_RATE} Hz...")
        audio_in = librosa.resample(audio_in, orig_sr=sr, target_sr=SAMPLE_RATE)
        print(f"Resampling complete. Audio length is now {len(audio_in)} samples.")
    # Ensure float32
    audio_in = audio_in.astype(np.float32)


    # Normalize
    max_val = np.max(np.abs(audio_in))
    if max_val > 1e-6:
        print(f"Normalizing input audio (peak was {max_val:.4f})...")
        audio_in = audio_in / max_val * 0.98
    else:
        print("Input audio appears to be silent, processing anyway...")

    # --- Process in Segments using Overlap-Add ---
    hop_length = SEGMENT_LENGTH // 2
    num_samples = len(audio_in)
    output_audio = np.zeros_like(audio_in, dtype=np.float32)
    num_segments = int(np.ceil(max(0, num_samples - SEGMENT_LENGTH) / hop_length)) + 1
    window = np.hanning(SEGMENT_LENGTH).astype(np.float32)

    print(f"Processing audio in {num_segments} overlapping segments...")
    for i in tqdm(range(num_segments)):
        start = i * hop_length
        end = start + SEGMENT_LENGTH

        segment = audio_in[start:end]
        current_segment_length = len(segment)
        if current_segment_length < SEGMENT_LENGTH:
            segment = np.pad(segment, (0, SEGMENT_LENGTH - current_segment_length), 'constant')

        segment_tensor = torch.from_numpy(segment).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output_segment = G_AB(segment_tensor)

        output_segment_np = output_segment.cpu().squeeze().numpy()

        valid_length = min(current_segment_length, SEGMENT_LENGTH)
        output_audio[start : start + valid_length] += (output_segment_np * window)[:valid_length]

    # --- Finalize Output ---
    max_out_val = np.max(np.abs(output_audio))
    if max_out_val > 1e-6:
        print(f"Normalizing output audio (peak was {max_out_val:.4f})...")
        output_audio = (output_audio / max_out_val * 0.98).astype(np.float32)
    else:
        print("Output audio appears to be silent.")
        output_audio = output_audio.astype(np.float32)

    # --- Save Output ---
    output_dir = os.path.dirname(output_wav_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    print(f"Saving output audio to: {output_wav_path}")
    try:
        sf.write(output_wav_path, output_audio, SAMPLE_RATE)
    except Exception as e:
        print(f"\n*** Error saving output file: {e} ***\n")
        raise

    print("Inference complete.")

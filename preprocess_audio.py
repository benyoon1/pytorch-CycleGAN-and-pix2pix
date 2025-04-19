import librosa
import numpy as np
import os
import soundfile as sf
from tqdm import tqdm
import argparse
import json

def preprocess_audio(input_dir, output_dir, domain, segment_len_sec, sr, n_fft, hop_length, win_length, global_max_abs_val=None):
    """
    Loads audio, segments, computes STFT, separates real/imag, normalizes, and saves as .npy.
    Returns the maximum absolute value found in this domain's data if global_max_abs_val is None.
    """
    print(f"Processing domain {domain}...")
    domain_output_dir = os.path.join(output_dir, f'train{domain}')
    os.makedirs(domain_output_dir, exist_ok=True)

    max_abs_val_local = 0.0
    segment_len_samples = int(segment_len_sec * sr)
    file_count = 0

    audio_files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.flac', '.mp3'))] # Add more extensions if needed

    for filename in tqdm(audio_files, desc=f"Processing {domain}"):
        try:
            filepath = os.path.join(input_dir, filename)
            y, current_sr = librosa.load(filepath, sr=sr, mono=True)

            if current_sr != sr:
                 print(f"Warning: File {filename} has sample rate {current_sr}, resampling to {sr}")
                 y = librosa.resample(y, orig_sr=current_sr, target_sr=sr)


            num_segments = len(y) // segment_len_samples

            for i in range(num_segments):
                segment = y[i * segment_len_samples : (i + 1) * segment_len_samples]

                # Ensure segment is exactly segment_len_samples (pad if needed, though unlikely with //)
                if len(segment) < segment_len_samples:
                     segment = np.pad(segment, (0, segment_len_samples - len(segment)))

                # Compute STFT
                stft_result = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
                # stft_result shape: (1 + n_fft/2, num_frames) complex

                # Separate Real and Imaginary parts
                real_part = np.real(stft_result)
                imag_part = np.imag(stft_result)

                # Stack them into a 2-channel tensor
                # Shape: [2, freq_bins, time_frames]
                stft_tensor = np.stack([real_part, imag_part], axis=0)

                # Update max absolute value if calculating globally
                if global_max_abs_val is None:
                    current_max = np.max(np.abs(stft_tensor))
                    if current_max > max_abs_val_local:
                        max_abs_val_local = current_max

                # Normalize if global_max_abs_val is provided
                if global_max_abs_val is not None:
                    stft_tensor_normalized = stft_tensor / global_max_abs_val
                    # Clip to [-1, 1] just in case, though division should handle it
                    stft_tensor_normalized = np.clip(stft_tensor_normalized, -1.0, 1.0)

                    # Save the normalized tensor
                    output_filename = os.path.join(domain_output_dir, f'{os.path.splitext(filename)[0]}_seg{i:04d}.npy')
                    np.save(output_filename, stft_tensor_normalized.astype(np.float32)) # Use float32 for space/efficiency
                    file_count += 1

        except Exception as e:
             print(f"Error processing {filename}: {e}")


    if global_max_abs_val is None:
        print(f"Domain {domain}: Found max absolute value: {max_abs_val_local}")
        return max_abs_val_local
    else:
        print(f"Domain {domain}: Saved {file_count} processed segments.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess audio data for CycleGAN using STFT Real/Imag parts.')
    parser.add_argument('--input_a', required=True, help='Directory containing audio files for domain A (e.g., Bass A)')
    parser.add_argument('--input_b', required=True, help='Directory containing audio files for domain B (e.g., Bass B)')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed .npy files (will create trainA and trainB subdirs)')
    parser.add_argument('--dataset_name', required=True, help='Name for the dataset (e.g., bass_transfer)')
    parser.add_argument('--segment_len_sec', type=float, default=4.0, help='Length of audio segments in seconds')
    parser.add_argument('--sr', type=int, default=22050, help='Target sample rate') # Lower SR reduces computation/size
    parser.add_argument('--n_fft', type=int, default=1024, help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=256, help='Hop length for STFT')
    parser.add_argument('--win_length', type=int, default=1024, help='Window length for STFT (often same as n_fft)')

    args = parser.parse_args()

    # --- Configuration ---
    dataset_output_dir = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    norm_data_path = os.path.join(dataset_output_dir, 'norm_stats.json')

    # --- Run 1: Calculate global max absolute value ---
    print("--- Pass 1: Calculating global max absolute value ---")
    max_a = preprocess_audio(args.input_a, dataset_output_dir, 'A', args.segment_len_sec, args.sr, args.n_fft, args.hop_length, args.win_length, global_max_abs_val=None)
    max_b = preprocess_audio(args.input_b, dataset_output_dir, 'B', args.segment_len_sec, args.sr, args.n_fft, args.hop_length, args.win_length, global_max_abs_val=None)
    global_max = max(max_a, max_b, 1e-6) # Avoid division by zero if dataset is silent
    print(f"Global maximum absolute value: {global_max}")

    # Save normalization stats
    norm_stats = {
        'global_max_abs_val': float(global_max),
        'sr': args.sr,
        'n_fft': args.n_fft,
        'hop_length': args.hop_length,
        'win_length': args.win_length,
        'segment_len_sec': args.segment_len_sec
    }
    with open(norm_data_path, 'w') as f:
        json.dump(norm_stats, f, indent=4)
    print(f"Normalization stats saved to {norm_data_path}")

    # --- Run 2: Normalize and save .npy files ---
    print("\n--- Pass 2: Normalizing and saving .npy files ---")
    preprocess_audio(args.input_a, dataset_output_dir, 'A', args.segment_len_sec, args.sr, args.n_fft, args.hop_length, args.win_length, global_max_abs_val=global_max)
    preprocess_audio(args.input_b, dataset_output_dir, 'B', args.segment_len_sec, args.sr, args.n_fft, args.hop_length, args.win_length, global_max_abs_val=global_max)

    print("\nPreprocessing complete.")
    print(f"Processed data saved in: {dataset_output_dir}")
    print(f"Use --dataroot {args.output_dir}/{args.dataset_name} for training.") # Note: dataroot is parent of trainA/trainB

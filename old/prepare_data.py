import numpy as np
import librosa
import soundfile as sf
import os
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

# --- Configuration ---
# Adjust these parameters based on your needs and data
SAMPLE_RATE = 16000         # Sample rate for audio processing
SEGMENT_LEN_S = 2.0         # Duration of each segment in seconds
SEGMENT_HOP_S = 1.0         # Overlap between segments in seconds (SEGMENT_LEN_S - SEGMENT_HOP_S = overlap)
N_FFT = 1024                # FFT window size
HOP_LENGTH = 256            # Hop length for STFT (determines time resolution)
WIN_LENGTH = N_FFT          # Window length for STFT (usually same as N_FFT)
OUTPUT_DIR = "./datasets/bass_cyclegan" # Output directory for processed data
RAW_DATA_DIR = "./datasets_raw"        # Directory containing bassA and bassB folders
N_JOBS = -1                 # Number of parallel jobs for processing (-1 uses all cores)
# ---------------------

# Calculate segment lengths in samples
SEGMENT_LEN_SAMPLES = int(SEGMENT_LEN_S * SAMPLE_RATE)
SEGMENT_HOP_SAMPLES = int(SEGMENT_HOP_S * SAMPLE_RATE)

# Calculate expected number of time frames per segment
# This needs to be consistent for the CycleGAN model
# n_frames = floor(segment_len_samples / hop_length) + 1 # Librosa's default center=True logic
# Or ensure segments are padded/trimmed to exact length before STFT
# Let's enforce segment length FIRST, then STFT
N_TIME_FRAMES = int(np.floor((SEGMENT_LEN_SAMPLES - WIN_LENGTH) / HOP_LENGTH)) + 1
N_FREQ_BINS = N_FFT // 2 + 1

print(f"Sample Rate: {SAMPLE_RATE}")
print(f"Segment Length: {SEGMENT_LEN_S}s ({SEGMENT_LEN_SAMPLES} samples)")
print(f"Segment Hop: {SEGMENT_HOP_S}s ({SEGMENT_HOP_SAMPLES} samples)")
print(f"STFT Params: n_fft={N_FFT}, hop={HOP_LENGTH}, win={WIN_LENGTH}")
print(f"Expected Spectrogram Shape: (2, {N_FREQ_BINS}, {N_TIME_FRAMES})")

def process_audio_file(filepath, domain, output_base_dir):
    """Loads, segments, computes STFT, and saves segments for one audio file."""
    processed_segments = []
    try:
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

        # Normalize audio (peak normalization)
        max_amp = np.max(np.abs(audio))
        if max_amp > 1e-6:
            audio = audio / max_amp
        else:
            print(f"Warning: Silent audio file skipped: {filepath}")
            return [] # Skip silent files

        # Pad audio at the end if needed to make sure the last segment is full length?
        # Or just take segments that fit completely. Let's take only full segments for simplicity.

        num_segments = 0
        for i in range(0, len(audio) - SEGMENT_LEN_SAMPLES + 1, SEGMENT_HOP_SAMPLES):
            segment = audio[i : i + SEGMENT_LEN_SAMPLES]

            # Ensure segment is exactly the right length (might be off slightly due to int conversion)
            if len(segment) != SEGMENT_LEN_SAMPLES:
                # This shouldn't happen with the loop bounds above, but safety check
                print(f"Warning: Segment length mismatch in {filepath}, skipping segment.")
                continue

            # Calculate STFT
            stft_result = librosa.stft(segment,
                                       n_fft=N_FFT,
                                       hop_length=HOP_LENGTH,
                                       win_length=WIN_LENGTH,
                                       window='hann', # Use hann window
                                       center=False) # Use center=False for easier frame calculation consistency

            # Verify shape (especially time frames)
            if stft_result.shape[1] != N_TIME_FRAMES:
                 # Pad or trim the time dimension if needed
                 # Example: Padding with zeros if too short
                 if stft_result.shape[1] < N_TIME_FRAMES:
                     pad_width = N_TIME_FRAMES - stft_result.shape[1]
                     stft_result = np.pad(stft_result, ((0, 0), (0, pad_width)), mode='constant')
                 # Example: Trimming if too long
                 elif stft_result.shape[1] > N_TIME_FRAMES:
                     stft_result = stft_result[:, :N_TIME_FRAMES]

            # Separate real and imaginary parts
            real = np.real(stft_result)
            imag = np.imag(stft_result)

            # Stack into 2 channels: (2, n_freq_bins, n_time_frames)
            spec_2ch = np.stack([real, imag], axis=0)

            # Save segment temporarily (normalization happens later)
            segment_filename = f"{os.path.splitext(os.path.basename(filepath))[0]}_seg{num_segments:05d}.npy"
            segment_output_path = os.path.join(output_base_dir, domain, segment_filename)
            np.save(segment_output_path, spec_2ch.astype(np.float32)) # Save as float32

            processed_segments.append(segment_output_path)
            num_segments += 1

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []

    return processed_segments

def calculate_stats(npy_files):
    """Calculates mean and std across all segments for real and imaginary channels."""
    sums = np.zeros(2, dtype=np.float64)     # Sum for mean calculation (Real, Imag)
    sums_sq = np.zeros(2, dtype=np.float64)  # Sum of squares for std calculation (Real, Imag)
    count = 0                                # Total number of elements (pixels) per channel

    print(f"Calculating stats for {len(npy_files)} files...")
    for f in tqdm(npy_files):
        try:
            spec_2ch = np.load(f) # Shape (2, F, T)
            sums += np.sum(spec_2ch, axis=(1, 2))
            sums_sq += np.sum(spec_2ch**2, axis=(1, 2))
            count += spec_2ch.shape[1] * spec_2ch.shape[2] # F * T
        except Exception as e:
            print(f"Warning: Could not load or process {f} for stats: {e}")
            continue

    if count == 0:
        raise ValueError("No valid segments found to calculate statistics.")

    mean = sums / count
    # Var = E[X^2] - (E[X])^2
    variance = (sums_sq / count) - (mean**2)
    # Add small epsilon for numerical stability before sqrt
    std = np.sqrt(variance + 1e-8)

    print(f"Stats calculated: Mean={mean}, Std={std}")
    # Reshape mean and std to be broadcastable for normalization: (2, 1, 1)
    return mean.reshape(2, 1, 1).astype(np.float32), std.reshape(2, 1, 1).astype(np.float32)

def normalize_segments(npy_files, mean, std, output_dir):
    """Applies normalization and saves segments to the final train/test directories."""
    print(f"Normalizing {len(npy_files)} files...")
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for f_path in tqdm(npy_files):
        try:
            spec_2ch = np.load(f_path)
            norm_spec_2ch = (spec_2ch - mean) / std

            # Construct final output path
            filename = os.path.basename(f_path)
            final_output_path = os.path.join(output_dir, filename)

            np.save(final_output_path, norm_spec_2ch)
            # Optionally remove the temporary unnormalized file
            # os.remove(f_path)
        except Exception as e:
            print(f"Warning: Could not normalize/save {f_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Bass CycleGAN Data")
    parser.add_argument('--raw_dir', type=str, default=RAW_DATA_DIR, help='Directory with bassA and bassB subfolders containing .wav files')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Output directory for processed .npy files and stats')
    args = parser.parse_args()

    RAW_DATA_DIR = args.raw_dir
    OUTPUT_DIR = args.output_dir

    print("Starting Data Preparation...")

    # Temporary storage for unnormalized segments
    temp_output_A = os.path.join(OUTPUT_DIR, "temp_A")
    temp_output_B = os.path.join(OUTPUT_DIR, "temp_B")
    os.makedirs(temp_output_A, exist_ok=True)
    os.makedirs(temp_output_B, exist_ok=True)

    # --- Process Domain A (Bass A) ---
    print("\nProcessing Domain A (Bass A)...")
    files_A = glob.glob(os.path.join(RAW_DATA_DIR, "bassA", "*.wav")) \
            + glob.glob(os.path.join(RAW_DATA_DIR, "bassA", "*.WAV"))
    print(f"Found {len(files_A)} audio files for Domain A.")
    results_A = Parallel(n_jobs=N_JOBS)(delayed(process_audio_file)(f, "temp_A", OUTPUT_DIR) for f in tqdm(files_A))
    all_npy_files_A = [item for sublist in results_A for item in sublist] # Flatten list
    print(f"Generated {len(all_npy_files_A)} segments for Domain A.")

    # --- Process Domain B (Bass B) ---
    print("\nProcessing Domain B (Bass B)...")
    files_B = glob.glob(os.path.join(RAW_DATA_DIR, "bassB", "*.wav")) \
            + glob.glob(os.path.join(RAW_DATA_DIR, "bassB", "*.WAV"))
    print(f"Found {len(files_B)} audio files for Domain B.")
    results_B = Parallel(n_jobs=N_JOBS)(delayed(process_audio_file)(f, "temp_B", OUTPUT_DIR) for f in tqdm(files_B))
    all_npy_files_B = [item for sublist in results_B for item in sublist] # Flatten list
    print(f"Generated {len(all_npy_files_B)} segments for Domain B.")

    if not all_npy_files_A or not all_npy_files_B:
        print("Error: No segments generated for one or both domains. Check input data and parameters.")
        # Consider cleaning up temp directories here
        return

    # --- Calculate Normalization Statistics ---
    print("\nCalculating normalization statistics...")
    mean_A, std_A = calculate_stats(all_npy_files_A)
    mean_B, std_B = calculate_stats(all_npy_files_B)

    # Save stats
    stats_path_A = os.path.join(OUTPUT_DIR, "statsA.npz")
    stats_path_B = os.path.join(OUTPUT_DIR, "statsB.npz")
    np.savez(stats_path_A, mean=mean_A, std=std_A)
    np.savez(stats_path_B, mean=mean_B, std=std_B)
    print(f"Saved stats for Domain A to {stats_path_A}")
    print(f"Saved stats for Domain B to {stats_path_B}")

    # --- Normalize and Save Segments ---
    print("\nApplying normalization and saving final segments...")
    final_output_A = os.path.join(OUTPUT_DIR, "trainA") # Final destination for normalized A
    final_output_B = os.path.join(OUTPUT_DIR, "trainB") # Final destination for normalized B

    # Use single thread for normalization as it's mostly I/O bound after loading stats
    normalize_segments(all_npy_files_A, mean_A, std_A, final_output_A)
    normalize_segments(all_npy_files_B, mean_B, std_B, final_output_B)

    # --- Clean up temporary files ---
    print("\nCleaning up temporary files...")
    # This assumes normalize_segments doesn't remove them; enable removal there or uncomment below
    # for f in all_npy_files_A: os.remove(f)
    # for f in all_npy_files_B: os.remove(f)
    # os.rmdir(temp_output_A)
    # os.rmdir(temp_output_B)
    # For safety, let's just inform the user to remove temp folders manually if needed
    print(f"Temporary files are in {temp_output_A} and {temp_output_B}. You may remove these folders.")

    print("\nData Preparation Complete!")
    print(f"Number of Freq Bins (H): {N_FREQ_BINS}")
    print(f"Number of Time Frames (W): {N_TIME_FRAMES}")
    print("Make sure to use these dimensions (or the corresponding load_size/crop_size)")
    print(f"when running the CycleGAN training script (e.g., --load_size {N_FREQ_BINS} --crop_size {N_FREQ_BINS}).")
    print(f"The width ({N_TIME_FRAMES}) should be handled automatically if using --preprocess none,")
    print("but double-check the CycleGAN code's data loading if issues arise.")

if __name__ == "__main__":
    main()
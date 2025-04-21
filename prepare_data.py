import librosa
import numpy as np
import os
import soundfile as sf
from tqdm.auto import tqdm

def create_segments(input_dir, output_dir, segment_length, target_sr, normalization='peak'):
    os.makedirs(output_dir, exist_ok=True)
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    segment_count = 0

    for filepath in tqdm(all_files, desc=f"Processing files in {input_dir}"):
        try:
            audio, sr = librosa.load(filepath, sr=None, mono=True)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            # Normalization
            if normalization == 'peak':
                peak = np.max(np.abs(audio))
                if peak > 1e-6: # Avoid division by zero for silence
                    audio = audio / peak * 0.98 # Normalize to slightly below 1.0
            elif normalization == 'rms':
                 # Add RMS normalization if desired
                 pass # Implement RMS normalization here


            # Calculate number of segments
            num_segments = len(audio) // segment_length

            for i in range(num_segments):
                start = i * segment_length
                end = start + segment_length
                segment = audio[start:end]

                # Optional: Add check for silence/low energy - skip quiet segments
                rms = np.sqrt(np.mean(segment**2))
                if rms < 0.01: # Adjust threshold as needed
                   # print(f"Skipping silent segment {segment_count} from {filepath}")
                   continue

                # Save segment as npy (faster loading) or wav
                segment_filename = os.path.join(output_dir, f"segment_{segment_count:06d}.npy")
                np.save(segment_filename, segment.astype(np.float32))
                # Or save as wav:
                # segment_filename = os.path.join(output_dir, f"segment_{segment_count:06d}.wav")
                # sf.write(segment_filename, segment, target_sr)
                segment_count += 1

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    print(f"Created {segment_count} segments in {output_dir}")
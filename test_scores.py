from frechet_audio_distance import FrechetAudioDistance
import torch
import os
import numpy as np
import librosa
from scipy.special import kl_div

# ref_dir = "ibanez_test_16"
# gen_dir = "output"
# ref_dir = "squier_train_16"
# gen_dir = "ibanez_train_16"

# FAD score
# fad = FrechetAudioDistance(
#     sample_rate=16000,
#     use_pca=False
# )
# fad_score = fad.score(ref_dir, gen_dir)
# print(f"FAD score: {fad_score:.4f}")

def calculate_kl_divergence(dir1, dir2, sample_rate=16000):
    audio_files1 = [os.path.join(dir1, f) for f in os.listdir(dir1) if f.endswith(('.wav', '.mp3'))]
    audio_files2 = [os.path.join(dir2, f) for f in os.listdir(dir2) if f.endswith(('.wav', '.mp3'))]
    
    audio_files1 = audio_files1[:10]
    audio_files2 = audio_files2[:10]
    
    # Extract MFCCs from all files in dir1
    all_mfccs1 = []
    for audio_file in audio_files1:
        y, _ = librosa.load(audio_file, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=20)
        all_mfccs1.append(mfcc.flatten())
    
    # Extract MFCCs from all files in dir2
    all_mfccs2 = []
    for audio_file in audio_files2:
        y, _ = librosa.load(audio_file, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=20)
        all_mfccs2.append(mfcc.flatten())
    
    mfccs1 = np.vstack(all_mfccs1)
    mfccs2 = np.vstack(all_mfccs2)
    
    mean1 = np.mean(mfccs1, axis=0)
    mean2 = np.mean(mfccs2, axis=0)
    
    p = np.maximum(mean1, 1e-10)
    q = np.maximum(mean2, 1e-10)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate KL divergence
    kl = np.sum(kl_div(p, q))
    
    return kl

# try:
#     kl_score = calculate_kl_divergence(ref_dir, gen_dir)
#     print(f"KL Divergence score: {kl_score:.4f}")
# except Exception as e:
#     print(f"Error calculating KL Divergence: {e}")
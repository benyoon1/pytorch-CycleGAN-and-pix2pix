import torch
import os
import numpy as np
import librosa
import soundfile as sf
# from options.test_options import TestOptions # No longer needed
from models import create_model
# from data.base_dataset import get_transform # Unused
import json
from tqdm import tqdm
import argparse # Keep argparse just for Namespace object

# --- Hardcoded Configuration ---
class HardcodedOptions:
    # --- Paths ---
    input_audio = './datasets_raw/squier_test.wav'
    output_audio = './datasets_raw/output.wav'
    norm_stats_path = './datasets/bass_transfer/norm_stats.json'
    checkpoints_dir = './checkpoints'
    dataroot = './datasets/bass_transfer' # Still needed for some path constructions potentially

    # --- Model/Experiment ---
    name = 'bass_cyclegan_experiment' # Name of the experiment (used for checkpoint loading)
    model = 'cycle_gan'             # Model type
    epoch = 'latest'                # Epoch to load ('latest' or specific number)
    load_iter = 0                   # <<<--- ADD THIS LINE (Load by iteration number, 0 means use epoch)
    netG = 'resnet_9blocks'         # Generator architecture (confirm if different from default)
    netD = 'basic'                  # Discriminator type (e.g., 'basic', 'n_layers')

    # --- Data/Preprocessing ---
    input_nc = 2                    # Input channels (Real, Imag)
    output_nc = 2                   # Output channels (Real, Imag)
    preprocess = 'none'             # Preprocessing mode (must match training)
    no_flip = True                  # Disable flipping (audio spectrograms shouldn't be flipped)
    direction = 'AtoB'              # Translation direction (used by CycleGAN model)

    # --- Environment/Execution ---
    gpu_ids = '0'                   # GPU IDs ('0', '0,1', '-1' for CPU)
    num_threads = 0                 # Test code only supports num_threads = 0
    batch_size = 1                  # Test code only supports batch_size = 1
    serial_batches = True           # Disable data shuffling
    display_id = -1                 # No visdom display
    verbose = False                 # Set to True for more model loading details if needed
    isTrain = False
    # eval = True                   # Usually set by default in model.setup for test phase

    # --- Network Architecture Parameters (Add these & MATCH TRAINING) ---
    ngf = 64                        # <<<--- ADD: Num Generator Filters
    ndf = 64                        # <<<--- ADD: Num Discriminator Filters
    norm = 'instance'               # <<<--- ADD: Normalization Type (instance, batch, none)
    init_type = 'normal'            # <<<--- ADD: Weight Initialization Type
    init_gain = 0.02                # <<<--- ADD: Initialization Gain
    no_dropout = True               # <<<--- ADD: Dropout Setting (True for resnet_9blocks default)
    n_layers_D = 3                  # <<<--- ADD: Num layers in PatchGAN discriminator if netD='n_layers' or 'pixel' (default is 3)

# Create an object similar to what parsing would produce
opt = HardcodedOptions()

# --- load_model and process_audio_segment functions (Minor change in load_model print) ---

def load_model(options): # Renamed argument for clarity
    # --- GPU ID Processing ---
    str_ids = options.gpu_ids.split(',')
    options.gpu_ids = [] # Re-assign options.gpu_ids to be the list of ints
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            options.gpu_ids.append(id)

    # Set device based on processed gpu_ids
    if len(options.gpu_ids) > 0 and torch.cuda.is_available():
        torch.cuda.set_device(options.gpu_ids[0])
        print(f"Using GPU: {options.gpu_ids}")
    else:
        options.gpu_ids = [] # Ensure it's empty if using CPU
        print("Using CPU.")
    # --- End GPU ID Processing ---

    model = create_model(options)      # create a model given options
    # The model's __init__ often uses opt.gpu_ids to set self.device
    # And setup calls init_net which uses the list format correctly now
    model.setup(options)               # regular setup: load and print networks; create schedulers

    # Ensure model is in eval mode after setup
    model.eval()

    # Construct path for clarity (adjust based on actual checkpoint naming if needed)
    g_model_name = 'latest_net_G_A.pth' # Adjust if your model uses just G
    # Or determine based on direction:
    # g_model_name = f'{options.epoch}_net_G_{"A" if options.direction == "AtoB" else "B"}.pth'

    load_path = os.path.join(options.checkpoints_dir, options.name, g_model_name)
    print(f"Attempting to load Generator model like: {load_path}")

    if not os.path.exists(load_path):
        print(f"Warning: Checkpoint file not found at {load_path}. Trying generic 'latest_net_G.pth'")
        generic_load_path = os.path.join(options.checkpoints_dir, options.name, 'latest_net_G.pth')
        if not os.path.exists(generic_load_path):
             print(f"Error: Neither {load_path} nor {generic_load_path} found. Check checkpoint directory and naming.")
             exit(1)
        else:
            print(f"Note: Using {generic_load_path}")

    return model


def process_audio_segment(segment_audio, model, options, norm_stats): # Renamed argument
    """Applies STFT, normalization, runs model, denormalizes, and applies ISTFT."""
    sr = norm_stats['sr']
    n_fft = norm_stats['n_fft']
    hop_length = norm_stats['hop_length']
    win_length = norm_stats['win_length']
    global_max_abs_val = norm_stats['global_max_abs_val']

    # 1. STFT
    stft_result = librosa.stft(segment_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # 2. Real/Imag Split
    real_part = np.real(stft_result)
    imag_part = np.imag(stft_result)
    stft_tensor = np.stack([real_part, imag_part], axis=0) # Shape: [2, F, T]

    # 3. Normalization
    stft_tensor_normalized = stft_tensor / global_max_abs_val
    stft_tensor_normalized = np.clip(stft_tensor_normalized, -1.0, 1.0)

    # 4. Convert to PyTorch Tensor and add batch dimension
    input_tensor = torch.from_numpy(stft_tensor_normalized).float().unsqueeze(0) # Shape: [1, 2, F, T]

    # Move tensor to GPU if specified (model.device already set up)
    if options.gpu_ids != "-1":
         input_tensor = input_tensor.to(model.device)

    # 5. Run Model (A->B)
    with torch.no_grad(): # Essential for inference
        # Ensure input keys match model.set_input expectations ('A', 'B')
        model.set_input({'A': input_tensor, 'A_paths': ['dummy_path']})
        model.forward() # Use forward() for inference

    # 6. Get output tensor
    output_visuals = model.get_current_visuals()
    output_key = 'fake_B' # For A->B translation
    if output_key not in output_visuals:
         available_keys = list(output_visuals.keys())
         raise KeyError(f"Could not find '{output_key}' in model output visuals. Available keys: {available_keys}.")

    output_tensor = output_visuals[output_key].squeeze(0).cpu().detach().numpy() # Shape: [2, F, T]

    # 7. Denormalization
    output_tensor_denorm = output_tensor * global_max_abs_val

    # 8. Complex Reconstruction
    complex_spec_out = output_tensor_denorm[0, :, :] + 1j * output_tensor_denorm[1, :, :]

    # 9. Inverse STFT
    output_audio = librosa.istft(complex_spec_out, hop_length=hop_length, win_length=win_length, length=len(segment_audio))

    return output_audio


# --- Main Execution Block (Simplified) ---
if __name__ == '__main__':

    # --- Load norm_stats using hardcoded path ---
    try:
        with open(opt.norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        print(f"Loaded normalization stats: {norm_stats}")
    except FileNotFoundError:
        print(f"Error: Normalization stats file not found at {opt.norm_stats_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading normalization stats from {opt.norm_stats_path}: {e}")
        exit(1)

    # --- Load the model using hardcoded options ---
    model = load_model(opt)

    # --- Load input audio using hardcoded path and norm_stats ---
    try:
        print(f"Loading input audio: {opt.input_audio}")
        y_in, sr_in = librosa.load(opt.input_audio, sr=norm_stats['sr'], mono=True)
        print(f"Input audio loaded: {len(y_in)/norm_stats['sr']:.2f} seconds, Sample Rate: {norm_stats['sr']}")
        if sr_in != norm_stats['sr']:
             print(f"Warning: Input audio sample rate {sr_in} differs from training rate {norm_stats['sr']}. Librosa handled resampling.")
    except FileNotFoundError:
         print(f"Error: Input audio file not found at {opt.input_audio}")
         exit(1)
    except Exception as e:
         print(f"Error loading input audio {opt.input_audio}: {e}")
         exit(1)

    # --- Segmentation and Processing using hardcoded options and norm_stats ---
    segment_len_sec = norm_stats['segment_len_sec']
    segment_len_samples = int(segment_len_sec * norm_stats['sr'])
    hop_length = norm_stats['hop_length']

    output_audio_full = np.array([], dtype=np.float32)

    num_segments = int(np.ceil(len(y_in) / segment_len_samples))
    print(f"Processing audio in {num_segments} segments...")

    for i in tqdm(range(num_segments), desc="Generating Audio"):
        start_sample = i * segment_len_samples
        end_sample = start_sample + segment_len_samples
        segment = y_in[start_sample:end_sample]

        if len(segment) < segment_len_samples:
             segment = np.pad(segment, (0, segment_len_samples - len(segment)), 'constant')

        # Process the segment
        try:
             output_segment = process_audio_segment(segment, model, opt, norm_stats)
             output_audio_full = np.concatenate((output_audio_full, output_segment))
        except Exception as e:
             print(f"\nError processing segment {i}: {e}")
             print("Skipping segment.")
             # Optionally pad with silence if skipping:
             # output_audio_full = np.concatenate((output_audio_full, np.zeros_like(segment)))


    # Trim the output to the original input length
    output_audio_full = output_audio_full[:len(y_in)]

    # --- Save the output audio using hardcoded path and norm_stats ---
    try:
        print(f"Saving output audio to: {opt.output_audio}")
        sf.write(opt.output_audio, output_audio_full, norm_stats['sr'])
        print("Inference complete.")
    except Exception as e:
        print(f"Error saving output audio to {opt.output_audio}: {e}")
        exit(1)
import torch
import os
import numpy as np
import librosa
import soundfile as sf
from options.test_options import TestOptions
from models import create_model
# from data.base_dataset import get_transform # Unused
import json
from tqdm import tqdm
import argparse # <--- Make sure argparse is imported

# --- load_model and process_audio_segment functions remain the same ---
def load_model(opt):
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    # Important: Ensure these match the data and training settings
    # These should ideally be read from opt parsed by TestOptions if possible,
    # but setting them directly works if consistent.
    opt.input_nc = 2
    opt.output_nc = 2
    opt.preprocess = 'none' # Match training

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # Ensure model is in eval mode after setup
    model.eval()
    # The previous check was `if opt.eval: model.eval()`.
    # For testing, we almost always want eval mode.
    # The `opt.eval` flag is typically used by the framework's standard test.py
    # to potentially load slightly differently. Forcing eval() here is safe.

    print(f"Model loaded from {opt.checkpoints_dir}/{opt.name}/{opt.epoch}_net_G_A.pth (or similar)")
    # Constructing the path might be safer:
    load_filename = f'{opt.epoch}_net_{opt.model}.pth' # Simplified, assumes model name matches file somewhat
    # A more robust way might involve checking which G network exists (G_A or G_B) if direction matters
    load_path = os.path.join(opt.checkpoints_dir, opt.name, load_filename)
    # print(f"Attempting to load model from path like: {load_path}") # Debug print

    return model

def process_audio_segment(segment_audio, model, opt, norm_stats):
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
    if opt.gpu_ids != "-1": # Move tensor to GPU if specified
         input_tensor = input_tensor.to(model.device)


    # 5. Run Model (A->B)
    # Ensure model is on correct device already (handled by model.setup)
    # Make sure input data dictionary keys match what model expects
    # The standard test.py uses 'real_A' and 'real_B', but CycleGANModel.set_input uses 'A' and 'B'.
    # Let's stick to 'A' as used in the original script.
    with torch.no_grad(): # Essential for inference to save memory and compute
        model.set_input({'A': input_tensor, 'A_paths': ['dummy_path']})
        model.forward() # Use forward() for inference, test() might do extra things

    # 6. Get output tensor
    output_visuals = model.get_current_visuals()
    output_key = 'fake_B' # Assuming we want A->B conversion
    if output_key not in output_visuals:
         # Try the other direction key just in case? Or error out clearly.
         available_keys = list(output_visuals.keys())
         raise KeyError(f"Could not find '{output_key}' in model output visuals. Available keys: {available_keys}. Check model output.")

    output_tensor = output_visuals[output_key].squeeze(0).cpu().detach().numpy() # Shape: [2, F, T]

    # 7. Denormalization
    output_tensor_denorm = output_tensor * global_max_abs_val

    # 8. Complex Reconstruction
    complex_spec_out = output_tensor_denorm[0, :, :] + 1j * output_tensor_denorm[1, :, :]

    # 9. Inverse STFT
    output_audio = librosa.istft(complex_spec_out, hop_length=hop_length, win_length=win_length, length=len(segment_audio))

    return output_audio


if __name__ == '__main__':
    # --- PARSING REVISED ---
    # 1. Create the TestOptions parser BUT DON'T PARSE YET
    test_opt_parser = TestOptions()

    # 2. Create a separate parser for the custom audio arguments
    audio_parser = argparse.ArgumentParser(description='Audio-specific arguments for CycleGAN testing')
    audio_parser.add_argument('--input_audio', required=True, help='Path to the input audio file (Domain A)')
    audio_parser.add_argument('--output_audio', required=True, help='Path to save the output audio file (Domain B sound)')
    audio_parser.add_argument('--norm_stats_path', required=True, help='Path to the norm_stats.json file created during preprocessing')
    # Add any other *custom* arguments here if needed in the future

    # 3. Parse ONLY the custom arguments first using parse_known_args
    # This allows standard arguments like --name, --gpu_ids etc. to pass through
    audio_args, remaining_args = audio_parser.parse_known_args()

    # 4. Now, parse the *remaining* arguments (which should be the standard ones)
    # using the TestOptions parser. Pass the parsed custom args namespace to modify it.
    opt = test_opt_parser.parse_args(remaining_args)

    # `opt` now holds standard framework options
    # `audio_args` holds the custom audio options

    # --- END PARSING REVISED ---

    # --- Load norm_stats using audio_args ---
    try:
        with open(audio_args.norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        print(f"Loaded normalization stats: {norm_stats}")
    except FileNotFoundError:
        print(f"Error: Normalization stats file not found at {audio_args.norm_stats_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading normalization stats from {audio_args.norm_stats_path}: {e}")
        exit(1)

    # --- Load the model using opt ---
    # The load_model function sets necessary defaults in opt
    model = load_model(opt)

    # --- Load input audio using audio_args and norm_stats ---
    try:
        print(f"Loading input audio: {audio_args.input_audio}")
        y_in, sr_in = librosa.load(audio_args.input_audio, sr=norm_stats['sr'], mono=True)
        print(f"Input audio loaded: {len(y_in)/norm_stats['sr']:.2f} seconds, Sample Rate: {norm_stats['sr']}")
        if sr_in != norm_stats['sr']:
             print(f"Warning: Input audio sample rate {sr_in} differs from training rate {norm_stats['sr']}. Librosa handled resampling.")
    except FileNotFoundError:
         print(f"Error: Input audio file not found at {audio_args.input_audio}")
         exit(1)
    except Exception as e:
         print(f"Error loading input audio {audio_args.input_audio}: {e}")
         exit(1)


    # --- Segmentation and Processing using opt and norm_stats ---
    segment_len_sec = norm_stats['segment_len_sec']
    segment_len_samples = int(segment_len_sec * norm_stats['sr'])
    hop_length = norm_stats['hop_length']

    output_audio_full = np.array([], dtype=np.float32)

    num_segments = int(np.ceil(len(y_in) / segment_len_samples))
    print(f"Processing audio in {num_segments} segments...")

    # Check if GPU is available based on opt.gpu_ids
    use_gpu = False
    if opt.gpu_ids != '-1':
        try:
            gpu_list = [int(x) for x in opt.gpu_ids.split(',')]
            if torch.cuda.is_available() and len(gpu_list) > 0:
                use_gpu = True
                # Assuming model is already moved to device in load_model/setup
                print(f"Using GPU: {opt.gpu_ids}")
            else:
                print("GPU specified but not available or list empty. Using CPU.")
        except ValueError:
            print(f"Invalid gpu_ids format: {opt.gpu_ids}. Using CPU.")


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
             # Decide whether to skip segment or stop execution
             print("Skipping segment.")
             # Pad with silence of equivalent length if skipping
             # output_audio_full = np.concatenate((output_audio_full, np.zeros_like(segment)))


    # Trim the output to the original input length
    output_audio_full = output_audio_full[:len(y_in)]

    # --- Save the output audio using audio_args and norm_stats ---
    try:
        print(f"Saving output audio to: {audio_args.output_audio}")
        sf.write(audio_args.output_audio, output_audio_full, norm_stats['sr'])
        print("Inference complete.")
    except Exception as e:
        print(f"Error saving output audio to {audio_args.output_audio}: {e}")
        exit(1)
import torch
import os
import numpy as np
import librosa
import soundfile as sf
from options.test_options import TestOptions
from models import create_model
from data.base_dataset import get_transform # We might not need this if preprocess is 'none'
import json
from tqdm import tqdm

def load_model(opt):
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    # Important: Ensure these match the data and training settings
    opt.input_nc = 2
    opt.output_nc = 2
    opt.preprocess = 'none' # Match training

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()
    print(f"Model loaded from {opt.checkpoints_dir}/{opt.name}/latest_net_G_A.pth (or similar)")
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
    # Get dimensions for potential padding/cropping if needed later
    # n_freq_bins, n_time_frames = stft_result.shape

    # 2. Real/Imag Split
    real_part = np.real(stft_result)
    imag_part = np.imag(stft_result)
    stft_tensor = np.stack([real_part, imag_part], axis=0) # Shape: [2, F, T]

    # 3. Normalization
    stft_tensor_normalized = stft_tensor / global_max_abs_val
    stft_tensor_normalized = np.clip(stft_tensor_normalized, -1.0, 1.0)

    # 4. Convert to PyTorch Tensor and add batch dimension
    input_tensor = torch.from_numpy(stft_tensor_normalized).float().unsqueeze(0) # Shape: [1, 2, F, T]

    # Check if model requires specific size (if crop_size was used strictly during training)
    # This is a basic check; more robust handling might be needed if sizes vary wildly
    # target_time_frames = opt.crop_size # Assumes crop_size = time dimension
    # if input_tensor.shape[3] != target_time_frames:
    #    print(f"Warning: Input time frames {input_tensor.shape[3]} != target {target_time_frames}. Cropping/Padding might be needed.")
       # Add cropping/padding logic here if necessary based on how training handled size mismatches

    # 5. Run Model (A->B)
    model.set_input({'A': input_tensor, 'A_paths': ['dummy_path']}) # Provide dummy path
    model.test() # Runs forward pass

    # Get output tensor (assuming we want G_A output)
    # Check visualizer output or model code to confirm key (usually 'fake_B')
    output_visuals = model.get_current_visuals()
    if 'fake_B' not in output_visuals:
        raise KeyError("Could not find 'fake_B' in model output visuals. Check model implementation/output keys.")

    output_tensor = output_visuals['fake_B'].squeeze(0).cpu().detach().numpy() # Shape: [2, F, T]

    # 6. Denormalization
    output_tensor_denorm = output_tensor * global_max_abs_val

    # 7. Complex Reconstruction
    complex_spec_out = output_tensor_denorm[0, :, :] + 1j * output_tensor_denorm[1, :, :]

    # 8. Inverse STFT
    # Use the original length of the segment for ISTFT to avoid artifacts from potential padding
    # If the segment was padded before STFT, use the padded length. Here we assume exact segments.
    output_audio = librosa.istft(complex_spec_out, hop_length=hop_length, win_length=win_length, length=len(segment_audio))

    return output_audio


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # --- Manual Settings specific to audio ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_audio', required=True, help='Path to the input audio file (Domain A)')
    parser.add_argument('--output_audio', required=True, help='Path to save the output audio file (Domain B sound)')
    parser.add_argument('--norm_stats_path', required=True, help='Path to the norm_stats.json file created during preprocessing')
    # Allow overriding options from command line for convenience
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--name', type=str, required=True, help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use.')
    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

    # Parse known args from TestOptions and unknown args for our script
    audio_args, unknown = parser.parse_known_args()

    # Update opt with args from our parser
    opt.checkpoints_dir = audio_args.checkpoints_dir
    opt.name = audio_args.name
    opt.gpu_ids = audio_args.gpu_ids
    opt.model = audio_args.model
    opt.epoch = audio_args.epoch
    # Add any other options you might need to override

    # Load normalization stats
    try:
        with open(audio_args.norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        print(f"Loaded normalization stats: {norm_stats}")
    except Exception as e:
        print(f"Error loading normalization stats from {audio_args.norm_stats_path}: {e}")
        exit()

    # Load the model
    model = load_model(opt)

    # Load input audio
    print(f"Loading input audio: {audio_args.input_audio}")
    y_in, sr_in = librosa.load(audio_args.input_audio, sr=norm_stats['sr'], mono=True)
    print(f"Input audio loaded: {len(y_in)/norm_stats['sr']:.2f} seconds, Sample Rate: {norm_stats['sr']}")

    # --- Segmentation and Processing ---
    segment_len_sec = norm_stats['segment_len_sec']
    segment_len_samples = int(segment_len_sec * norm_stats['sr'])
    hop_length = norm_stats['hop_length'] # Needed for overlap-add calculation

    output_audio_full = np.array([], dtype=np.float32)

    # Simple non-overlapping segmentation for now
    num_segments = int(np.ceil(len(y_in) / segment_len_samples)) # Use ceil to process the last part
    print(f"Processing audio in {num_segments} segments...")

    for i in tqdm(range(num_segments), desc="Generating Audio"):
        start_sample = i * segment_len_samples
        end_sample = start_sample + segment_len_samples
        segment = y_in[start_sample:end_sample]

        # Pad the last segment if it's shorter
        if len(segment) < segment_len_samples:
             segment = np.pad(segment, (0, segment_len_samples - len(segment)), 'constant')

        # Process the segment
        output_segment = process_audio_segment(segment, model, opt, norm_stats)

        # Append to the full output audio
        # If using overlap-add later, this part would be different
        output_audio_full = np.concatenate((output_audio_full, output_segment))

    # Trim the output to the original input length (in case of padding in the last segment)
    output_audio_full = output_audio_full[:len(y_in)]

    # Save the output audio
    print(f"Saving output audio to: {audio_args.output_audio}")
    sf.write(audio_args.output_audio, output_audio_full, norm_stats['sr'])

    print("Inference complete.")
import librosa
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

# --- Constants and Configuration ---
CHANNEL_MAPPING = {
    'R': 'intensity',  # Red channel for audio intensity (loudness)
    'B': 'frequency',  # Green channel for frequency
    'G': 'arbitrary'   # An arbitrary, constant value for the blue channel
}

N_FFT = 2048      # Number of FFT components
HOP_LENGTH = 200   # Hop length for STFT


def generate_custom_spectrogram(audio_path, output_path, mode, min_db, max_db, 
                                segment_len_sec, overlap_sec, figsize, dpi):
    """
    Generates custom spectrogram image(s) from an audio file.
    Can process the whole file or slice it into segments.
    """
    print(f"Loading audio file: {audio_path}")
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # If segment_len is 0, process the whole file as one segment
    if segment_len_sec <= 0:
        print("Processing entire file as a single segment.")
        segments = [y]
    else:
        print(f"Creating segments of {segment_len_sec}s with {overlap_sec}s overlap...")
        segment_samples = int(segment_len_sec * sr)
        overlap_samples = int(overlap_sec * sr)
        hop_between_segments = segment_samples - overlap_samples
        
        segments = []
        start_sample = 0
        while start_sample + segment_samples <= len(y):
            end_sample = start_sample + segment_samples
            segments.append(y[start_sample:end_sample])
            start_sample += hop_between_segments

    if not segments:
        print("Audio file is shorter than the specified segment length. No images generated.")
        return

    # --- Process each segment ---
    output_base, output_ext = os.path.splitext(output_path)
    for i, segment in enumerate(segments):
        stft_result = librosa.stft(segment, n_fft=N_FFT, hop_length=HOP_LENGTH)
        height, width = stft_result.shape
        intensity_db = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)

        if mode == "autoscale":
            min_db_auto, max_db_auto = intensity_db.min(), intensity_db.max()
            print(f"Autoscaled dB range: {min_db_auto} to {max_db_auto}")
            if max_db_auto == min_db_auto:
                intensity_channel = np.zeros_like(intensity_db, dtype=np.uint8)
            else:
                intensity_channel = ((intensity_db - min_db_auto) / (max_db_auto - min_db_auto) * 255).astype(np.uint8)

        elif mode == "scale":
            if max_db == min_db:
                intensity_channel = np.zeros_like(intensity_db, dtype=np.uint8)
            else:
                intensity_channel = ((intensity_db - min_db) / (max_db - min_db) * 255).clip(0, 255).astype(np.uint8)

        elif mode == "cut":
            intensity_channel = np.clip(intensity_db, min_db, max_db)
            if max_db == min_db:
                intensity_channel = np.zeros_like(intensity_db, dtype=np.uint8)
            else:
                intensity_channel = ((intensity_channel - min_db) / (max_db - min_db) * 255).clip(0, 255).astype(np.uint8)



        # FREQUENCY Channel
        freq_gradient = np.linspace(0, 255, height, dtype=np.uint8)
        frequency_channel = np.tile(freq_gradient[:, np.newaxis], (1, width))
        # ARBITRARY Channel
        arbitrary_channel = np.full((height, width), 69, dtype=np.uint8)
        # Map data to RGB
        channel_data = {
            'intensity': intensity_channel, 'frequency': frequency_channel, 'arbitrary': arbitrary_channel
        }
        r, g, b = channel_data[CHANNEL_MAPPING['R']], channel_data[CHANNEL_MAPPING['G']], channel_data[CHANNEL_MAPPING['B']]
        
        rgb_image_array = np.stack([r, g, b], axis=-1)

        # Create and save the plot
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.imshow(rgb_image_array, aspect='auto')
        ax.axis('off')

        # Determine final output path for the segment
        segment_output_path = f"{output_base}_{i:04d}{output_ext}" if len(segments) > 1 else output_path
        fig.savefig(segment_output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig) # IMPORTANT: Close figure to free memory in a loop
        print(f"  -> Saved {segment_output_path}")

    print(f"\nDone! Processed {len(segments)} segment(s).")


def main():
    parser = argparse.ArgumentParser(description="Convert an audio file into custom spectrogram images.", formatter_class=argparse.RawTextHelpFormatter)
    
    # --- File Arguments ---
    parser.add_argument("input_file", type=str, help="Path to the input audio file.")
    parser.add_argument("-o", "--output", type=str, help="Path for the output image file(s).")

    # --- Segmentation Arguments ---
    seg_group = parser.add_argument_group('Segmentation Options')
    seg_group.add_argument("--segment-length", type=float, default=0, help="Length of each segment in seconds. Default: 0 (process whole file).")
    seg_group.add_argument("--overlap", type=float, default=2.0, help="Overlap between segments in seconds. Default: 2.0.")

    # --- Figure Size Arguments ---
    size_group = parser.add_argument_group('Figure Size Options')
    size_group.add_argument("--figsize", type=float, nargs=2, default=[12, 4], metavar=('WIDTH', 'HEIGHT'), help="Figure size in inches. Default: 12 4.")
    size_group.add_argument("--dpi", type=int, default=150, help="Dots Per Inch (DPI) for the output image. Default: 150.")

    # --- Normalization Arguments ---
    norm_group = parser.add_argument_group('Normalization Options')
    norm_group.add_argument("--mode", type=str, default="autoscale", choices=["autoscale", "scale", "cut"], help="Intensity normalization mode.")
    norm_group.add_argument("--min-db", type=float, default=-80.0, help="Min dB for 'scale' or 'cut' mode.")
    norm_group.add_argument("--max-db", type=float, default=0.0, help="Max dB for 'scale' or 'cut' mode.")

    args = parser.parse_args()

    input_path = args.input_file
    if not args.output:
        input_path = os.path.abspath(input_path)
        os.makedirs(os.path.dirname(input_path+"out"), exist_ok=True)
    output_path = args.output if args.output else f"{os.path.splitext(input_path)[0]}.png"
    
    generate_custom_spectrogram(
        input_path, output_path, args.mode, args.min_db, args.max_db,
        args.segment_length, args.overlap, args.figsize, args.dpi
    )

if __name__ == '__main__':
    main()
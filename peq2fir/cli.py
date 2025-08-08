#!/usr/bin/env python3
import argparse
import json
import os
import datetime
import numpy as np
from .converter import PEQtoFIR, parse_autoeq_file
from .output_handler import OutputHandler, WAVOutputHandler, TextOutputHandler, JSONOutputHandler


def save_fir_files(basename: str, output_dir: str, fs: int, fir_coeffs: np.ndarray, 
              num_taps: int, phase_type: str, bit_depth: int, num_channels: int = 1):
    """
    Save FIR coefficients using output handlers
    
    Returns:
        Tuple of output file paths
    """
    # Create output handlers
    wav_handler = WAVOutputHandler(bit_depth=bit_depth)
    text_handler = TextOutputHandler()
    json_handler = JSONOutputHandler()
    
    # Prepare metadata
    metadata = {
        'fs': fs,
        'num_taps': num_taps,
        'phase_type': phase_type,
        'num_channels': num_channels,
        'basename': basename
    }
    
    # Save using handlers
    wav_handler.save(fir_coeffs, metadata, output_dir)
    text_handler.save(fir_coeffs, metadata, output_dir)
    
    # Return paths (implementation in handlers would set these)
    channel_str = "Stereo_" if num_channels == 2 else ""
    wav_path = os.path.join(output_dir, f"{basename}_{channel_str}{phase_type.capitalize()}_{num_taps}taps_{fs}Hz.wav")
    txt_path = os.path.join(output_dir, f"{basename}_{phase_type.capitalize()}_{num_taps}taps_{fs}Hz.txt")
    
    return wav_path, txt_path


def main():
    parser = argparse.ArgumentParser(description='PEQ to FIR Converter CLI')
    parser.add_argument('input', type=str, help='Input PEQ configuration file')
    parser.add_argument('-o', '--output', type=str, default='output',
                        help='Output directory (default: output/)')
    parser.add_argument('--taps', type=int, choices=[2047, 4095, 8191], default=4095,
                        help='Number of FIR filter taps')
    parser.add_argument('--phase', type=str, choices=['linear', 'minimum'], default='linear',
                        help='Phase type of the FIR filter')
    parser.add_argument('--bit-depth', type=int, choices=[16, 24, 32], default=16,
                        help='Output bit depth')
    parser.add_argument('--multiprocess', action='store_true',
                        help='Use multiprocessing for large FIR designs')
    parser.add_argument('--channels', type=int, choices=[1,2], default=1,
                        help='Number of channels (1=mono, 2=stereo)')
    parser.add_argument('--sample-rates', nargs='+', type=int, 
                        default=[44100, 48000], help='Sample rates to generate (default: 44100 48000)')
    parser.add_argument('--no-file-preamp', action='store_false', dest='file_preamp',
                        help='Disable applying preamp from file')
    parser.add_argument('--no-auto-preamp', action='store_false', dest='auto_preamp',
                        help='Disable automatic clipping prevention')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate basename
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    basename = f"{input_name}_FIR" if input_name != 'output' else f"{timestamp}_FIR"
    
    # Parse PEQ filters
    peq_filters = parse_autoeq_file(args.input)
    
    results = {}
    for fs in args.sample_rates:
        print(f"Converting for {fs} Hz...")
        converter = PEQtoFIR(fs=fs, num_taps=args.taps)
        
        # Design FIR filter
        fir_coeffs = converter.design_fir_filter(
            peq_filters,
            use_file_preamp=args.file_preamp,
            use_auto_preamp=args.auto_preamp,
            phase_type=args.phase,
            use_multiprocessing=args.multiprocess
        )

        # Save output files
        wav_path, txt_path = save_fir_files(
            basename, args.output, fs, fir_coeffs,
            args.taps, args.phase, args.bit_depth, num_channels=args.channels
        )
        
        # Analyze results
        analysis = converter.analyze_filter(
            fir_coeffs,
            peq_filters,
            use_file_preamp=args.file_preamp,
            use_auto_preamp=args.auto_preamp
        )
        results[fs] = {
            'latency_ms': analysis['latency_ms'],
            'max_error_db': analysis['max_error_db'],
            'rms_error_db': analysis['rms_error_db']
        }

    # Save metadata JSON
    metadata = {
        'peq_filters': peq_filters,
        'num_taps': args.taps,
        'phase_type': args.phase,
        'bit_depth': args.bit_depth,
        'num_channels': args.channels,
        'file_preamp_applied': args.file_preamp,
        'auto_preamp_applied': args.auto_preamp,
        'sample_rates': args.sample_rates,
        'results': results
    }
    
    # Use JSON output handler
    json_handler = JSONOutputHandler()
    json_handler.save(np.array([]), metadata, args.output)

    # Print completion summary
    print("\nConversion completed!")
    for fs, res in results.items():
        print(f"\n{fs} Hz:")
        print(f"  - Latency: {res['latency_ms']:.1f} ms")
        print(f"  - Max error: {res['max_error_db']:.2f} dB")
        print(f"  - RMS error: {res['rms_error_db']:.2f} dB")
    print(f"\nFiles saved to: {args.output}")

if __name__ == '__main__':
    main()
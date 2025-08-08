import pytest
import os
import numpy as np
from peq2fir.cli import save_fir_files


def test_stereo_output_generation(tmp_path):
    """Verify stereo output has identical channels"""
    # Create test data
    basename = "test"
    output_dir = str(tmp_path)
    fs = 48000
    fir_coeffs = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
    num_taps = 5
    phase_type = "linear"
    bit_depth = 16
    
    # Save stereo output
    wav_path, _ = save_fir_files(
        basename, output_dir, fs, fir_coeffs,
        num_taps, phase_type, bit_depth, num_channels=2
    )
    
    # Verify file exists
    assert os.path.exists(wav_path)
    
    # Load and verify stereo channels
    import scipy.io.wavfile as wavfile
    fs_out, data = wavfile.read(wav_path)
    assert fs_out == fs
    assert data.shape[1] == 2  # Stereo
    assert np.array_equal(data[:, 0], data[:, 1])  # Identical channels


def test_mono_output_generation(tmp_path):
    """Verify mono output has single channel"""
    # Create test data
    basename = "test"
    output_dir = str(tmp_path)
    fs = 48000
    fir_coeffs = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
    num_taps = 5
    phase_type = "linear"
    bit_depth = 16
    
    # Save mono output
    wav_path, _ = save_fir_files(
        basename, output_dir, fs, fir_coeffs,
        num_taps, phase_type, bit_depth, num_channels=1
    )
    
    # Verify file exists
    assert os.path.exists(wav_path)
    
    # Load and verify mono channel
    import scipy.io.wavfile as wavfile
    fs_out, data = wavfile.read(wav_path)
    assert fs_out == fs
    assert len(data.shape) == 1  # Mono
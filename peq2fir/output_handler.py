"""Abstract base classes for output handlers"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import os
import scipy.io.wavfile
import json

class OutputHandler(ABC):
    """Interface for output handlers"""
    
    @abstractmethod
    def save(self, fir_coeffs: np.ndarray, metadata: Dict[str, Any], output_dir: str) -> None:
        """
        Save FIR coefficients and metadata
        
        Args:
            fir_coeffs: FIR filter coefficients
            metadata: Dictionary containing filter metadata
            output_dir: Directory to save output files
        """
        pass


class WAVOutputHandler(OutputHandler):
    """Handler for WAV file output"""
    
    def __init__(self, bit_depth: int = 24):
        self.bit_depth = bit_depth
    
    def save(self, fir_coeffs: np.ndarray, metadata: Dict[str, Any], output_dir: str) -> str:
        """Save FIR coefficients as WAV file and return path"""
        # Form the filename
        channel_str = "Stereo_" if metadata.get('num_channels') == 2 else ""
        filename = f"{metadata['basename']}_{channel_str}{metadata['phase_type'].capitalize()}_{metadata['num_taps']}taps_{metadata['fs']}Hz.wav"
        path = os.path.join(output_dir, filename)
        
        # Convert coefficients to appropriate format
        if self.bit_depth == 32:
            wav_data = fir_coeffs.astype(np.float32)
        elif self.bit_depth == 16:
            # Check for clipping
            if np.max(np.abs(fir_coeffs)) > 1.0:
                print(f"Warning: FIR coefficients exceed [-1, 1] (max: {np.max(np.abs(fir_coeffs)):.3f})")
            wav_data = (fir_coeffs * 32767).astype(np.int16)
        else:  # 24-bit
            if np.max(np.abs(fir_coeffs)) > 1.0:
                print(f"Warning: FIR coefficients exceed [-1, 1] (max: {np.max(np.abs(fir_coeffs)):.3f})")
            wav_data = (fir_coeffs * 8388607).astype(np.int32)

        # If stereo, duplicate to both channels
        if metadata.get('num_channels') == 2:
            wav_data = np.column_stack((wav_data, wav_data))

        # Write WAV file
        scipy.io.wavfile.write(path, metadata['fs'], wav_data)
        return path


class TextOutputHandler(OutputHandler):
    """Handler for text coefficient files"""
    
    def save(self, fir_coeffs: np.ndarray, metadata: Dict[str, Any], output_dir: str) -> str:
        """Save FIR coefficients as text file and return path"""
        filename = f"{metadata['basename']}_{metadata['phase_type'].capitalize()}_{metadata['num_taps']}taps_{metadata['fs']}Hz.txt"
        path = os.path.join(output_dir, filename)
        np.savetxt(path, fir_coeffs, fmt='%.10e')
        return path


class JSONOutputHandler(OutputHandler):
    """Handler for JSON metadata files"""
    
    def save(self, fir_coeffs: np.ndarray, metadata: Dict[str, Any], output_dir: str) -> str:
        """Save metadata as JSON file and return path"""
        filename = metadata.get('json_filename', 'filter_metadata.json')
        path = os.path.join(output_dir, filename)
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        return path
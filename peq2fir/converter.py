import numpy as np
from scipy.signal import firwin2, freqz, minimum_phase, group_delay  # type: ignore
from typing import List, Dict, Tuple, Optional, cast
from .output_handler import OutputHandler
from .exceptions import *

class PEQtoFIR:
    """Convert AutoEQ-style Parametric EQ settings to FIR filter coefficients"""
    
    def __init__(self, fs: int = 48000, num_taps: int = 4097):
        # Validate parameters
        if fs <= 0:
            raise InvalidSampleRateError(fs)
        if num_taps % 2 == 0 or num_taps < 3:
            raise InvalidTapCountError(num_taps, "Number of taps must be odd and at least 3")
        """
        Initialize the converter
        
        Args:
            fs: Sampling frequency in Hz
            num_taps: Number of FIR filter taps (should be odd for Type 1 linear phase)
        """
        self.fs = fs
        self.nyquist = fs / 2
        self.num_taps = num_taps
        
    def biquad_peaking_response(self, frequencies: np.ndarray, fc: float, Q: float, gain_db: float) -> np.ndarray:
        """
        Calculate frequency response of a peaking EQ filter
        
        Args:
            frequencies: Array of frequencies in Hz
            fc: Center frequency in Hz
            Q: Quality factor
            gain_db: Gain in dB
        
        Returns:
            Complex frequency response
        
        Raises:
            FilterValidationError: If filter parameters are invalid
        """
        # Validate parameters
        if fc <= 0 or fc >= self.nyquist:
            raise FilterValidationError("peaking", {"fc": fc, "Q": Q, "gain_db": gain_db}, 
                                      f"Center frequency must be between 0 and {self.nyquist} Hz")
        if Q <= 0:
            raise FilterValidationError("peaking", {"fc": fc, "Q": Q, "gain_db": gain_db}, 
                                      "Quality factor must be positive")
        if np.max(frequencies) > self.nyquist + 1e-6:  # Allow small floating-point tolerance
            raise FilterValidationError("peaking", {"fc": fc, "Q": Q, "gain_db": gain_db}, 
                                      f"Frequencies must be below Nyquist ({self.nyquist} Hz)")
        w = 2 * np.pi * frequencies / self.fs
        wc = 2 * np.pi * fc / self.fs
        
        A = 10**(gain_db / 40)
        alpha = np.sin(wc) / (2 * Q)
        
        # Peaking EQ coefficients
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(wc)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(wc)
        a2 = 1 - alpha / A
        
        # Normalize
        b0, b1, b2 = b0/a0, b1/a0, b2/a0
        a1, a2 = a1/a0, a2/a0
        
        # Calculate frequency response
        z = np.exp(-1j * w)
        H = (b0 + b1 * z + b2 * z**2) / (1 + a1 * z + a2 * z**2)
        
        return H
    
    def biquad_highshelf_response(self, frequencies: np.ndarray, fc: float, Q: float, gain_db: float) -> np.ndarray:
        """
        Calculate frequency response of a high shelf filter
        
        Args:
            frequencies: Array of frequencies in Hz
            fc: Cutoff frequency in Hz
            Q: Quality factor (typically sqrt(2)/2 ≈ 0.707)
            gain_db: Gain in dB
        
        Returns:
            Complex frequency response
        
        Raises:
            FilterValidationError: If filter parameters are invalid
        """
        # Validate parameters
        if fc <= 0 or fc >= self.nyquist:
            raise FilterValidationError("highshelf", {"fc": fc, "Q": Q, "gain_db": gain_db}, 
                                      f"Cutoff frequency must be between 0 and {self.nyquist} Hz")
        if Q <= 0:
            raise FilterValidationError("highshelf", {"fc": fc, "Q": Q, "gain_db": gain_db}, 
                                      "Quality factor must be positive")
        if np.max(frequencies) > self.nyquist + 1e-6:  # Allow small floating-point tolerance
            raise FilterValidationError("highshelf", {"fc": fc, "Q": Q, "gain_db": gain_db}, 
                                      f"Frequencies must be below Nyquist ({self.nyquist} Hz)")
        w = 2 * np.pi * frequencies / self.fs
        wc = 2 * np.pi * fc / self.fs
        
        A = 10**(gain_db / 40)
        alpha = np.sin(wc) / (2 * Q)
        
        # High shelf coefficients
        b0 = A * ((A + 1) + (A - 1) * np.cos(wc) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(wc))
        b2 = A * ((A + 1) + (A - 1) * np.cos(wc) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(wc) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(wc))
        a2 = (A + 1) - (A - 1) * np.cos(wc) - 2 * np.sqrt(A) * alpha
        
        # Normalize
        b0, b1, b2 = b0/a0, b1/a0, b2/a0
        a1, a2 = a1/a0, a2/a0
        
        # Calculate frequency response
        z = np.exp(-1j * w)
        H = (b0 + b1 * z + b2 * z**2) / (1 + a1 * z + a2 * z**2)
        
        return H
    
    def biquad_lowshelf_response(self, frequencies: np.ndarray, fc: float, Q: float, gain_db: float) -> np.ndarray:
        """
        Calculate frequency response of a low shelf filter
        
        Args:
            frequencies: Array of frequencies in Hz
            fc: Cutoff frequency in Hz
            Q: Quality factor (typically sqrt(2)/2 ≈ 0.707)
            gain_db: Gain in dB
        
        Returns:
            Complex frequency response
        
        Raises:
            FilterValidationError: If filter parameters are invalid
        """
        # Validate parameters
        if fc <= 0 or fc >= self.nyquist:
            raise FilterValidationError("lowshelf", {"fc": fc, "Q": Q, "gain_db": gain_db}, 
                                      f"Cutoff frequency must be between 0 and {self.nyquist} Hz")
        if Q <= 0:
            raise FilterValidationError("lowshelf", {"fc": fc, "Q": Q, "gain_db": gain_db}, 
                                      "Quality factor must be positive")
        if np.max(frequencies) > self.nyquist + 1e-6:  # Allow small floating-point tolerance
            raise FilterValidationError("lowshelf", {"fc": fc, "Q": Q, "gain_db": gain_db}, 
                                      f"Frequencies must be below Nyquist ({self.nyquist} Hz)")
        w = 2 * np.pi * frequencies / self.fs
        wc = 2 * np.pi * fc / self.fs
        
        A = 10**(gain_db / 40)
        alpha = np.sin(wc) / (2 * Q)
        
        # Low shelf coefficients
        b0 = A * ((A + 1) - (A - 1) * np.cos(wc) + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(wc))
        b2 = A * ((A + 1) - (A - 1) * np.cos(wc) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * np.cos(wc) + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(wc))
        a2 = (A + 1) + (A - 1) * np.cos(wc) - 2 * np.sqrt(A) * alpha
        
        # Normalize
        b0, b1, b2 = b0/a0, b1/a0, b2/a0
        a1, a2 = a1/a0, a2/a0
        
        # Calculate frequency response
        z = np.exp(-1j * w)
        H = (b0 + b1 * z + b2 * z**2) / (1 + a1 * z + a2 * z**2)
        
        return H
    
    def calculate_target_response(self, peq_filters: List[Dict], num_points: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the combined frequency response of all PEQ filters
        
        Args:
            peq_filters: List of dictionaries with keys 'type', 'freq', 'q', 'gain'
            num_points: Number of frequency points for evaluation
        
        Returns:
            frequencies: Frequency axis in Hz
            response_db: Combined response in dB
        """
        # Create logarithmic frequency axis
        freq_points = np.logspace(np.log10(20), np.log10(self.nyquist), num_points)
        frequencies = np.concatenate(([0], freq_points))
        
        # Initialize combined response
        combined_response = np.ones_like(frequencies, dtype=complex)
        
        # Calculate response for each filter
        for filt in peq_filters:
            filter_type = filt['type'].lower()
            fc = filt['freq']
            Q = filt['q']
            gain_db = filt['gain']
            
            if filter_type in ['peaking', 'peak', 'bell']:
                H = self.biquad_peaking_response(frequencies, fc, Q, gain_db)
            elif filter_type in ['highshelf', 'high_shelf', 'hs']:
                H = self.biquad_highshelf_response(frequencies, fc, Q, gain_db)
            elif filter_type in ['lowshelf', 'low_shelf', 'ls']:
                H = self.biquad_lowshelf_response(frequencies, fc, Q, gain_db)
            else:
                print(f"Unknown filter type: {filter_type}, skipping...")
                continue
            
            combined_response *= H
        
        # Convert to dB
        response_db = 20 * np.log10(np.abs(combined_response))
        
        return frequencies, response_db
    
    def get_final_target_response(self, peq_filters: List[Dict], 
                                use_file_preamp: bool = True, 
                                use_auto_preamp: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the final target response with all preamps applied.
        This is the single source of truth for target response calculation.
        
        Args:
            peq_filters: List of all filters including preamp entries
            use_file_preamp: Whether to apply preamp from file
            use_auto_preamp: Whether to apply automatic clipping prevention
            
        Returns:
            frequencies: Frequency axis in Hz
            response_db: Final response in dB with all preamps applied
        """
        # Extract file preamp value
        file_preamp_db = next((f['gain'] for f in peq_filters if f.get('type') == 'preamp'), 0.0)
        
        # Extract actual filters
        actual_filters = [f for f in peq_filters if f.get('type') != 'preamp']
        
        # Calculate base response
        frequencies, response_db = self.calculate_target_response(actual_filters)
        
        # Apply file preamp if enabled
        if use_file_preamp and file_preamp_db != 0:
            response_db += file_preamp_db
            
        # Apply auto preamp if enabled
        if use_auto_preamp:
            max_gain = np.max(response_db)
            if max_gain > 0:
                response_db -= max_gain
                
        return frequencies, response_db
    
    def design_fir_filter(self, peq_filters: List[Dict], 
                         use_file_preamp: bool = True,
                         use_auto_preamp: bool = True,
                         phase_type: str = 'linear', 
                         window: str = 'hamming',
                         use_multiprocessing: bool = False) -> np.ndarray:
        """
        Design FIR filter from PEQ settings
        
        Args:
            peq_filters: List of all filters including preamp entries
            use_file_preamp: Whether to apply preamp from file
            use_auto_preamp: Whether to apply automatic clipping prevention
            phase_type: 'linear' or 'minimum' phase
            window: Window function for firwin2
        
        Returns:
            FIR filter coefficients
        """
        # Get the final target response using the unified method
        frequencies, response_db = self.get_final_target_response(
            peq_filters, use_file_preamp, use_auto_preamp
        )
        
        print(f"Final target response range: {np.min(response_db):.1f} to {np.max(response_db):.1f} dB")
        
        # Convert to linear gain
        gain_linear = 10**(response_db / 20)
        
        # Prepare frequency and gain arrays
        nyq = self.fs / 2
        
        # Build frequency array with 0 and Nyquist
        freq = np.r_[0, frequencies[frequencies < nyq], nyq]
        gain = np.r_[gain_linear[0], gain_linear[frequencies < nyq], gain_linear[-1]]
        
        # Remove duplicate frequencies (keep first occurrence)
        freq, idx = np.unique(freq, return_index=True)
        gain = gain[idx]
        
        # Design FIR filter with optional multiprocessing
        if use_multiprocessing and self.num_taps > 2048:
            # Use multiprocessing for large tap counts
            from concurrent.futures import ProcessPoolExecutor
            
            def design_segment(segment):
                return firwin2(len(segment), freq, gain, fs=self.fs, window=window)
                
            # Split into segments for parallel processing
            segment_size = max(1024, self.num_taps // 4)
            segments = [np.arange(i, min(i+segment_size, self.num_taps)) 
                       for i in range(0, self.num_taps, segment_size)]
            
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(design_segment, segments))
                
            fir_coeffs = np.concatenate(results)
        else:
            # Single-threaded design
            fir_coeffs = firwin2(self.num_taps, freq, gain, fs=self.fs, window=window)
        
        # Convert to minimum phase if requested
        if phase_type.lower() == 'minimum':
            fir_coeffs = minimum_phase(fir_coeffs, method='hilbert')
        
        # Verify the designed filter (NO normalization!)
        w_test, h_test = freqz(fir_coeffs, worN=1024, fs=self.fs)
        test_db = 20 * np.log10(np.abs(h_test))
        print(f"Final FIR response range: {np.min(test_db):.1f} to {np.max(test_db):.1f} dB")
        
        return cast(np.ndarray, fir_coeffs)
    
    def check_phase_linearity(self, fir_coeffs: np.ndarray) -> Dict:
        """
        Check if the FIR filter has linear phase
        
        Args:
            fir_coeffs: FIR filter coefficients
            
        Returns:
            Dictionary with phase linearity metrics
        """
        N = len(fir_coeffs)
        center = (N - 1) // 2
        
        # 1. Symmetry error
        left_half = fir_coeffs[:center]
        right_half = fir_coeffs[-center:][::-1]
        symmetry_error = np.mean((left_half - right_half)**2)
        
        # 2. Impulse peak location
        peak_idx = np.argmax(np.abs(fir_coeffs))
        peak_offset = peak_idx - center
        
        # 3. Group delay check
        w, h = freqz(fir_coeffs, worN=1024, fs=self.fs)
        _, gd = group_delay((fir_coeffs, [1]), w=w, fs=self.fs)  # type: ignore[arg-type]
        gd_std = np.std(gd[10:-10]) if len(gd) > 20 else np.std(gd)
        
        # 4. Overall assessment
        is_linear_phase = (
            symmetry_error < 1e-10 and
            abs(peak_offset) < 2 and
            gd_std < 0.01
        )
        
        return {
            'symmetry_error': float(symmetry_error),
            'peak_offset': int(peak_offset),
            'group_delay_std': float(gd_std),
            'is_linear_phase': bool(is_linear_phase),
            'phase_type': 'Linear Phase' if is_linear_phase else 'Non-linear Phase'
        }
    
    def analyze_filter(self, fir_coeffs: np.ndarray, peq_filters: List[Dict], 
                      use_file_preamp: bool = True,
                      use_auto_preamp: bool = True) -> Dict:
        """
        Analyze the designed FIR filter and compare with target
        
        Args:
            fir_coeffs: FIR filter coefficients
            peq_filters: Original PEQ settings
            plot: Whether to create plots
        
        Returns:
            Dictionary with analysis results
        """
        # Calculate actual FIR response
        w, h = freqz(fir_coeffs, worN=8192, fs=self.fs)
        actual_db = 20 * np.log10(np.abs(h))
        phase = np.unwrap(np.angle(h))
        
        # Get the final target response using the unified method
        target_freq, target_db = self.get_final_target_response(
            peq_filters, use_file_preamp, use_auto_preamp
        )
        
        # Interpolate target to match actual frequencies
        target_interp = np.interp(w, target_freq, target_db)  # type: ignore[operator]
        
        # Calculate metrics
        max_error = np.max(np.abs(actual_db - target_interp))
        rms_error = np.sqrt(np.mean((actual_db - target_interp)**2))
        
        # Calculate group delay
        _, gd = freqz(fir_coeffs, worN=8192, fs=self.fs, whole=False)
        group_delay_ms = -np.diff(np.unwrap(np.angle(h))) / (2 * np.pi * np.diff(cast(np.ndarray, w))) * 1000  # type: ignore[arg-type]
        
        # Phase linearity analysis
        center = (len(fir_coeffs) - 1) // 2
        
        # 1. Symmetry error
        left_half = fir_coeffs[:center]
        right_half = fir_coeffs[-center:][::-1]
        symmetry_error = np.mean((left_half - right_half)**2)
        
        # 2. Group delay standard deviation
        gd_std = np.std(gd[10:-10]) if len(gd) > 20 else np.std(gd)
        
        # 3. Phase linearity (R²)
        from sklearn.metrics import r2_score  # type: ignore
        try:
            linear_fit = np.polyfit(w[10:-10], phase[10:-10], 1)  # type: ignore[operator]
            phase_linear = np.polyval(linear_fit, w[10:-10])  # type: ignore[arg-type]
            phase_r2 = r2_score(phase[10:-10], phase_linear)
        except:
            phase_r2 = 0
        
        # 4. Maximum phase deviation
        ideal_delay = center / self.fs
        ideal_phase = -w * ideal_delay * 2 * np.pi
        phase_deviation = np.abs(phase - ideal_phase)
        max_phase_deviation_deg = np.degrees(np.max(phase_deviation[10:-10]))
        
        # 5. Impulse response peak location
        peak_idx = np.argmax(np.abs(fir_coeffs))
        peak_offset = peak_idx - center
        
        
        return {
            'max_error_db': max_error,
            'rms_error_db': rms_error,
            'latency_ms': (self.num_taps - 1) / (2 * self.fs) * 1000,
            'frequencies': w,
            'target_response_db': target_interp,
            'actual_response_db': actual_db
        }

def parse_autoeq_file(filename: str) -> List[Dict]:
    """
    Parse AutoEQ ParametricEQ.txt file
    
    Args:
        filename: Path to ParametricEQ.txt file
    
    Returns:
        List of filter dictionaries
    """
    filters = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('Filter'):
            # Parse AutoEQ format: Filter 1: ON PK Fc 105 Hz Gain -3.0 dB Q 1.41
            parts = line.split()
            if len(parts) >= 9 and parts[2] == 'ON':
                filt = {
                    'type': parts[3].lower(),
                    'freq': float(parts[5]),
                    'gain': float(parts[8]),
                    'q': float(parts[11]) if len(parts) > 11 else 0.707
                }
                
                # Map AutoEQ types to our types
                if filt['type'] == 'pk':
                    filt['type'] = 'peaking'
                elif filt['type'] == 'hs':
                    filt['type'] = 'highshelf'
                elif filt['type'] == 'ls':
                    filt['type'] = 'lowshelf'
                
                filters.append(filt)
    
    return filters


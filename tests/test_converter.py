import pytest
import numpy as np
from peq2fir.converter import PEQtoFIR


def test_peaking_response_accuracy():
    """Verify peaking EQ frequency response at center frequency"""
    converter = PEQtoFIR()
    frequencies = np.array([1000.0])  # Test at 1kHz
    response = converter.biquad_peaking_response(frequencies, fc=1000, Q=1.41, gain_db=-3.0)
    
    # Expected magnitude at center frequency: 10^(-3/20) = ~0.7079
    expected_mag = 10**(-3.0/20)
    measured_mag = np.abs(response[0])
    
    assert np.isclose(measured_mag, expected_mag, atol=1e-3), \
        f"Expected {expected_mag:.4f}, got {measured_mag:.4f}"


def test_highshelf_response_at_cutoff():
    """Verify high shelf response at cutoff frequency"""
    converter = PEQtoFIR()
    frequencies = np.array([10000.0])  # Test at 10kHz
    response = converter.biquad_highshelf_response(frequencies, fc=10000, Q=0.707, gain_db=6.0)
    
    # Expected magnitude: 10^(6/20) * 0.707 = 3.0 * 0.707 = ~2.12
    expected_mag = 10**(6.0/20) * np.sqrt(0.5)
    measured_mag = np.abs(response[0])
    
    assert np.isclose(measured_mag, expected_mag, atol=1e-2)


def test_lowshelf_response_at_cutoff():
    """Verify low shelf response at cutoff frequency"""
    converter = PEQtoFIR()
    frequencies = np.array([100.0])  # Test at 100Hz
    response = converter.biquad_lowshelf_response(frequencies, fc=100, Q=0.707, gain_db=-6.0)
    
    # Expected magnitude: 10^(-6/20) * 0.707 = 0.5 * 0.707 = ~0.353
    expected_mag = 10**(-6.0/40)
    measured_mag = np.abs(response[0])
    
    assert np.isclose(measured_mag, expected_mag, atol=1e-2)


def test_design_fir_linear_phase():
    """Verify basic FIR design works without errors"""
    converter = PEQtoFIR(fs=48000, num_taps=101)
    peq_filters = [
        {'type': 'peaking', 'freq': 1000, 'q': 1.41, 'gain': -3.0},
        {'type': 'preamp', 'gain': 0.0}
    ]
    
    fir_coeffs = converter.design_fir_filter(
        peq_filters,
        use_file_preamp=False,
        use_auto_preamp=False,
        phase_type='linear'
    )
    
    assert isinstance(fir_coeffs, np.ndarray)
    assert len(fir_coeffs) == 101
    assert np.isclose(np.sum(fir_coeffs), 1.0, atol=0.1)  # Approximate unity gain


def test_preamp_handling():
    """Verify preamp is correctly applied in target response"""
    converter = PEQtoFIR()
    peq_filters = [
        {'type': 'preamp', 'gain': 3.0},
        {'type': 'peaking', 'freq': 1000, 'q': 1.41, 'gain': 0.0}
    ]
    
    # With file preamp enabled
    _, response_with = converter.get_final_target_response(
        peq_filters,
        use_file_preamp=True,
        use_auto_preamp=False
    )
    assert np.isclose(np.max(response_with), 3.0, atol=0.1)

    # With file preamp disabled
    _, response_without = converter.get_final_target_response(
        peq_filters,
        use_file_preamp=False,
        use_auto_preamp=False
    )
    assert np.isclose(np.max(response_without), 0.0, atol=0.1)


def test_auto_preamp_clipping_prevention():
    """Verify auto preamp correctly normalizes response"""
    converter = PEQtoFIR()
    peq_filters = [
        {'type': 'peaking', 'freq': 1000, 'q': 1.41, 'gain': 5.0}
    ]
    
    # Without auto preamp
    _, response_no_auto = converter.get_final_target_response(
        peq_filters,
        use_file_preamp=False,
        use_auto_preamp=False
    )
    assert np.isclose(np.max(response_no_auto), 5.0, atol=0.1)

    # With auto preamp
    _, response_with_auto = converter.get_final_target_response(
        peq_filters,
        use_file_preamp=False,
        use_auto_preamp=True
    )
    assert np.isclose(np.max(response_with_auto), 0.0, atol=0.1)
    assert np.isclose(np.min(response_with_auto), -5.0, atol=0.1)


def test_invalid_filter_type():
    """Verify invalid filter types are skipped"""
    converter = PEQtoFIR()
    peq_filters = [
        {'type': 'invalid', 'freq': 1000, 'q': 1.41, 'gain': -3.0},
        {'type': 'peaking', 'freq': 1000, 'q': 1.41, 'gain': 0.0}
    ]
    
    _, response = converter.get_final_target_response(peq_filters)
    # Should only have response from the valid peaking filter
    assert np.isclose(np.max(response), 0.0, atol=0.1)


def test_minimum_phase_conversion():
    """Verify minimum phase conversion alters impulse location"""
    # Create linear phase FIR
    linear_converter = PEQtoFIR(num_taps=101)
    linear_coeffs = linear_converter.design_fir_filter([
        {'type': 'peaking', 'freq': 1000, 'q': 1.41, 'gain': 0.0}
    ])
    
    # Create minimum phase FIR with same magnitude
    min_converter = PEQtoFIR(num_taps=101)
    min_coeffs = min_converter.design_fir_filter([
        {'type': 'peaking', 'freq': 1000, 'q': 1.41, 'gain': 0.0}
    ], phase_type='minimum')
    
    # Check center of energy is earlier in minimum phase
    linear_center = (len(linear_coeffs) - 1) / 2
    min_center = np.argmax(np.abs(min_coeffs))
    assert min_center < linear_center - 10  # Should be significantly earlier


def test_analyze_filter_metrics():
    """Verify analysis reports reasonable metrics"""
    converter = PEQtoFIR()
    peq_filters = [
        {'type': 'peaking', 'freq': 1000, 'q': 1.41, 'gain': -3.0}
    ]
    fir_coeffs = converter.design_fir_filter(peq_filters)
    analysis = converter.analyze_filter(fir_coeffs, peq_filters)
    
    assert 'max_error_db' in analysis
    assert 'rms_error_db' in analysis
    assert 'latency_ms' in analysis
    assert analysis['max_error_db'] < 1.0  # Should be within reasonable error bounds
    """Verify analysis reports reasonable metrics"""
    converter = PEQtoFIR()
    peq_filters = [
        {'type': 'peaking', 'freq': 1000, 'q': 1.41, 'gain': -3.0}
    ]
    fir_coeffs = converter.design_fir_filter(peq_filters)
    analysis = converter.analyze_filter(fir_coeffs, peq_filters)
    
    assert 'max_error_db' in analysis
    assert 'rms_error_db' in analysis
    assert 'latency_ms' in analysis
    assert analysis['max_error_db'] < 1.0  # Should be within reasonable error bounds
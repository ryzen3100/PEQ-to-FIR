"""Custom exceptions for PEQ-to-FIR application"""

class PEQtoFIRException(Exception):
    """Base class for all PEQ-to-FIR exceptions"""
    pass

class FilterValidationError(PEQtoFIRException):
    """Raised when a filter configuration is invalid"""
    def __init__(self, filter_type, filter_params, message="Invalid filter configuration"):
        self.filter_type = filter_type
        self.filter_params = filter_params
        super().__init__(f"{message} for {filter_type} filter with params {filter_params}")

class InvalidSampleRateError(PEQtoFIRException):
    """Raised when an invalid sample rate is provided"""
    def __init__(self, sample_rate, message="Invalid sample rate"):
        self.sample_rate = sample_rate
        super().__init__(f"{message}: {sample_rate} Hz")

class InvalidTapCountError(PEQtoFIRException):
    """Raised when an invalid tap count is provided"""
    def __init__(self, tap_count, message="Invalid tap count"):
        self.tap_count = tap_count
        super().__init__(f"{message}: {tap_count}")

class InputFileError(PEQtoFIRException):
    """Raised when there's an issue with input file processing"""
    def __init__(self, file_path, message="Error processing input file"):
        self.file_path = file_path
        super().__init__(f"{message}: {file_path}")

class PhaseTypeError(PEQtoFIRException):
    """Raised when an invalid phase type is provided"""
    def __init__(self, phase_type, message="Invalid phase type"):
        self.phase_type = phase_type
        super().__init__(f"{message}: {phase_type}")
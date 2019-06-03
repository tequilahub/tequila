class OpenVQEException(Exception):
    """
    Base class for other exceptions
    """
    pass

class ParameterError(OpenVQEException):
    """
    Raised when a specific choice of a parameter is not accepted
    """
    pass

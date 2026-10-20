class GIGNNError(Exception):
    """Base exception for GIGNN."""


class ValidationError(GIGNNError, ValueError):
    """Raised when user-provided data violates the input contract."""


class NotFittedError(GIGNNError, RuntimeError):
    """Raised when an estimator is used before fitting."""


class EdgeWeightWarning(UserWarning):
    """Emitted when Gaussian edge weights collapse and neighbors stop mattering."""

"""Inference protocol failures must never become trading fallbacks."""


class InferenceError(RuntimeError):
    """A service failure that must not silently turn into an offline decision."""


class InferenceContractError(InferenceError):
    """The server did not deliver a complete, correctly separated final answer."""


class TransientInferenceError(InferenceError):
    """A temporary service failure exhausted its request retries."""


class InferenceServiceError(InferenceError):
    """Authentication, invalid requests, and other non-retryable service errors."""

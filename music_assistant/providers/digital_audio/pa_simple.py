"""
PulseAudio simple API bindings for the S/PDIF Audio Out provider.

Copied from local_audio/pa_simple with pa_buffer_attr support added
(matches multichannel_audio/pa_simple).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os

# ---------------------------------------------------------------------------
# Load libpulse-simple
# ---------------------------------------------------------------------------

_lib_name = ctypes.util.find_library("pulse-simple") or "libpulse-simple.so.0"
_lib = ctypes.CDLL(_lib_name)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PA_STREAM_PLAYBACK = 1

# Sample formats
PA_SAMPLE_S16LE = 3
PA_SAMPLE_S24_32LE = 8  # S24 in lower 24 bits of 32-bit word
PA_SAMPLE_S32LE = 5
PA_SAMPLE_FLOAT32LE = 6

_PA_FORMAT_MAP: dict[tuple[int, int], int] = {
    (16, 2): PA_SAMPLE_S16LE,
    (24, 4): PA_SAMPLE_S24_32LE,
    (32, 4): PA_SAMPLE_S32LE,
}


def pa_format_for(bit_depth: int) -> int:
    """Return the PA sample format constant for *bit_depth* bits."""
    if bit_depth <= 16:
        return PA_SAMPLE_S16LE
    if bit_depth <= 24:
        return PA_SAMPLE_S24_32LE
    return PA_SAMPLE_S32LE


# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------


class pa_sample_spec(ctypes.Structure):
    _fields_ = [
        ("format", ctypes.c_int),
        ("rate", ctypes.c_uint32),
        ("channels", ctypes.c_uint8),
    ]


class pa_buffer_attr(ctypes.Structure):
    """PulseAudio buffer attributes for tuning latency / jitter absorption."""

    _fields_ = [
        ("maxlength", ctypes.c_uint32),
        ("tlength", ctypes.c_uint32),  # playback target length
        ("prebuf", ctypes.c_uint32),  # pre-buffering before playback starts
        ("minreq", ctypes.c_uint32),  # minimum request size
        ("fragsize", ctypes.c_uint32),  # recording fragment size (ignored for playback)
    ]


# PA_USEC_INVALID sentinel — use (uint32)-1 to mean "let PA choose"
PA_USEC_INVALID = ctypes.c_uint32(-1).value


# ---------------------------------------------------------------------------
# Function signatures
# ---------------------------------------------------------------------------

_lib.pa_simple_new.restype = ctypes.c_void_p
_lib.pa_simple_new.argtypes = [
    ctypes.c_char_p,  # server
    ctypes.c_char_p,  # name
    ctypes.c_int,  # dir (PA_STREAM_PLAYBACK)
    ctypes.c_char_p,  # dev (sink name)
    ctypes.c_char_p,  # stream_name
    ctypes.POINTER(pa_sample_spec),
    ctypes.c_void_p,  # channel map (NULL → default)
    ctypes.POINTER(pa_buffer_attr),  # buffer attributes (NULL → default)
    ctypes.POINTER(ctypes.c_int),  # error output
]

_lib.pa_simple_write.restype = ctypes.c_int
_lib.pa_simple_write.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int),
]

_lib.pa_simple_drain.restype = ctypes.c_int
_lib.pa_simple_drain.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
]

_lib.pa_simple_free.restype = None
_lib.pa_simple_free.argtypes = [ctypes.c_void_p]

_lib.pa_strerror.restype = ctypes.c_char_p
_lib.pa_strerror.argtypes = [ctypes.c_int]


# ---------------------------------------------------------------------------
# Python helpers
# ---------------------------------------------------------------------------


def pa_strerror(err: int) -> str:
    """Return a human-readable PulseAudio error string."""
    return _lib.pa_strerror(err).decode()


def pa_simple_new(
    server: str | None,
    app_name: str,
    sink_name: str,
    stream_name: str,
    sample_format: int,
    sample_rate: int,
    channels: int,
    buffer_msec: int = 0,
) -> tuple[ctypes.c_void_p, int | None]:
    """
    Open a PulseAudio simple playback stream.

    Returns ``(stream_ptr, None)`` on success or ``(None, error_code)`` on
    failure.  *buffer_msec* sets the target playback buffer length; 0 means
    let PA choose the default.
    """
    spec = pa_sample_spec(format=sample_format, rate=sample_rate, channels=channels)
    err = ctypes.c_int(0)

    attr_ptr: ctypes.POINTER(pa_buffer_attr) | None = None  # type: ignore[type-arg]
    if buffer_msec > 0:
        bytes_per_ms = (sample_rate // 1000) * channels * ctypes.sizeof(ctypes.c_int32)
        tlength = buffer_msec * bytes_per_ms
        attr = pa_buffer_attr(
            maxlength=PA_USEC_INVALID,
            tlength=tlength,
            prebuf=PA_USEC_INVALID,
            minreq=PA_USEC_INVALID,
            fragsize=PA_USEC_INVALID,
        )
        attr_ptr = ctypes.byref(attr)  # type: ignore[assignment]

    pulse_server = os.environ.get("PULSE_SERVER", "")
    server_arg = pulse_server.encode() if pulse_server else None

    stream = _lib.pa_simple_new(
        server_arg,
        app_name.encode(),
        PA_STREAM_PLAYBACK,
        sink_name.encode(),
        stream_name.encode(),
        ctypes.byref(spec),
        None,
        attr_ptr,
        ctypes.byref(err),
    )

    if stream is None:
        return None, err.value
    return ctypes.c_void_p(stream), None


def pa_simple_write(stream: ctypes.c_void_p, data: bytes) -> int | None:
    """
    Write *data* to an open PA stream.

    Returns ``None`` on success or an error code on failure.
    """
    err = ctypes.c_int(0)
    rc = _lib.pa_simple_write(
        stream,
        ctypes.c_char_p(data),
        ctypes.c_size_t(len(data)),
        ctypes.byref(err),
    )
    return err.value if rc < 0 else None


def pa_simple_drain(stream: ctypes.c_void_p) -> int | None:
    """Drain the PA stream.  Returns ``None`` on success or an error code."""
    err = ctypes.c_int(0)
    rc = _lib.pa_simple_drain(stream, ctypes.byref(err))
    return err.value if rc < 0 else None


def pa_simple_free(stream: ctypes.c_void_p) -> None:
    """Free a PA simple stream."""
    _lib.pa_simple_free(stream)

"""
PulseAudio passthrough stream for IEC 61937 encoded audio (AC3, DTS, E-AC3).

Uses the PA async API (pa_threaded_mainloop + pa_stream_new_extended) to open
a passthrough stream with PA_ENCODING_AC3_IEC61937 (or DTS/E-AC3).  The simple
API (pa_simple) only supports PCM and will cause PA to resample the bitstream.

Usage (blocking, call from executor thread):
    stream = PAPassthroughStream(sink_name, encoding="ac3", sample_rate=48000)
    stream.open()          # connects to PA, blocks until ready
    stream.write(data)     # write IEC 61937 bytes
    stream.drain()         # flush before close
    stream.close()         # disconnect and free
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import threading
import time

# ---------------------------------------------------------------------------
# Load libpulse (async API — NOT libpulse-simple)
# ---------------------------------------------------------------------------

_lib_name = ctypes.util.find_library("pulse") or "libpulse.so.0"
_lib = ctypes.CDLL(_lib_name)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PA_ENCODING_PCM = 1
PA_ENCODING_AC3_IEC61937 = 2
PA_ENCODING_EAC3_IEC61937 = 3
PA_ENCODING_DTS_IEC61937 = 4

ENCODING_MAP = {
    "ac3": PA_ENCODING_AC3_IEC61937,
    "eac3": PA_ENCODING_EAC3_IEC61937,
    "dts": PA_ENCODING_DTS_IEC61937,
}

PA_STREAM_PLAYBACK = 1
PA_STREAM_READY = 3
PA_STREAM_FAILED = 4
PA_STREAM_TERMINATED = 5

PA_CONTEXT_READY = 4
PA_CONTEXT_FAILED = 5
PA_CONTEXT_TERMINATED = 6

# pa_stream_flags
PA_STREAM_ADJUST_LATENCY = 0x2000
PA_STREAM_AUTO_TIMING_UPDATE = 0x40

# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------


class pa_sample_spec(ctypes.Structure):
    _fields_ = [
        ("format", ctypes.c_int),   # PA_SAMPLE_S16LE = 3
        ("rate", ctypes.c_uint32),
        ("channels", ctypes.c_uint8),
    ]


class pa_buffer_attr(ctypes.Structure):
    _fields_ = [
        ("maxlength", ctypes.c_uint32),
        ("tlength", ctypes.c_uint32),
        ("prebuf", ctypes.c_uint32),
        ("minreq", ctypes.c_uint32),
        ("fragsize", ctypes.c_uint32),
    ]


# ---------------------------------------------------------------------------
# Function signatures — threaded mainloop
# ---------------------------------------------------------------------------

_lib.pa_threaded_mainloop_new.restype = ctypes.c_void_p
_lib.pa_threaded_mainloop_get_api.restype = ctypes.c_void_p
_lib.pa_threaded_mainloop_get_api.argtypes = [ctypes.c_void_p]
_lib.pa_threaded_mainloop_start.restype = ctypes.c_int
_lib.pa_threaded_mainloop_start.argtypes = [ctypes.c_void_p]
_lib.pa_threaded_mainloop_stop.restype = None
_lib.pa_threaded_mainloop_stop.argtypes = [ctypes.c_void_p]
_lib.pa_threaded_mainloop_lock.restype = None
_lib.pa_threaded_mainloop_lock.argtypes = [ctypes.c_void_p]
_lib.pa_threaded_mainloop_unlock.restype = None
_lib.pa_threaded_mainloop_unlock.argtypes = [ctypes.c_void_p]
_lib.pa_threaded_mainloop_wait.restype = None
_lib.pa_threaded_mainloop_wait.argtypes = [ctypes.c_void_p]
_lib.pa_threaded_mainloop_signal.restype = None
_lib.pa_threaded_mainloop_signal.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.pa_threaded_mainloop_free.restype = None
_lib.pa_threaded_mainloop_free.argtypes = [ctypes.c_void_p]

# context
_lib.pa_context_new.restype = ctypes.c_void_p
_lib.pa_context_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.pa_context_connect.restype = ctypes.c_int
_lib.pa_context_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
_lib.pa_context_get_state.restype = ctypes.c_int
_lib.pa_context_get_state.argtypes = [ctypes.c_void_p]
_lib.pa_context_set_state_callback.restype = None
_lib.pa_context_set_state_callback.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_lib.pa_context_unref.restype = None
_lib.pa_context_unref.argtypes = [ctypes.c_void_p]

# format info
_lib.pa_format_info_new.restype = ctypes.c_void_p
_lib.pa_format_info_free.restype = None
_lib.pa_format_info_free.argtypes = [ctypes.c_void_p]
_lib.pa_format_info_set_prop_int.restype = None
_lib.pa_format_info_set_prop_int.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]

# stream
_lib.pa_stream_new_extended.restype = ctypes.c_void_p
_lib.pa_stream_new_extended.argtypes = [
    ctypes.c_void_p,   # context
    ctypes.c_char_p,   # name
    ctypes.POINTER(ctypes.c_void_p),  # formats array
    ctypes.c_uint,     # n_formats
    ctypes.c_void_p,   # proplist
]
_lib.pa_stream_connect_playback.restype = ctypes.c_int
_lib.pa_stream_connect_playback.argtypes = [
    ctypes.c_void_p,   # stream
    ctypes.c_char_p,   # dev
    ctypes.POINTER(pa_buffer_attr),  # attr
    ctypes.c_int,      # flags
    ctypes.c_void_p,   # volume
    ctypes.c_void_p,   # sync_stream
]
_lib.pa_stream_get_state.restype = ctypes.c_int
_lib.pa_stream_get_state.argtypes = [ctypes.c_void_p]
_lib.pa_stream_set_state_callback.restype = None
_lib.pa_stream_set_state_callback.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_lib.pa_stream_write.restype = ctypes.c_int
_lib.pa_stream_write.argtypes = [
    ctypes.c_void_p,   # stream
    ctypes.c_void_p,   # data
    ctypes.c_size_t,   # nbytes
    ctypes.c_void_p,   # free_cb
    ctypes.c_int64,    # offset
    ctypes.c_int,      # seek (PA_SEEK_RELATIVE = 0)
]
_lib.pa_stream_drain.restype = ctypes.c_void_p
_lib.pa_stream_drain.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_lib.pa_stream_disconnect.restype = ctypes.c_int
_lib.pa_stream_disconnect.argtypes = [ctypes.c_void_p]
_lib.pa_stream_unref.restype = None
_lib.pa_stream_unref.argtypes = [ctypes.c_void_p]
_lib.pa_stream_writable_size.restype = ctypes.c_size_t
_lib.pa_stream_writable_size.argtypes = [ctypes.c_void_p]

_lib.pa_strerror.restype = ctypes.c_char_p
_lib.pa_strerror.argtypes = [ctypes.c_int]

# PA_FORMAT_INFO props
PA_PROP_FORMAT_RATE = b"format.rate"
PA_PROP_FORMAT_CHANNELS = b"format.channels"

# ---------------------------------------------------------------------------
# Callback types
# ---------------------------------------------------------------------------

_STATE_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)
_DRAIN_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)


# ---------------------------------------------------------------------------
# PAPassthroughStream
# ---------------------------------------------------------------------------


class PAPassthroughStream:
    """
    Blocking passthrough stream for IEC 61937 encoded audio.

    All public methods are blocking and intended to be called from an
    executor thread (not the asyncio event loop).
    """

    def __init__(
        self,
        sink_name: str,
        encoding: str = "ac3",
        sample_rate: int = 48000,
        buffer_msec: int = 200,
        app_name: str = "music_assistant_spdif",
    ) -> None:
        self._sink_name = sink_name.encode()
        self._encoding = ENCODING_MAP.get(encoding, PA_ENCODING_AC3_IEC61937)
        self._sample_rate = sample_rate
        self._buffer_msec = buffer_msec
        self._app_name = app_name.encode()

        self._mainloop: ctypes.c_void_p | None = None
        self._context: ctypes.c_void_p | None = None
        self._stream: ctypes.c_void_p | None = None
        self._fmt_info: ctypes.c_void_p | None = None
        self._opened = False
        self._error: str | None = None

        # Keep callback references alive to prevent GC
        self._ctx_cb_ref: _STATE_CB_TYPE | None = None
        self._stream_cb_ref: _STATE_CB_TYPE | None = None

    # ------------------------------------------------------------------

    def open(self, timeout: float = 10.0) -> None:
        """Connect to PA and wait until stream is ready."""
        pulse_server = os.environ.get("PULSE_SERVER", "").encode() or None

        self._mainloop = _lib.pa_threaded_mainloop_new()
        if not self._mainloop:
            raise OSError("pa_threaded_mainloop_new failed")

        api = _lib.pa_threaded_mainloop_get_api(self._mainloop)
        self._context = _lib.pa_context_new(api, self._app_name)
        if not self._context:
            raise OSError("pa_context_new failed")

        # Context state callback
        def _ctx_state_cb(ctx: ctypes.c_void_p, userdata: ctypes.c_void_p) -> None:
            state = _lib.pa_context_get_state(ctx)
            if state in (PA_CONTEXT_READY, PA_CONTEXT_FAILED, PA_CONTEXT_TERMINATED):
                _lib.pa_threaded_mainloop_signal(self._mainloop, 0)

        self._ctx_cb_ref = _STATE_CB_TYPE(_ctx_state_cb)
        _lib.pa_context_set_state_callback(self._context, self._ctx_cb_ref, None)

        _lib.pa_threaded_mainloop_lock(self._mainloop)
        _lib.pa_threaded_mainloop_start(self._mainloop)

        _lib.pa_context_connect(self._context, pulse_server, 0, None)

        # Wait for context ready
        deadline = time.monotonic() + timeout
        while True:
            state = _lib.pa_context_get_state(self._context)
            if state == PA_CONTEXT_READY:
                break
            if state in (PA_CONTEXT_FAILED, PA_CONTEXT_TERMINATED):
                _lib.pa_threaded_mainloop_unlock(self._mainloop)
                raise OSError(f"PA context failed (state={state})")
            if time.monotonic() > deadline:
                _lib.pa_threaded_mainloop_unlock(self._mainloop)
                raise TimeoutError("Timed out waiting for PA context")
            _lib.pa_threaded_mainloop_wait(self._mainloop)

        # Build format info
        self._fmt_info = _lib.pa_format_info_new()
        # Set encoding type directly via struct offset 0 (pa_encoding_t is first field)
        ctypes.cast(self._fmt_info, ctypes.POINTER(ctypes.c_int))[0] = self._encoding
        _lib.pa_format_info_set_prop_int(self._fmt_info, PA_PROP_FORMAT_RATE, self._sample_rate)
        _lib.pa_format_info_set_prop_int(self._fmt_info, PA_PROP_FORMAT_CHANNELS, 2)

        formats_array = (ctypes.c_void_p * 1)(self._fmt_info)

        self._stream = _lib.pa_stream_new_extended(
            self._context,
            b"spdif_passthrough",
            formats_array,
            1,
            None,
        )
        if not self._stream:
            _lib.pa_threaded_mainloop_unlock(self._mainloop)
            raise OSError("pa_stream_new_extended failed")

        # Stream state callback
        def _stream_state_cb(stream: ctypes.c_void_p, userdata: ctypes.c_void_p) -> None:
            state = _lib.pa_stream_get_state(stream)
            if state in (PA_STREAM_READY, PA_STREAM_FAILED, PA_STREAM_TERMINATED):
                _lib.pa_threaded_mainloop_signal(self._mainloop, 0)

        self._stream_cb_ref = _STATE_CB_TYPE(_stream_state_cb)
        _lib.pa_stream_set_state_callback(self._stream, self._stream_cb_ref, None)

        # Buffer attrs — use large buffer to absorb jitter
        bytes_per_ms = (self._sample_rate // 1000) * 2 * 2  # 2ch * 2 bytes (s16le framing)
        tlength = self._buffer_msec * bytes_per_ms
        attr = pa_buffer_attr(
            maxlength=ctypes.c_uint32(-1).value,
            tlength=tlength,
            prebuf=ctypes.c_uint32(-1).value,
            minreq=ctypes.c_uint32(-1).value,
            fragsize=ctypes.c_uint32(-1).value,
        )

        flags = PA_STREAM_ADJUST_LATENCY | PA_STREAM_AUTO_TIMING_UPDATE
        ret = _lib.pa_stream_connect_playback(
            self._stream,
            self._sink_name,
            ctypes.byref(attr),
            flags,
            None,
            None,
        )
        if ret < 0:
            _lib.pa_threaded_mainloop_unlock(self._mainloop)
            raise OSError(f"pa_stream_connect_playback failed: {ret}")

        # Wait for stream ready
        deadline = time.monotonic() + timeout
        while True:
            state = _lib.pa_stream_get_state(self._stream)
            if state == PA_STREAM_READY:
                break
            if state in (PA_STREAM_FAILED, PA_STREAM_TERMINATED):
                _lib.pa_threaded_mainloop_unlock(self._mainloop)
                raise OSError(f"PA stream failed (state={state})")
            if time.monotonic() > deadline:
                _lib.pa_threaded_mainloop_unlock(self._mainloop)
                raise TimeoutError("Timed out waiting for PA stream")
            _lib.pa_threaded_mainloop_wait(self._mainloop)

        _lib.pa_threaded_mainloop_unlock(self._mainloop)
        self._opened = True

    def write(self, data: bytes) -> None:
        """Write IEC 61937 encoded bytes to the stream."""
        if not self._opened or not self._stream:
            raise OSError("Stream not open")
        _lib.pa_threaded_mainloop_lock(self._mainloop)
        try:
            ret = _lib.pa_stream_write(
                self._stream,
                ctypes.c_char_p(data),
                ctypes.c_size_t(len(data)),
                None,   # no free callback
                0,      # offset
                0,      # PA_SEEK_RELATIVE
            )
            if ret < 0:
                raise OSError(f"pa_stream_write failed: {ret}")
        finally:
            _lib.pa_threaded_mainloop_unlock(self._mainloop)

    def drain(self) -> None:
        """Drain remaining buffered data."""
        if not self._opened or not self._stream:
            return
        # Simple timed wait — drain op callback complexity not worth it here
        time.sleep(self._buffer_msec / 1000.0 + 0.1)

    def close(self) -> None:
        """Disconnect stream and free PA resources."""
        self._opened = False
        if self._stream:
            _lib.pa_threaded_mainloop_lock(self._mainloop)
            _lib.pa_stream_disconnect(self._stream)
            _lib.pa_stream_unref(self._stream)
            _lib.pa_threaded_mainloop_unlock(self._mainloop)
            self._stream = None
        if self._fmt_info:
            _lib.pa_format_info_free(self._fmt_info)
            self._fmt_info = None
        if self._context:
            _lib.pa_context_unref(self._context)
            self._context = None
        if self._mainloop:
            _lib.pa_threaded_mainloop_stop(self._mainloop)
            _lib.pa_threaded_mainloop_free(self._mainloop)
            self._mainloop = None

"""S/PDIF Audio Out — player implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING


from music_assistant.helpers.ffmpeg import FFMpeg

from .constants import (
    CONF_ENCODING_FORMAT,
    CONF_PA_SINK_NAME,
    ENCODING_AC3,
    ENCODING_BITRATES,
    ENCODING_DTS,
    ENCODING_EAC3,
    ENCODING_MAX_CHANNELS,
)
from .pa_simple import (
    PA_SAMPLE_S16LE,
    pa_simple_drain,
    pa_simple_free,
    pa_simple_new,
    pa_simple_write,
    pa_strerror,
)

if TYPE_CHECKING:
    from music_assistant_models.player_queue import PlayerQueue

LOGGER = logging.getLogger(__name__)

# IEC 61937 bitstreams are framed into a 2ch / 16-bit / 48 kHz stereo container
# regardless of the encoded format — this is what PA's iec958 sink expects.
_SPDIF_CHANNELS = 2
_SPDIF_SAMPLE_RATE = 48000
_SPDIF_BIT_DEPTH = 16

# PA write buffer in ms — absorbs flow-stream chunk delivery jitter (~75 ms
# for 6ch content).  Keep ≥ 2× the expected delivery period.
_PA_BUFFER_MSEC = 160

# ffmpeg output chunk size in bytes fed to PA per loop iteration.
# 10 ms of 2ch s16le @ 48 kHz = 48000 * 2ch * 2bytes * 0.010 s = 1920 bytes
_CHUNK_BYTES = 1920


def _ffmpeg_encode_args(
    encoding: str,
    source_channels: int,
    source_sample_rate: int,
) -> list[str]:
    """
    Build the ffmpeg output-side arguments for IEC 61937 encoding.

    The input side is handled by FFMpeg / MA's standard helper; we only
    supply the output codec + muxer arguments here.

    IEC 61937 always delivers a 2ch / 48 kHz / s16le bitstream at the
    physical layer, so we force those output parameters.

    *source_channels* is capped to the format's max channel count so ffmpeg
    doesn't silently upmix a stereo source to 5.1 (which would waste bits and
    produce incorrect surround).
    """
    bitrate = ENCODING_BITRATES[encoding]
    max_ch = ENCODING_MAX_CHANNELS[encoding]
    out_channels = min(source_channels, max_ch)

    # Channel layout string understood by ffmpeg
    ch_layout_map = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}
    ch_layout = ch_layout_map.get(out_channels, f"{out_channels}c")

    # Codec name for ffmpeg -c:a
    codec_map = {
        ENCODING_AC3: "ac3",
        ENCODING_DTS: "dts",
        ENCODING_EAC3: "eac3",
    }
    codec = codec_map[encoding]

    # DTS 96/24 profile (preserves 96 kHz / 24-bit source fidelity in the
    # encoded bitstream; still framed as IEC 61937 2ch container)
    extra: list[str] = []
    if encoding == ENCODING_DTS and source_sample_rate >= 96000:
        extra = ["-profile:a", "3"]  # DTS 96/24

    args = [
        # --- input channel selection (no silent upmix) ---
        "-ac",
        str(out_channels),
        "-channel_layout",
        ch_layout,
        # --- encode ---
        "-c:a",
        codec,
        "-b:a",
        str(bitrate),
        *extra,
        # --- IEC 61937 mux into stereo PCM container ---
        "-f",
        "spdif",
        # Force output to the fixed IEC 61937 physical parameters
        "-ar",
        str(_SPDIF_SAMPLE_RATE),
        # spdif muxer outputs s16le interleaved into 2ch by default;
        # make it explicit to avoid surprises.
        "-sample_fmt",
        "s16",
    ]
    return args


class SPDIFPlayer:
    """
    Handles playback for a single S/PDIF output sink.

    Flow:
        MA flow stream URL
            → FFMpeg (encode to AC3/DTS/E-AC3 wrapped in IEC 61937 spdif muxer)
            → raw s16le 2ch 48 kHz byte stream
            → pa_simple write loop
            → PA iec958 sink
            → S/PDIF optical / HDMI → receiver / soundbar
    """

    def __init__(self, provider: "SPDIFAudioProvider") -> None:  # noqa: F821
        self._provider = provider
        self._playback_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public interface (called by SPDIFAudioProvider)
    # ------------------------------------------------------------------

    async def play_media(self, player_id: str, queue: PlayerQueue) -> None:
        """Start playing the current queue item."""
        await self.stop()

        sink_name: str = self._provider.config.get_value(CONF_PA_SINK_NAME)
        encoding: str = self._provider.config.get_value(CONF_ENCODING_FORMAT)

        # Resolve the MA flow stream URL for the current queue item
        url = await self._provider.mass.streams.resolve_stream_url(
            queue_item=queue.current_item,
            player_id=player_id,
            content_type=None,  # let MA choose PCM
        )

        # Determine source channel count from streamdetails
        source_channels = 2  # safe default
        source_sample_rate = 48000
        try:
            sd = queue.current_item.streamdetails
            if sd and sd.audio_format:
                source_channels = sd.audio_format.channels or 2
                source_sample_rate = sd.audio_format.sample_rate or 48000
        except AttributeError:
            pass

        LOGGER.debug(
            "play_media: sink=%s encoding=%s source_ch=%d source_sr=%d url=%s",
            sink_name,
            encoding,
            source_channels,
            source_sample_rate,
            url,
        )

        self._stop_event.clear()
        self._playback_task = asyncio.create_task(
            self._playback_loop(
                url=url,
                sink_name=sink_name,
                encoding=encoding,
                source_channels=source_channels,
                source_sample_rate=source_sample_rate,
            )
        )

    async def stop(self) -> None:
        """Stop playback."""
        self._stop_event.set()
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
        self._playback_task = None

    async def pause(self) -> None:
        """Pause — stop the PA write loop; MA will restart on resume."""
        await self.stop()

    # ------------------------------------------------------------------
    # Playback loop
    # ------------------------------------------------------------------

    async def _playback_loop(
        self,
        url: str,
        sink_name: str,
        encoding: str,
        source_channels: int,
        source_sample_rate: int,
    ) -> None:
        """
        Main playback coroutine.

        Opens a PA simple stream to *sink_name*, runs ffmpeg to encode the
        MA flow stream, and writes the IEC 61937 bitstream chunks to PA.
        """
        loop = asyncio.get_running_loop()
        pa_stream = None

        try:
            # --- Open PA stream (2ch / s16le / 48 kHz) ---
            pa_stream, err = await loop.run_in_executor(
                None,
                lambda: pa_simple_new(
                    server=None,
                    app_name="music_assistant_spdif",
                    sink_name=sink_name,
                    stream_name="spdif_out",
                    sample_format=PA_SAMPLE_S16LE,
                    sample_rate=_SPDIF_SAMPLE_RATE,
                    channels=_SPDIF_CHANNELS,
                    buffer_msec=_PA_BUFFER_MSEC,
                ),
            )
            if pa_stream is None:
                LOGGER.error(
                    "Failed to open PA stream to sink '%s': %s",
                    sink_name,
                    pa_strerror(err),
                )
                return

            # --- Build ffmpeg encode args ---
            extra_output_args = _ffmpeg_encode_args(
                encoding=encoding,
                source_channels=source_channels,
                source_sample_rate=source_sample_rate,
            )

            # MA's FFMpeg helper opens the flow stream URL and re-encodes.
            # We request PCM input → let ffmpeg do codec + spdif mux.
            # iter_chunked gives us small fixed-size chunks (10 ms target)
            # which keeps PA write latency predictable.
            ffmpeg = FFMpeg(
                audio_input=url,
                input_format=None,  # let ffmpeg probe
                output_format="spdif",
                extra_output_args=extra_output_args,
            )

            async for chunk in ffmpeg.iter_chunked(_CHUNK_BYTES):
                if self._stop_event.is_set():
                    break
                if not chunk:
                    continue

                write_err = await loop.run_in_executor(
                    None,
                    lambda c=chunk: pa_simple_write(pa_stream, c),
                )
                if write_err is not None:
                    LOGGER.error(
                        "PA write error on sink '%s': %s",
                        sink_name,
                        pa_strerror(write_err),
                    )
                    break

        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            LOGGER.exception("Unexpected error in S/PDIF playback loop")
        finally:
            if pa_stream is not None:
                # Drain remaining buffered audio before closing
                try:
                    await loop.run_in_executor(None, lambda: pa_simple_drain(pa_stream))
                except Exception:  # noqa: BLE001
                    pass
                await loop.run_in_executor(None, lambda: pa_simple_free(pa_stream))

            LOGGER.debug("S/PDIF playback loop exited for sink '%s'", sink_name)

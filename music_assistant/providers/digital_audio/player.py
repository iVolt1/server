"""S/PDIF Audio Out — Player implementation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from music_assistant_models.enums import PlayerFeature, PlaybackState
from music_assistant_models.player import DeviceInfo

from music_assistant.models.player import Player, PlayerMedia

from .constants import (
    CONF_ENCODING_FORMAT,
    ENCODING_BITRATES,
    ENCODING_DTS,
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
    from .provider import SPDIFAudioProvider

_SPDIF_CHANNELS = 2
_SPDIF_SAMPLE_RATE = 48000
_PA_BUFFER_MSEC = 160
_CHUNK_BYTES = 1920


def _ffmpeg_encode_args(
    encoding: str, source_channels: int, source_sample_rate: int
) -> list[str]:
    """Build ffmpeg output args for IEC 61937 encoding."""
    bitrate = ENCODING_BITRATES[encoding]
    max_ch = ENCODING_MAX_CHANNELS[encoding]
    out_channels = min(source_channels, max_ch)
    ch_layout_map = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}
    ch_layout = ch_layout_map.get(out_channels, f"{out_channels}c")
    codec_map = {"ac3": "ac3", "dts": "dts", "eac3": "eac3"}
    codec = codec_map[encoding]
    extra: list[str] = []
    if encoding == ENCODING_DTS and source_sample_rate >= 96000:
        extra = ["-profile:a", "3"]
    return [
        "-ac",
        str(out_channels),
        "-channel_layout",
        ch_layout,
        "-c:a",
        codec,
        "-b:a",
        str(bitrate),
        *extra,
        "-f",
        "spdif",
        "-ar",
        str(_SPDIF_SAMPLE_RATE),
        "-sample_fmt",
        "s16",
    ]


class SPDIFPlayer(Player):
    """S/PDIF output player — encodes to IEC 61937 and writes to PA iec958 sink."""

    def __init__(
        self, provider: SPDIFAudioProvider, player_id: str, sink_name: str
    ) -> None:
        """Initialize the S/PDIF player."""
        self._attr_name = f"S/PDIF {sink_name}"
        self._sink_name = sink_name
        self._playback_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        # super().__init__ resets _attr_supported_features and _attr_device_info
        # to empty defaults — set those AFTER calling super.
        super().__init__(provider, player_id)
        self._attr_available = True
        self._attr_device_info = DeviceInfo(
            model="S/PDIF IEC 61937", manufacturer="PulseAudio"
        )
        self._attr_supported_features = {
            PlayerFeature.POWER,
            PlayerFeature.VOLUME_SET,
            PlayerFeature.PAUSE,
            PlayerFeature.PLAY_MEDIA,
        }
        self._attr_playback_state = PlaybackState.IDLE

    async def power(self, powered: bool) -> None:
        """Power on/off."""
        self._attr_powered = powered
        if not powered:
            await self.stop()

    async def volume_set(self, volume_level: int) -> None:
        """Set volume via pactl."""
        pa_vol = int(volume_level / 100 * 65536)
        try:
            proc = await asyncio.create_subprocess_exec(
                "pactl",
                "set-sink-volume",
                self._sink_name,
                str(pa_vol),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=3)
            self._attr_volume_level = volume_level
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("pactl set-sink-volume failed: %s", exc)

    async def play(self) -> None:
        """Resume — MA calls play_media for actual content."""

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
        self._attr_playback_state = PlaybackState.IDLE

    async def pause(self) -> None:
        """Pause."""
        await self.stop()
        self._attr_playback_state = PlaybackState.PAUSED

    async def play_media(self, media: PlayerMedia) -> None:
        """Start playing media."""
        await self.stop()
        encoding: str = self._provider.config.get_value(CONF_ENCODING_FORMAT)
        source_channels = 2
        source_sample_rate = 48000
        try:
            if media.streamdetails and media.streamdetails.audio_format:
                source_channels = media.streamdetails.audio_format.channels or 2
                source_sample_rate = (
                    media.streamdetails.audio_format.sample_rate or 48000
                )
        except AttributeError:
            pass
        self.logger.debug(
            "play_media: sink=%s encoding=%s ch=%d sr=%d uri=%s",
            self._sink_name,
            encoding,
            source_channels,
            source_sample_rate,
            media.uri,
        )
        url = await self._provider.mass.streams.resolve_stream_url(
            self.player_id, media
        )
        self.logger.debug("Resolved stream URL: %s", url)
        self._stop_event.clear()
        self._attr_playback_state = PlaybackState.PLAYING
        self._attr_current_media = media
        self.update_state()
        self._playback_task = self.mass.create_task(
            self._playback_loop(url, encoding, source_channels, source_sample_rate)
        )

    async def _playback_loop(
        self,
        url: str,
        encoding: str,
        source_channels: int,
        source_sample_rate: int,
    ) -> None:
        """Encode MA flow stream to IEC 61937 and write to PA sink."""

        loop = asyncio.get_running_loop()
        pa_stream = None
        ffmpeg_proc: asyncio.subprocess.Process | None = None
        try:
            pa_stream, err = await loop.run_in_executor(
                None,
                lambda: pa_simple_new(
                    server=None,
                    app_name="music_assistant_spdif",
                    sink_name=self._sink_name,
                    stream_name="spdif_out",
                    sample_format=PA_SAMPLE_S16LE,
                    sample_rate=_SPDIF_SAMPLE_RATE,
                    channels=_SPDIF_CHANNELS,
                    buffer_msec=_PA_BUFFER_MSEC,
                ),
            )
            if pa_stream is None:
                self.logger.error(
                    "Failed to open PA stream to '%s': %s",
                    self._sink_name,
                    pa_strerror(err),
                )
                self._attr_playback_state = PlaybackState.IDLE
                return

            extra_output_args = _ffmpeg_encode_args(
                encoding, source_channels, source_sample_rate
            )
            # Build ffmpeg command directly — FFMpeg helper doesn't support spdif output.
            ffmpeg_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-reconnect",
                "1",
                "-reconnect_delay_max",
                "10",
                "-reconnect_streamed",
                "1",
                "-i",
                url,
                *extra_output_args,
                "-",  # write IEC 61937 bitstream to stdout
            ]
            self.logger.debug("ffmpeg cmd: %s", " ".join(ffmpeg_cmd))
            ffmpeg_proc = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Drain stderr in background so it doesn't block
            async def _log_stderr() -> None:
                assert ffmpeg_proc.stderr is not None
                async for line in ffmpeg_proc.stderr:
                    self.logger.debug(
                        "ffmpeg: %s", line.decode(errors="replace").rstrip()
                    )

            stderr_task = asyncio.create_task(_log_stderr())
            try:
                assert ffmpeg_proc.stdout is not None
                while True:
                    if self._stop_event.is_set():
                        break
                    chunk = await ffmpeg_proc.stdout.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    write_err = await loop.run_in_executor(
                        None, lambda c=chunk: pa_simple_write(pa_stream, c)
                    )
                    if write_err is not None:
                        self.logger.error(
                            "PA write error on '%s': %s",
                            self._sink_name,
                            pa_strerror(write_err),
                        )
                        break
            finally:
                stderr_task.cancel()
                if ffmpeg_proc.returncode is None:
                    ffmpeg_proc.kill()
                    await ffmpeg_proc.wait()
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            self.logger.exception("Unexpected error in S/PDIF playback loop")
        finally:
            if pa_stream is not None:
                try:
                    await loop.run_in_executor(None, lambda: pa_simple_drain(pa_stream))
                except Exception:  # noqa: BLE001
                    pass
                await loop.run_in_executor(None, lambda: pa_simple_free(pa_stream))
            self._attr_playback_state = PlaybackState.IDLE
            self.logger.debug("S/PDIF playback loop exited for '%s'", self._sink_name)

"""Multichannel Audio Player implementation."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING

import numpy as np
from music_assistant_models.enums import (
    ContentType,
    IdentifierType,
    PlayerFeature,
    PlayerType,
    PlaybackState,
)
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.player import DeviceInfo

from music_assistant.helpers.ffmpeg import get_ffmpeg_stream
from music_assistant.models.player import Player, PlayerMedia

from .constants import (
    CACHE_CATEGORY_PREV_STATE,
    CONF_VOLUME_CONTROL,
    DEFAULT_HARDWARE_VOLUME_CEILING,
    DEFAULT_PLAYER_VOLUME,
    DEVICE_UUID_NAMESPACE,
    VOLUME_CONTROL_HARDWARE,
    VOLUME_CONTROL_SOFTWARE,
)

try:
    import pulsectl
    _PULSECTL_AVAILABLE = True
except ImportError:
    _PULSECTL_AVAILABLE = False

if TYPE_CHECKING:
    from .provider import MultiChannelAudioProvider


# Channel pair definitions for demuxing interleaved multichannel PCM.
# Each entry maps a PA sink name suffix to the channel indices it carries.
# FLAC channel ordering per spec:
# 5.1: FL=0, FR=1, FC=2, LFE=3, RL=4, RR=5
# 7.1: FL=0, FR=1, FC=2, LFE=3, RL=4, RR=5, SL=6, SR=7
_PAIR_CHANNEL_INDICES_71 = {
    "front_stereo":  (0, 1),   # FL, FR
    "rear_stereo":   (4, 5),   # RL, RR
    "center_sub":    (2, 3),   # FC, LFE
    "side_stereo":   (6, 7),   # SL, SR
}
_PAIR_CHANNEL_INDICES_51 = {
    "front_stereo":  (0, 1),   # FL, FR
    "rear_stereo":   (4, 5),   # RL, RR
    "center_sub":    (2, 3),   # FC, LFE
}


def get_player_uuid(pa_sink_name: str) -> str:
    """
    Generate a stable UUID for a multichannel player from its PA sink name.

    :param pa_sink_name: The PulseAudio sink name.
    """
    return str(uuid.uuid5(DEVICE_UUID_NAMESPACE, pa_sink_name))


def _build_pair_sinks(card_name: str, layout: str) -> dict[str, tuple[int, int]]:
    """
    Build mapping of PA sink name -> (ch_index_left, ch_index_right) for a card.

    :param card_name: Card name prefix used in remap sink names e.g. 'Creative_X_Fi'.
    :param layout: Layout string '5.1' or '7.1'.
    """
    pairs = _PAIR_CHANNEL_INDICES_71 if layout == "7.1" else _PAIR_CHANNEL_INDICES_51
    return {f"{card_name}_{pair}": indices for pair, indices in pairs.items()}


class MultiChannelPlayer(Player):
    """
    Player for a multichannel surround output via stereo PA remap sink pairs.

    Receives an 8ch (7.1) or 6ch (5.1) PCM stream from MA, demuxes it into
    stereo pairs, and writes each pair to its dedicated PA remap sink
    simultaneously. All remap sinks share the same underlying ALSA hardware
    clock so the pairs stay in sync.
    """

    def __init__(
        self,
        provider: MultiChannelAudioProvider,
        player_id: str,
        card_name: str,
        display_name: str,
        channels: int,
        layout: str,
        sample_rate: int,
        bit_depth: int,
    ) -> None:
        """
        Initialize the Multichannel Audio player.

        :param provider: The Multichannel Audio provider instance.
        :param player_id: Stable player ID.
        :param card_name: Card name prefix used in PA remap sink names.
        :param display_name: Human-readable name shown in the MA UI.
        :param channels: Number of source channels (6 for 5.1, 8 for 7.1).
        :param layout: Layout identifier '5.1' or '7.1'.
        :param sample_rate: Native sample rate of the PA sinks.
        :param bit_depth: Bit depth (16, 24, or 32).
        """
        super().__init__(provider, player_id)
        self._attr_type = PlayerType.PLAYER
        self._attr_name = display_name
        self._attr_available = True
        self._attr_supported_features = {
            PlayerFeature.PLAY_MEDIA,
            PlayerFeature.VOLUME_SET,
            PlayerFeature.VOLUME_MUTE,
            PlayerFeature.PAUSE,
        }
        self._attr_device_info = DeviceInfo(
            model=display_name,
            manufacturer="Multichannel Audio",
        )
        self._attr_device_info.add_identifier(IdentifierType.UUID, player_id)
        self._attr_can_group_with = set()
        self._attr_volume_level = DEFAULT_PLAYER_VOLUME

        self.card_name = card_name
        self.channels = channels
        self.layout = layout
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth

        # Map of PA sink name -> (left_ch_index, right_ch_index)
        self._pair_sinks: dict[str, tuple[int, int]] = _build_pair_sinks(card_name, layout)

        self._hardware_volume_fallback = False
        self._playback_task: asyncio.Task[None] | None = None
        self._paused = False

    @property
    def needs_poll(self) -> bool:
        """Return if the player needs to be polled for state updates."""
        return False

    @property
    def volume_control_mode(self) -> str:
        """Return the effective volume control mode for this player."""
        if self._hardware_volume_fallback:
            return VOLUME_CONTROL_SOFTWARE
        return str(
            self._provider.config.get_value(CONF_VOLUME_CONTROL) or VOLUME_CONTROL_HARDWARE
        )

    # --- MA mandatory player interface ---

    async def play_media(self, media: PlayerMedia) -> None:
        """Handle PLAY_MEDIA command."""
        await self._stop_playback()
        url = await self._provider.mass.streams.resolve_stream_url(self.player_id, media)
        self.logger.info("Starting multichannel playback from %s", url)
        # Get source channel count from active queue streamdetails
        source_channels = self.channels
        try:
            queue = self.mass.player_queues.get_active_queue(self.player_id)
            if queue and queue.current_item and queue.current_item.streamdetails:
                sd = queue.current_item.streamdetails
                self.logger.debug(
                    "streamdetails: channels=%d sample_rate=%d uri=%s",
                    sd.audio_format.channels,
                    sd.audio_format.sample_rate,
                    sd.uri,
                )
                if sd.audio_format.channels > 0:
                    source_channels = sd.audio_format.channels
        except Exception as err:
            self.logger.debug("Could not read streamdetails: %s", err)
        self._attr_current_media = media
        self._attr_playback_state = PlaybackState.PLAYING
        self._paused = False
        self.update_state()
        self._playback_task = self.mass.create_task(
            self._playback_loop(url, source_channels)
        )

    @property
    def _source_channels(self) -> int:
        """Return stored source channel count, defaulting to player channels."""
        return getattr(self, "_stored_source_channels", self.channels)

    @_source_channels.setter
    def _source_channels(self, value: int) -> None:
        self._stored_source_channels = value

    async def stop(self) -> None:
        """Handle STOP command."""
        await self._stop_playback()
        self._attr_playback_state = PlaybackState.IDLE
        self._attr_current_media = None
        self.update_state()

    async def pause(self) -> None:
        """Handle PAUSE command."""
        self._paused = True
        self._attr_playback_state = PlaybackState.PAUSED
        self.update_state()

    async def play(self) -> None:
        """Handle PLAY/resume command."""
        self._paused = False
        self._attr_playback_state = PlaybackState.PLAYING
        self.update_state()

    # --- Playback loop ---

    async def _playback_loop(self, url: str, source_channels: int = 0) -> None:
        """
        Fetch the MA PCM stream and demux it to stereo PA sink pairs.

        Opens one PASimpleStream per stereo pair, then for each PCM chunk
        extracts the two relevant channels and writes them to the
        corresponding sink. All writes happen in the same executor thread
        to keep the pairs as tightly coupled as possible.
        """
        from .pa_simple import PASimpleStream  # noqa: PLC0415

        if source_channels == 0:
            source_channels = self.channels

        output_format = AudioFormat(
            content_type=ContentType.from_bit_depth(self.bit_depth),
            sample_rate=self.sample_rate,
            bit_depth=self.bit_depth,
            channels=source_channels,
        )
        self.logger.debug(
            "Requesting output format: %dch %dHz %dbit %s (source=%d player=%d)",
            output_format.channels,
            output_format.sample_rate,
            output_format.bit_depth,
            output_format.content_type,
            source_channels,
            self.channels,
        )
        streams: dict[str, PASimpleStream] = {}
        try:
            # Open a PA stream for each stereo pair
            for sink_name in self._pair_sinks:
                sname = sink_name
                stream = await self.mass.loop.run_in_executor(
                    None,
                    lambda s=sname: PASimpleStream(
                        sink_name=s,
                        app_name="music-assistant-multichannel",
                        rate=self.sample_rate,
                        channels=2,
                        bit_depth=self.bit_depth,
                    ),
                )
                streams[sink_name] = stream
                self.logger.debug("Opened PA stream for %s", sink_name)
            self.logger.info(
                "Multichannel playback started: %d pairs, %dch, %dHz, %dbit",
                len(streams),
                self.channels,
                self.sample_rate,
                self.bit_depth,
            )
            self.logger.debug(
                "pair_sinks=%s streams=%s",
                list(self._pair_sinks.keys()),
                list(streams.keys()),
            )

            first_chunk = True
            actual_channels = source_channels
            async for chunk in get_ffmpeg_stream(
                audio_input=url,
                input_format=AudioFormat(content_type=ContentType.UNKNOWN),
                output_format=output_format,
            ):
                if first_chunk:
                    # Detect actual channel count from chunk size.
                    # bytes_per_sample = bit_depth // 8 (use 4 for 24-bit containers)
                    bps = 4 if self.bit_depth >= 24 else 2
                    samples_total = len(chunk) // bps
                    # Try each possible channel count to find what divides evenly
                    for candidate in (6, 8, 2, 4, 1):
                        if samples_total % candidate == 0:
                            actual_channels = candidate
                            break
                    self.logger.debug(
                        "First PCM chunk: len=%d detected_channels=%d requested=%d content_type=%s",
                        len(chunk),
                        actual_channels,
                        source_channels,
                        output_format.content_type,
                    )
                    # Log RMS energy per detected channel
                    ct_val = str(output_format.content_type.value).lower()
                    is_float_check = "f32" in ct_val or "float" in ct_val
                    dtype_check = np.float32 if is_float_check else (
                        np.int16 if self.bit_depth == 16 else np.int32
                    )
                    s = np.frombuffer(chunk, dtype=dtype_check)
                    nf = len(s) // actual_channels
                    if nf > 0:
                        s = s[:nf * actual_channels].reshape(nf, actual_channels)
                        for ch in range(actual_channels):
                            rms = float(np.sqrt(np.mean(s[:, ch].astype(np.float64) ** 2)))
                            self.logger.debug("  ch[%d] RMS=%.1f", ch, rms)
                    first_chunk = False

                if self._paused:
                    await asyncio.sleep(0.05)
                    continue

                chunk = self._apply_software_volume(chunk)
                # MA uses F32 internally when processing is applied (normalization, DSP etc.)
                # Detect by checking if the content type string contains 'f32' or 'float'
                ct_val = str(output_format.content_type.value).lower()
                is_float = "f32" in ct_val or "float" in ct_val
                await self.mass.loop.run_in_executor(
                    None, self._write_demuxed, chunk, streams, is_float, actual_channels
                )

        except asyncio.CancelledError:
            pass
        except Exception as err:
            self.logger.error("Playback error: %s", err)
        finally:
            for sink_name, stream in streams.items():
                with suppress(Exception):
                    await self.mass.loop.run_in_executor(None, stream.close)
                self.logger.debug("Closed PA stream for %s", sink_name)
            self._attr_playback_state = PlaybackState.IDLE
            self._attr_current_media = None
            self.update_state()
            if self._playback_task is asyncio.current_task():
                self._playback_task = None

    def _write_demuxed(
        self,
        pcm_data: bytes,
        streams: dict[str, PASimpleStream],
        is_float: bool = False,
        source_channels: int = 0,
    ) -> None:
        """
        Demux interleaved multichannel PCM and write each stereo pair to its sink.

        Called in an executor thread. Uses source_channels for reshape so that
        a 6ch source is not misinterpreted as 8ch.

        :param pcm_data: Interleaved multichannel PCM bytes.
        :param streams: Map of sink name to open PASimpleStream.
        :param is_float: True if pcm_data is float32.
        :param source_channels: Actual channel count in pcm_data (0 = use self.channels).
        """
        channels = source_channels if source_channels > 0 else self.channels
        if is_float:
            samples_f = np.frombuffer(pcm_data, dtype=np.float32)
            num_frames = len(samples_f) // channels
            if num_frames == 0:
                return
            samples_f = samples_f[: num_frames * channels].reshape(num_frames, channels)
            for sink_name, (left_idx, right_idx) in self._pair_sinks.items():
                if sink_name not in streams or left_idx >= channels or right_idx >= channels:
                    continue
                pair_f = np.column_stack((samples_f[:, left_idx], samples_f[:, right_idx]))
                pair_i32 = np.clip(pair_f * 2147483647.0, -2147483648, 2147483647).astype(np.int32)
                streams[sink_name].write(pair_i32.tobytes())
        else:
            dtype = np.int16 if self.bit_depth == 16 else np.int32
            samples = np.frombuffer(pcm_data, dtype=dtype)
            num_frames = len(samples) // channels
            if num_frames == 0:
                return
            samples = samples[: num_frames * channels].reshape(num_frames, channels)
            for sink_name, (left_idx, right_idx) in self._pair_sinks.items():
                if sink_name not in streams or left_idx >= channels or right_idx >= channels:
                    import logging  # noqa: PLC0415
                    logging.getLogger("music_assistant.Multichannel Audio Out").debug(
                        "_write_demuxed: skipping %s (idx %d,%d >= %dch)",
                        sink_name, left_idx, right_idx, channels,
                    )
                    continue
                pair = np.column_stack((samples[:, left_idx], samples[:, right_idx]))
                pair_bytes = pair.astype(dtype).tobytes()
                if self.bit_depth == 24:
                    pair_bytes = pair.view(np.uint8).reshape(-1, 4)[:, 1:].tobytes()
                streams[sink_name].write(pair_bytes)

    async def _stop_playback(self) -> None:
        """Cancel and await the playback task if running."""
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._playback_task
        self._playback_task = None

    async def stop_stream(self) -> None:
        """Stop streaming — alias for stop() used by provider unload."""
        await self._stop_playback()

    # --- Volume control ---

    async def volume_set(self, volume_level: int) -> None:
        """Handle VOLUME_SET command."""
        self._attr_volume_level = volume_level
        if self.volume_control_mode == VOLUME_CONTROL_HARDWARE:
            # Set volume on all stereo pair sinks
            for sink_name in self._pair_sinks:
                loop = asyncio.get_running_loop()
                ok = await loop.run_in_executor(
                    None, self._set_pulse_volume, sink_name, volume_level
                )
                if not ok:
                    self._hardware_volume_fallback = True
                    break
        await self._save_state()
        self.update_state()

    async def volume_mute(self, muted: bool) -> None:
        """Handle VOLUME_MUTE command."""
        self._attr_volume_muted = muted
        if self.volume_control_mode == VOLUME_CONTROL_HARDWARE:
            for sink_name in self._pair_sinks:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, self._set_pulse_mute, sink_name, muted
                )
        await self._save_state()
        self.update_state()

    async def apply_hardware_ceiling(self) -> None:
        """Set PA sink hardware volume ceiling on all stereo pair sinks."""
        for sink_name in self._pair_sinks:
            loop = asyncio.get_running_loop()
            ok = await loop.run_in_executor(
                None, self._set_pulse_volume, sink_name, DEFAULT_HARDWARE_VOLUME_CEILING
            )
            if ok:
                self.logger.debug(
                    "Hardware ceiling set to %d%% for sink %s",
                    DEFAULT_HARDWARE_VOLUME_CEILING,
                    sink_name,
                )

    def _set_pulse_volume(self, pa_sink_name: str, volume: int) -> bool:
        """
        Set PulseAudio sink volume via pulsectl. Returns True on success.

        :param pa_sink_name: The PulseAudio sink name.
        :param volume: Volume level 0-100.
        """
        if not _PULSECTL_AVAILABLE:
            return False
        try:
            with pulsectl.Pulse("ma-multichannel") as pulse:
                for sink in pulse.sink_list():
                    if sink.name == pa_sink_name:
                        pulse.volume_set_all_chans(sink, volume / 100.0)
                        return True
            return False
        except Exception as err:
            self.logger.warning("pulsectl volume error for %s: %s", pa_sink_name, err)
            return False

    def _set_pulse_mute(self, pa_sink_name: str, muted: bool) -> bool:
        """
        Set PulseAudio sink mute state via pulsectl. Returns True on success.

        :param pa_sink_name: The PulseAudio sink name.
        :param muted: Whether to mute or unmute.
        """
        if not _PULSECTL_AVAILABLE:
            return False
        try:
            with pulsectl.Pulse("ma-multichannel") as pulse:
                for sink in pulse.sink_list():
                    if sink.name == pa_sink_name:
                        pulse.mute(sink, muted)
                        return True
            return False
        except Exception as err:
            self.logger.warning("pulsectl mute error for %s: %s", pa_sink_name, err)
            return False

    def _apply_software_volume(self, pcm_data: bytes) -> bytes:
        """Apply software volume scaling to PCM data."""
        if self.volume_control_mode != VOLUME_CONTROL_SOFTWARE:
            return pcm_data
        if self._attr_volume_muted:
            return b"\x00" * len(pcm_data)
        volume = self._attr_volume_level
        if volume is None or volume >= 100:
            return pcm_data
        scale = volume / 100.0
        if self.bit_depth == 32:
            samples = np.frombuffer(pcm_data, dtype=np.int32).copy()
            scaled = np.clip(samples.astype(np.float64) * scale, -2147483648, 2147483647)
            return scaled.astype(np.int32).tobytes()
        if self.bit_depth == 24:
            samples = np.frombuffer(pcm_data, dtype=np.int32).copy()
            scaled = np.clip(
                samples.astype(np.float64) * scale, -2147483648, 2147483647
            ).astype(np.int32)
            return scaled.view(np.uint8).reshape(-1, 4)[:, 1:].tobytes()
        samples_16 = np.frombuffer(pcm_data, dtype=np.int16).copy()
        scaled = np.clip(samples_16.astype(np.float64) * scale, -32768, 32767)
        return scaled.astype(np.int16).tobytes()

    # --- State persistence ---

    async def restore_state(self) -> None:
        """Restore cached volume/mute state from a previous session."""
        if last_state := await self.mass.cache.get(
            key=self.player_id,
            provider=self._provider.instance_id,
            category=CACHE_CATEGORY_PREV_STATE,
        ):
            self._attr_volume_muted = last_state[0]
            self._attr_volume_level = last_state[1]
        else:
            self._attr_volume_muted = False
            self._attr_volume_level = DEFAULT_PLAYER_VOLUME

    async def _save_state(self) -> None:
        """Persist current volume/mute state to cache."""
        await self.mass.cache.set(
            key=self.player_id,
            data=[self._attr_volume_muted, self._attr_volume_level],
            provider=self._provider.instance_id,
            category=CACHE_CATEGORY_PREV_STATE,
        )


def _probe_stream_channels(url: str) -> int:
    """
    Probe the channel count of an audio stream URL using ffprobe.

    Called in an executor thread. Returns 0 on failure.

    :param url: The audio stream URL to probe.
    """
    import json  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "a:0",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams:
                return int(streams[0].get("channels", 0))
    except Exception:
        pass
    return 0

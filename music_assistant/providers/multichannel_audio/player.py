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


def get_player_uuid(pa_sink_name: str) -> str:
    """
    Generate a stable UUID for a multichannel player from its PA sink name.

    :param pa_sink_name: The PulseAudio sink name.
    """
    return str(uuid.uuid5(DEVICE_UUID_NAMESPACE, pa_sink_name))


class MultiChannelPlayer(Player):
    """Player for a multichannel PulseAudio surround sink."""

    def __init__(
        self,
        provider: MultiChannelAudioProvider,
        player_id: str,
        sink_name: str,
        display_name: str,
        channels: int,
        layout: str,
        channel_map: str,
        sample_rate: int,
        bit_depth: int,
        is_remap: bool = False,
    ) -> None:
        """
        Initialize the Multichannel Audio player.

        :param provider: The Multichannel Audio provider instance.
        :param player_id: Stable player ID derived from sink UUID.
        :param sink_name: PulseAudio sink name to stream to.
        :param display_name: Human-readable name shown in the MA UI.
        :param channels: Number of output channels (6 for 5.1, 8 for 7.1).
        :param layout: Layout identifier e.g. "5.1" or "7.1".
        :param channel_map: Comma-separated PA channel map string.
        :param sample_rate: Native sample rate of the PA sink.
        :param bit_depth: Bit depth of the PA sink (16, 24, or 32).
        :param is_remap: True if this is a remap/filter sink.
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

        self.sink_name = sink_name
        self.channels = channels
        self.layout = layout
        self.channel_map = channel_map
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self._is_remap = is_remap
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
        """Handle PLAY_MEDIA command — resolve stream URL and start PCM playback."""
        await self._stop_playback()
        url = await self._provider.mass.streams.resolve_stream_url(self.player_id, media)
        self.logger.info("Starting multichannel playback from %s", url)
        self._attr_current_media = media
        self._attr_playback_state = PlaybackState.PLAYING
        self._paused = False
        self.update_state()
        self._playback_task = self.mass.create_task(self._playback_loop(url))

    async def stop(self) -> None:
        """Handle STOP command."""
        await self._stop_playback()
        self._attr_playback_state = PlaybackState.IDLE
        self._attr_current_media = None
        self.update_state()

    async def pause(self) -> None:
        """Handle PAUSE command — stop writing to PA but keep task alive."""
        self._paused = True
        self._attr_playback_state = PlaybackState.PAUSED
        self.update_state()

    async def play(self) -> None:
        """Handle PLAY/resume command."""
        self._paused = False
        self._attr_playback_state = PlaybackState.PLAYING
        self.update_state()

    # --- Playback loop ---

    async def _playback_loop(self, url: str) -> None:
        """Fetch the MA stream URL and write PCM chunks to the PA sink."""
        from .pa_simple import PASimpleStream  # noqa: PLC0415

        output_format = AudioFormat(
            content_type=ContentType.from_bit_depth(self.bit_depth),
            sample_rate=self.sample_rate,
            bit_depth=self.bit_depth,
            channels=self.channels,
        )
        stream: PASimpleStream | None = None
        try:
            sink_name = self.sink_name
            stream = await self.mass.loop.run_in_executor(
                None,
                lambda: PASimpleStream(
                    sink_name=sink_name,
                    app_name="music-assistant-multichannel",
                    rate=self.sample_rate,
                    channels=self.channels,
                    bit_depth=self.bit_depth,
                ),
            )
            self.logger.debug(
                "PA multichannel stream opened: sink=%s rate=%d channels=%d bit_depth=%d",
                self.sink_name,
                self.sample_rate,
                self.channels,
                self.bit_depth,
            )

            chunk_count = 0
            async for chunk in get_ffmpeg_stream(
                audio_input=url,
                input_format=AudioFormat(content_type=ContentType.UNKNOWN),
                output_format=output_format,
            ):
                chunk_count += 1
                if chunk_count == 1:
                    self.logger.debug(
                        "First PCM chunk received len=%d format=%s",
                        len(chunk),
                        output_format,
                    )
                if self._paused:
                    # drain chunks silently while paused to avoid buffer backup
                    await asyncio.sleep(0.1)
                    continue
                chunk = self._apply_software_volume(chunk)
                await self.mass.loop.run_in_executor(None, stream.write, chunk)

        except asyncio.CancelledError:
            pass
        except Exception as err:
            self.logger.error("Playback error for %s: %s", self.sink_name, err)
        finally:
            if stream is not None:
                with suppress(Exception):
                    await self.mass.loop.run_in_executor(None, stream.close)
            self._attr_playback_state = PlaybackState.IDLE
            self._attr_current_media = None
            self.update_state()
            if self._playback_task is asyncio.current_task():
                self._playback_task = None

    async def _stop_playback(self) -> None:
        """Cancel and await the playback task if running."""
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._playback_task
        self._playback_task = None

    # --- Volume control ---

    async def volume_set(self, volume_level: int) -> None:
        """Handle VOLUME_SET command."""
        self._attr_volume_level = volume_level
        if self.volume_control_mode == VOLUME_CONTROL_HARDWARE:
            loop = asyncio.get_running_loop()
            ok = await loop.run_in_executor(
                None, self._set_pulse_volume, self.sink_name, volume_level
            )
            if not ok:
                self.logger.warning(
                    "PulseAudio volume control failed for %s, falling back to software",
                    self.sink_name,
                )
                self._hardware_volume_fallback = True
        await self._save_state()
        self.update_state()

    async def volume_mute(self, muted: bool) -> None:
        """Handle VOLUME_MUTE command."""
        self._attr_volume_muted = muted
        if self.volume_control_mode == VOLUME_CONTROL_HARDWARE:
            loop = asyncio.get_running_loop()
            ok = await loop.run_in_executor(
                None, self._set_pulse_mute, self.sink_name, muted
            )
            if not ok:
                self._hardware_volume_fallback = True
        await self._save_state()
        self.update_state()

    async def apply_hardware_ceiling(self) -> None:
        """Set PA sink hardware volume ceiling (Linux only)."""
        target = 100 if self._is_remap else DEFAULT_HARDWARE_VOLUME_CEILING
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(
            None, self._set_pulse_volume, self.sink_name, target
        )
        if ok:
            self.logger.debug(
                "Hardware ceiling set to %d%% for sink %s", target, self.sink_name
            )
        else:
            self.logger.warning(
                "Failed to set hardware ceiling for sink %s", self.sink_name
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
            self.logger.warning("PA sink %s not found for volume control", pa_sink_name)
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
            self.logger.warning("PA sink %s not found for mute", pa_sink_name)
            return False
        except Exception as err:
            self.logger.warning("pulsectl mute error for %s: %s", pa_sink_name, err)
            return False

    def _apply_software_volume(self, pcm_data: bytes) -> bytes:
        """Apply software volume scaling to interleaved multichannel PCM."""
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

"""Multichannel Audio Player implementation."""

from __future__ import annotations

import asyncio
import threading
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
    CHANNEL_MAP_CUSTOM,
    CHANNEL_MAP_FLAC,
    DEFAULT_HARDWARE_VOLUME_CEILING,
    DEFAULT_PLAYER_VOLUME,
    DEVICE_UUID_NAMESPACE,
    PAIR_INDICES_BY_MAP,
    VOLUME_CONTROL_SOFTWARE,
)

if TYPE_CHECKING:
    from .provider import MultiChannelAudioProvider



def get_player_uuid(pa_sink_name: str) -> str:
    """
    Generate a stable UUID for a multichannel player from its PA sink name.

    :param pa_sink_name: The PulseAudio sink name.
    """
    return str(uuid.uuid5(DEVICE_UUID_NAMESPACE, pa_sink_name))


def _parse_custom_map(
    card_name: str, layout: str, custom_str: str
) -> dict[str, tuple[int, int]] | None:
    """
    Parse a custom channel map string into a pair-sinks dict.

    Expected format: flat comma-separated indices, two per sink pair, in order:
    front_stereo, center_sub, rear_stereo[, side_stereo].
    e.g. "0,1,2,3,4,5" for FLAC 5.1  or  "0,1,3,2,4,5" for DVD 5.1.

    Returns None if the string is empty or malformed.
    """
    if not custom_str.strip():
        return None
    try:
        indices = [int(x.strip()) for x in custom_str.split(",")]
    except ValueError:
        return None
    suffixes_51 = ["front_stereo", "center_sub", "rear_stereo"]
    suffixes_71 = ["front_stereo", "center_sub", "rear_stereo", "side_stereo"]
    suffixes = suffixes_71 if layout == "7.1" else suffixes_51
    expected = len(suffixes) * 2
    if len(indices) < expected:
        return None
    return {
        f"{card_name}_{suffix}": (indices[i * 2], indices[i * 2 + 1])
        for i, suffix in enumerate(suffixes)
    }


def _build_pair_sinks(
    card_name: str,
    layout: str,
    channel_map: str = CHANNEL_MAP_FLAC,
    custom_channel_map: str = "",
) -> dict[str, tuple[int, int]]:
    """
    Build mapping of PA sink name -> (ch_index_left, ch_index_right).

    Resolves the channel index map from the configured preset or custom string.

    :param card_name: Card name prefix used in remap sink names e.g. 'Creative_X_Fi'.
    :param layout: Layout string '5.1' or '7.1'.
    :param channel_map: Channel map preset identifier ('flac', 'dvd', 'custom').
    :param custom_channel_map: Custom flat index string, used when channel_map='custom'.
    """
    if channel_map == CHANNEL_MAP_CUSTOM:
        parsed = _parse_custom_map(card_name, layout, custom_channel_map)
        if parsed is not None:
            return parsed
        # Fall through to FLAC default if custom string is missing/invalid

    pairs = PAIR_INDICES_BY_MAP.get(channel_map, {}).get(
        layout, PAIR_INDICES_BY_MAP[CHANNEL_MAP_FLAC][layout]
    )
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
        channel_map: str = CHANNEL_MAP_FLAC,
        custom_channel_map: str = "",
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
        :param channel_map: Channel map preset ('flac', 'dvd', 'custom').
        :param custom_channel_map: Custom flat index string when channel_map='custom'.
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
        self._pair_sinks: dict[str, tuple[int, int]] = _build_pair_sinks(
            card_name, layout, channel_map, custom_channel_map
        )

        self._playback_task: asyncio.Task[None] | None = None
        self._paused = False

    @property
    def needs_poll(self) -> bool:
        """Return if the player needs to be polled for state updates."""
        return False

    @property
    def volume_control_mode(self) -> str:
        """Return the volume control mode. Always software (numpy PCM scaling)."""
        return VOLUME_CONTROL_SOFTWARE

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

        Sequential architecture: for each ffmpeg chunk, demux into per-sink
        buffers then write all sinks in a single executor thread, one sink at
        a time. PA buffers are sized to absorb the full chunk so pa_simple_write
        returns immediately without blocking for drain.
        """
        from .pa_simple import PASimpleStream  # noqa: PLC0415
        from music_assistant.helpers.ffmpeg import FFMpeg  # noqa: PLC0415

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

        # PA buffer sized to 2× the expected ffmpeg burst (~640ms at 8ch/96kHz).
        # This ensures pa_simple_write returns immediately without blocking for
        # drain, keeping all sink writes synchronous and in phase.
        buffer_msec = 1500

        # Request chunks sized to ~640ms — matches flow stream burst size.
        chunk_size = int(self.sample_rate * 0.640) * source_channels * 4

        streams: dict[str, PASimpleStream] = {}
        ffmpeg_proc: FFMpeg | None = None
        try:
            for sink_name, (left_idx, right_idx) in self._pair_sinks.items():
                if left_idx >= source_channels or right_idx >= source_channels:
                    continue
                sname = sink_name
                stream = await self.mass.loop.run_in_executor(
                    None,
                    lambda s=sname: PASimpleStream(
                        sink_name=s,
                        app_name="music-assistant-multichannel",
                        rate=self.sample_rate,
                        channels=2,
                        bit_depth=self.bit_depth,
                        buffer_msec=buffer_msec,
                    ),
                )
                streams[sink_name] = stream
                self.logger.debug("Opened PA stream for %s", sink_name)

            self.logger.info(
                "Multichannel playback started: %d active pairs, %dch source, %dHz, %dbit",
                len(streams),
                source_channels,
                self.sample_rate,
                self.bit_depth,
            )

            ffmpeg_proc = FFMpeg(
                audio_input=url,
                input_format=AudioFormat(content_type=ContentType.UNKNOWN),
                output_format=output_format,
                extra_output_args=["-flush_packets", "1"],
                collect_log_history=True,
            )
            await ffmpeg_proc.start()

            first_chunk = True
            ct_val: str = ""
            is_float = False
            async for chunk in ffmpeg_proc.iter_chunked(chunk_size):
                if first_chunk:
                    ct_val = str(output_format.content_type.value).lower()
                    is_float = "f32" in ct_val or "float" in ct_val
                    self.logger.debug(
                        "First PCM chunk: len=%d channels=%d content_type=%s",
                        len(chunk), source_channels, output_format.content_type,
                    )
                    first_chunk = False

                if self._paused:
                    await asyncio.sleep(0.05)
                    continue

                chunk = self._apply_software_volume(chunk)
                await self.mass.loop.run_in_executor(
                    None, self._demux_and_write_all, chunk, streams, is_float, source_channels
                )

        except asyncio.CancelledError:
            pass
        except Exception as err:
            self.logger.error("Playback error: %s", err)
        finally:
            if ffmpeg_proc is not None:
                with suppress(Exception):
                    await ffmpeg_proc.close()
            for sink_name, stream in streams.items():
                with suppress(Exception):
                    await self.mass.loop.run_in_executor(None, stream.close)
                self.logger.debug("Closed PA stream for %s", sink_name)
            self._attr_playback_state = PlaybackState.IDLE
            self._attr_current_media = None
            self.update_state()
            if self._playback_task is asyncio.current_task():
                self._playback_task = None

    def _demux_and_write_all(
        self,
        pcm_data: bytes,
        streams: dict[str, PASimpleStream],
        is_float: bool,
        source_channels: int,
    ) -> None:
        """Demux interleaved PCM and write each pair to its PA sink sequentially.

        PA buffers are sized to absorb the full chunk so each write returns
        immediately. Sequential writes keep all sinks frame-aligned.

        :param pcm_data: Interleaved multichannel PCM bytes (s32le or f32).
        :param streams: Map of sink_name -> open PASimpleStream.
        :param is_float: True if pcm_data is float32.
        :param source_channels: Channel count in pcm_data.
        """
        channels = source_channels if source_channels > 0 else self.channels
        if is_float:
            samples = np.frombuffer(pcm_data, dtype=np.float32)
            num_frames = len(samples) // channels
            if num_frames == 0:
                return
            samples = samples[: num_frames * channels].reshape(num_frames, channels)
            for sink_name, (left_idx, right_idx) in self._pair_sinks.items():
                if sink_name not in streams:
                    continue
                if left_idx >= channels or right_idx >= channels:
                    continue
                pair = np.column_stack((samples[:, left_idx], samples[:, right_idx]))
                streams[sink_name].write(
                    np.clip(pair * 2147483647.0, -2147483648, 2147483647)
                    .astype(np.int32)
                    .tobytes()
                )
        else:
            dtype = np.int16 if self.bit_depth == 16 else np.int32
            samples = np.frombuffer(pcm_data, dtype=dtype)
            num_frames = len(samples) // channels
            if num_frames == 0:
                return
            samples = samples[: num_frames * channels].reshape(num_frames, channels)
            for sink_name, (left_idx, right_idx) in self._pair_sinks.items():
                if sink_name not in streams:
                    continue
                if left_idx >= channels or right_idx >= channels:
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
        """Handle VOLUME_SET command. Volume applied via software PCM scaling."""
        self._attr_volume_level = volume_level
        await self._save_state()
        self.update_state()

    async def volume_mute(self, muted: bool) -> None:
        """Handle VOLUME_MUTE command. Mute applied via software PCM scaling."""
        self._attr_volume_muted = muted
        await self._save_state()
        self.update_state()

    async def apply_restored_volume(self) -> None:
        """Clamp restored volume to DEFAULT_HARDWARE_VOLUME_CEILING on startup.

        Ensures a corrupt or missing cache entry never causes full-blast output.
        Volume is applied to the PCM stream via software scaling during playback.
        """
        volume = min(
            self._attr_volume_level or DEFAULT_PLAYER_VOLUME,
            DEFAULT_HARDWARE_VOLUME_CEILING,
        )
        self._attr_volume_level = volume
        self.logger.debug(
            "Restored volume %d%% muted=%s",
            volume,
            self._attr_volume_muted,
        )

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
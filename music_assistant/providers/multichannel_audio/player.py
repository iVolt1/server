"""Multichannel Audio Player implementation."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
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
    StreamType,
)
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.player import DeviceInfo

from music_assistant.helpers.ffmpeg import get_ffmpeg_stream
from music_assistant.models.player import PlayerMedia
from music_assistant.models.player_protocol import PlayerProtocol

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
    from .pa_simple import PASimpleStream
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


class MultiChannelPlayer(PlayerProtocol):
    """
    Player for a multichannel surround output via stereo PA remap sink pairs.

    Receives an 8ch (7.1) or 6ch (5.1) PCM stream from MA, demuxes it into
    stereo pairs, and writes each pair to its dedicated PA remap sink
    simultaneously. All remap sinks share the same underlying ALSA hardware
    clock so the pairs stay in sync.
    """

    def __init__(  # noqa: PLR0913
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
        pair_sink_names: dict[str, str] | None = None,
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
        :param pair_sink_names: Explicit PA sink per pair name (e.g. 'front_stereo'),
            used instead of the '<card_name>_<pair>' remap sink names.
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
        if pair_sink_names:
            self._pair_sinks = {
                pair_sink_names.get(key.removeprefix(f"{card_name}_"), key): indices
                for key, indices in self._pair_sinks.items()
            }

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

    @property
    def supported_sample_rates(self) -> list[tuple[int, int]]:
        """
        Declare the only sample rate this player actually supports.

        The PA remap sinks are fixed at self.sample_rate/self.bit_depth — they
        do not resample. Declaring this (rather than falling back to the
        generic CONF_SAMPLE_RATES list, which usually includes 44100/48000)
        forces select_flow_pcm_format's 'smart'/'bit_perfect' anchoring to
        always snap up to self.sample_rate instead of passing a lower source
        rate straight through, which previously caused sped-up ("chipmunk")
        playback for 48kHz sources.
        """
        return [(self.sample_rate, self.bit_depth)]

    @property
    def requires_flow_mode(self) -> bool:
        """Return False: play_media must see each queue item to open its source file."""
        return False

    # --- MA mandatory player interface ---

    async def play_media(self, media: PlayerMedia) -> None:
        """Handle PLAY_MEDIA command."""
        await self._stop_playback()
        source_channels = 2
        file_path: str | None = None
        seek_position = 0.0
        try:
            queue = (
                self.mass.player_queues.get_active_queue(self.player.player_id)
                if self.player
                else None
            )
            item = None
            if queue and media.queue_item_id:
                item = self.mass.player_queues.get_item(queue.queue_id, media.queue_item_id)
            sd = item.streamdetails if item else None
            self.logger.debug(
                "play_media: queue_item_id=%s item_found=%s sd_present=%s",
                media.queue_item_id,
                item is not None,
                sd is not None,
            )
            if sd:
                self.logger.debug(
                    "streamdetails: stream_type=%s path=%s channels=%d sample_rate=%d",
                    sd.stream_type,
                    sd.path,
                    sd.audio_format.channels,
                    sd.audio_format.sample_rate,
                )
                # MA folds anything wider than stereo down to 2 channels before it reaches
                # a player, so a multichannel local file is read straight from disk instead
                if (
                    sd.stream_type == StreamType.LOCAL_FILE
                    and isinstance(sd.path, str)
                    and sd.audio_format.channels > 2
                ):
                    file_path = sd.path
                    source_channels = sd.audio_format.channels
                    seek_position = float(sd.seek_position or 0)
        except Exception as err:
            self.logger.debug("Could not read streamdetails: %s", err)
        if file_path:
            url = file_path
            self.logger.info(
                "Starting multichannel playback from file %s (%dch, seek %.1fs)",
                file_path,
                source_channels,
                seek_position,
            )
        else:
            url = await self.mass.streams.resolve_stream_url(self.protocol_id, media)
            self.logger.info("Starting multichannel playback from %s", url)
        self._attr_current_media = media
        self._attr_playback_state = PlaybackState.PLAYING
        self._paused = False
        self.update_state()
        self._playback_task = self.mass.create_task(
            self._playback_loop(url, source_channels, seek_position)
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
        if self._attr_elapsed_time is not None and self._attr_elapsed_time_last_updated:
            now = time.time()
            self._attr_elapsed_time += now - self._attr_elapsed_time_last_updated
            self._attr_elapsed_time_last_updated = now
        self._attr_playback_state = PlaybackState.PAUSED
        self.update_state()

    async def play(self) -> None:
        """Handle PLAY/resume command."""
        self._paused = False
        if self._attr_elapsed_time is not None:
            self._attr_elapsed_time_last_updated = time.time()
        self._attr_playback_state = PlaybackState.PLAYING
        self.update_state()

    # --- Playback loop ---

    async def _playback_loop(
        self, url: str, source_channels: int = 0, seek_position: float = 0.0
    ) -> None:
        """
        Fetch the MA PCM stream and demux it to stereo PA sink pairs.

        MA delivers s32le PCM at self.sample_rate with source_channels channels
        via the patched select_flow_pcm_format. Sequential demux writes to all
        PA remap sinks in a single executor thread keeping them frame-aligned.
        PA buffers are sized to absorb full ffmpeg burst chunks without blocking.
        """
        from .pa_simple import PASimpleStream  # noqa: PLC0415
        from music_assistant.helpers.ffmpeg import FFMpeg  # noqa: PLC0415

        if source_channels == 0:
            source_channels = self.channels

        # MA delivers s32le at hardware sample rate
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

        # Small chunks (10ms) with moderate PA buffer (300ms).
        # Large chunks cause pa_simple_write to block on the first sink
        # while the buffer fills, starving subsequent sinks of data.
        chunk_size = int(self.sample_rate * 0.010) * source_channels * 4
        chunk_size = max((chunk_size // (source_channels * 4)) * (source_channels * 4),
                         source_channels * 4)
        buffer_msec = 300

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
            # elapsed is reported relative to the start of this stream; the queue adds
            # the item's seek_position itself
            self._attr_elapsed_time = 0.0
            self._attr_elapsed_time_last_updated = time.time()
            self.update_state()

            # resolve_stream_url's CONF_OUTPUT_CODEC resolution is non-deterministic
            # across player (re)registration/restart cycles — sometimes it resolves
            # to our intended explicit PCM string (sometimes even with the wrong
            # rate/channels params), and sometimes it falls all the way back to
            # plain "flac". We can't assume either outcome; we have to look at what
            # the URL actually says was served. Forcing ContentType.UNKNOWN here
            # makes ffmpeg auto-probe undeclared multichannel PCM and reject it
            # outright ("Invalid data found", returncode=183); forcing PCM
            # unconditionally (as before) makes ffmpeg silently misread genuine
            # FLAC-compressed bytes as raw PCM samples, producing hissy static
            # instead of music. So: parse the real extension/codec MA put in the
            # URL and only force explicit PCM params when MA is actually serving
            # PCM; otherwise let ffmpeg auto-probe (it correctly identifies FLAC,
            # AAC, MP3, etc. from the byte stream on its own).
            url_fmt_str = url.rsplit(".", 1)[-1] if "." in url.rsplit("/", 1)[-1] else ""
            url_content_type = ContentType.try_parse(url_fmt_str)
            if url_content_type.is_pcm():
                # MA is serving raw PCM — declare the real format explicitly so
                # ffmpeg doesn't have to (and can't reliably) probe it from
                # undeclared multichannel bytes.
                actual_input_format = AudioFormat(
                    content_type=ContentType.from_bit_depth(self.bit_depth),
                    sample_rate=self.sample_rate,
                    bit_depth=self.bit_depth,
                    channels=source_channels,
                )
            else:
                # MA is serving a compressed/encoded format (flac, aac, mp3, ...) —
                # let ffmpeg auto-probe and decode it normally, same as MA's own
                # internal pipeline does for the source file itself.
                actual_input_format = AudioFormat(content_type=ContentType.UNKNOWN)
            self.logger.debug(
                "Resolved stream format from URL: url_content_type=%s -> input_format=%s",
                url_content_type, actual_input_format.content_type,
            )
            ffmpeg_proc = FFMpeg(
                audio_input=url,
                input_format=actual_input_format,
                output_format=output_format,
                extra_input_args=["-ss", f"{seek_position:.3f}"] if seek_position else None,
                extra_output_args=["-flush_packets", "1"],
                collect_log_history=True,
            )
            await ffmpeg_proc.start()
            self.logger.debug(
                "ffmpeg started: pid=%s chunk_size=%d",
                ffmpeg_proc.proc.pid if ffmpeg_proc.proc else None, chunk_size,
            )

            first_chunk = True
            ct_val: str = ""
            is_float = False
            first_demux_logged = False
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
                if not first_demux_logged:
                    first_demux_logged = True
                    self.logger.debug(
                        "First demux: pair_sinks=%s streams=%s source_channels=%d",
                        list(self._pair_sinks.keys()),
                        list(streams.keys()),
                        source_channels,
                    )
                await self.mass.loop.run_in_executor(
                    None, self._demux_and_write_all, chunk, streams, is_float, source_channels
                )

        except asyncio.CancelledError:
            pass
        except Exception as err:
            self.logger.error("Playback error: %s", err, exc_info=True)
        finally:
            if ffmpeg_proc is not None:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "ffmpeg exit: returncode=%s closed=%s log_history=%s",
                        ffmpeg_proc.returncode,
                        ffmpeg_proc.closed,
                        list(ffmpeg_proc.log_history),
                    )
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
        source_bit_depth: int = 32,
    ) -> None:
        """Demux interleaved PCM and write each pair to its PA sink sequentially.

        PA buffers are sized to absorb the full chunk so each write returns
        immediately. Sequential writes keep all sinks frame-aligned.

        :param pcm_data: Interleaved multichannel PCM bytes (s32le, s16le, or f32).
        :param streams: Map of sink_name -> open PASimpleStream.
        :param is_float: True if pcm_data is float32.
        :param source_channels: Channel count in pcm_data.
        :param source_bit_depth: Bit depth of pcm_data (16, 24, or 32).
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
            dtype = np.int16 if source_bit_depth == 16 else np.int32
            samples = np.frombuffer(pcm_data, dtype=dtype)
            num_frames = len(samples) // channels
            if num_frames == 0:
                return
            samples = samples[: num_frames * channels].reshape(num_frames, channels)
            # Always log on first chunk — local var, not instance state
            debug_first = True
            for sink_name, (left_idx, right_idx) in self._pair_sinks.items():
                if sink_name not in streams:
                    continue
                if left_idx >= channels or right_idx >= channels:
                    continue
                pair = np.column_stack((samples[:, left_idx], samples[:, right_idx]))
                if debug_first:
                    self.logger.debug(
                        "demux %s: idx=(%d,%d) frames=%d max_abs=%d",
                        sink_name, left_idx, right_idx,
                        len(pair), int(np.abs(pair).max()),
                    )
                if source_bit_depth == 16:
                    streams[sink_name].write((pair.astype(np.int32) << 16).tobytes())
                elif source_bit_depth == 24:
                    pair_bytes = pair.view(np.uint8).reshape(-1, 4)[:, 1:].tobytes()
                    streams[sink_name].write(pair_bytes)
                else:
                    streams[sink_name].write(pair.tobytes())

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
        """Apply software volume scaling to s32le PCM data."""
        if self.volume_control_mode != VOLUME_CONTROL_SOFTWARE:
            return pcm_data
        if self._attr_volume_muted:
            return b"\x00" * len(pcm_data)
        volume = self._attr_volume_level
        if volume is None or volume >= 100:
            return pcm_data
        scale = volume / 100.0
        samples = np.frombuffer(pcm_data, dtype=np.int32).copy()
        scaled = np.clip(samples.astype(np.float64) * scale, -2147483648, 2147483647)
        return scaled.astype(np.int32).tobytes()

    # --- State persistence ---

    async def restore_state(self) -> None:
        """Restore cached volume/mute state from a previous session."""
        if last_state := await self.mass.cache.get(
            key=self.protocol_id,
            provider=self.provider_id,
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
            key=self.protocol_id,
            data=[self._attr_volume_muted, self._attr_volume_level],
            provider=self.provider_id,
            category=CACHE_CATEGORY_PREV_STATE,
        )

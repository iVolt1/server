"""Digital Audio Player — direct PulseAudio multichannel output via ffmpeg."""

from __future__ import annotations

import asyncio
import logging
import shutil
import signal
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING

from music_assistant_models.enums import (
    ContentType,
    IdentifierType,
    PlayerFeature,
    PlayerType,
    PlaybackState,
)
from music_assistant_models.player import DeviceInfo

from music_assistant.models.player import Player, PlayerMedia

from .constants import (
    CACHE_CATEGORY_PREV_STATE,
    DEFAULT_HARDWARE_VOLUME_CEILING,
    DEFAULT_PLAYER_VOLUME,
    DEVICE_UUID_NAMESPACE,
)

if TYPE_CHECKING:
    from .provider import DigitalAudioProvider


def get_player_uuid(pa_sink_name: str) -> str:
    """
    Generate a stable UUID for a digital audio player from its PA sink name.

    :param pa_sink_name: The PulseAudio sink name.
    """
    return str(uuid.uuid5(DEVICE_UUID_NAMESPACE, pa_sink_name))


class DigitalAudioPlayer(Player):
    """
    Player for direct multichannel output to a PulseAudio sink via ffmpeg -f pulse.

    Uses ``ffmpeg -f pulse <sink_name>`` to write an N-channel PCM stream
    directly to the named hardware sink, bypassing the HA supervisor's
    2-channel ``pa_simple`` proxy restriction entirely.  ffmpeg connects as a
    full PulseAudio client, negotiating the sink's native multichannel format.

    Supports any PA sink: analog surround (5.1/7.1), HDMI audio, USB
    multichannel.  No remap sinks, no libpulse-simple, no numpy demux.

    Volume control is via ``pactl set-sink-volume`` on the hardware sink.
    Pause is implemented via SIGSTOP/SIGCONT so ffmpeg freezes in place
    without tearing down the PA stream connection.
    """

    def __init__(
        self,
        provider: DigitalAudioProvider,
        player_id: str,
        sink_name: str,
        display_name: str,
        channels: int,
        sample_rate: int,
        bit_depth: int,
    ) -> None:
        """
        Initialize the Digital Audio player.

        :param provider: The Digital Audio provider instance.
        :param player_id: Stable player ID (UUID5 derived from sink name).
        :param sink_name: PulseAudio sink name
            (e.g. 'alsa_output.pci-0000_07_00.6.analog-surround-71').
        :param display_name: Human-readable name shown in the MA UI.
        :param channels: Native channel count of the PA sink (6 for 5.1, 8 for 7.1).
        :param sample_rate: Native sample rate of the PA sink (Hz).
        :param bit_depth: Bit depth of the PA sink (16, 24, or 32).
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
            manufacturer="Digital Audio",
        )
        self._attr_device_info.add_identifier(IdentifierType.UUID, player_id)
        self._attr_can_group_with = set()
        self._attr_volume_level = DEFAULT_PLAYER_VOLUME

        self._sink_name = sink_name
        self.channels = channels
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth

        self._playback_task: asyncio.Task[None] | None = None
        self._paused = False

    @property
    def needs_poll(self) -> bool:
        """Return if the player needs to be polled for state updates."""
        return False

    @property
    def supported_sample_rates(self) -> list[tuple[int, int]]:
        """
        Declare only the hardware sample rate this player supports.

        The PA sink is fixed at self.sample_rate/self.bit_depth and does not
        resample.  Declaring this (rather than falling back to the generic
        CONF_SAMPLE_RATES list) forces select_flow_pcm_format's 'smart'/'bit_perfect'
        anchoring to snap up to self.sample_rate instead of passing a lower
        source rate straight through — which would cause sped-up ("chipmunk")
        playback for 48kHz sources on a 96kHz sink.
        """
        return [(self.sample_rate, self.bit_depth)]

    # --- MA mandatory player interface ---

    async def play_media(self, media: PlayerMedia) -> None:
        """Handle PLAY_MEDIA command."""
        await self._stop_playback()
        url = await self._provider.mass.streams.resolve_stream_url(self.player_id, media)
        self.logger.info("Starting digital audio playback from %s", url)
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
        """Handle PAUSE command — sets flag; _playback_loop sends SIGSTOP."""
        self._paused = True
        self._attr_playback_state = PlaybackState.PAUSED
        self.update_state()

    async def play(self) -> None:
        """Handle PLAY/resume command — clears flag; _playback_loop sends SIGCONT."""
        self._paused = False
        self._attr_playback_state = PlaybackState.PLAYING
        self.update_state()

    # --- Playback loop ---

    async def _playback_loop(self, url: str) -> None:
        """
        Fetch the MA stream and write it directly to the PA sink via ffmpeg -f pulse.

        The HA supervisor's libpulse-simple proxy only passes 2-channel audio.
        Using ffmpeg as a full PA client (`-f pulse`) bypasses the proxy and
        allows the sink's native channel count to be negotiated directly.

        Input format detection
        ----------------------
        MA's CONF_OUTPUT_CODEC URL resolution is non-deterministic across
        player restart cycles — sometimes it resolves to our intended PCM
        string, sometimes it falls back to 'flac'.  We therefore read the
        actual format from the URL extension and only declare explicit PCM
        input params when MA is genuinely serving raw PCM.  Assuming PCM
        unconditionally causes ffmpeg to misread FLAC bytes as PCM samples
        (hissy static); assuming UNKNOWN unconditionally causes ffmpeg to
        reject undeclared multichannel PCM (returncode=183).

        Pause
        -----
        SIGSTOP/SIGCONT freezes ffmpeg in place without closing the PA
        stream connection, preserving buffer state across pause/resume.
        SIGKILL is delivered to stopped processes on Linux without SIGCONT
        first, so stop-while-paused works correctly.
        """
        # Detect the actual stream format from the URL extension.
        url_fmt_str = url.rsplit(".", 1)[-1] if "." in url.rsplit("/", 1)[-1] else ""
        url_content_type = ContentType.try_parse(url_fmt_str)

        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

        if url_content_type.is_pcm():
            # MA is serving raw PCM — declare the real format explicitly so
            # ffmpeg does not attempt to probe undeclared multichannel bytes.
            input_args = [
                "-f", str(url_content_type.value),
                "-ar", str(self.sample_rate),
                "-ac", str(self.channels),
                "-i", url,
            ]
        else:
            # Compressed format (flac, aac, mp3, ...) — ffmpeg identifies it natively.
            input_args = ["-i", url]

        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            *input_args,
            "-acodec", "pcm_s32le",
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "-f", "pulse",
            "-name", f"music-assistant-{self._sink_name}",
            self._sink_name,
        ]

        self.logger.debug(
            "ffmpeg cmd: %s  (sink=%s %dch %dHz url_ct=%s)",
            " ".join(cmd),
            self._sink_name,
            self.channels,
            self.sample_rate,
            url_content_type,
        )

        proc: asyncio.subprocess.Process | None = None
        stderr_task: asyncio.Task[None] | None = None
        _was_paused = False

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            self.logger.info(
                "Digital audio playback started: sink=%s %dch %dHz pid=%d",
                self._sink_name, self.channels, self.sample_rate, proc.pid,
            )

            # Drain stderr in a background task to prevent pipe buffer saturation.
            stderr_task = self.mass.create_task(self._drain_stderr(proc))

            # Monitor ffmpeg lifetime; handle pause transitions via SIGSTOP/SIGCONT.
            while proc.returncode is None:
                if self._paused and not _was_paused:
                    with suppress(ProcessLookupError, OSError):
                        proc.send_signal(signal.SIGSTOP)
                    _was_paused = True
                elif not self._paused and _was_paused:
                    with suppress(ProcessLookupError, OSError):
                        proc.send_signal(signal.SIGCONT)
                    _was_paused = False
                await asyncio.sleep(0.1)

            # Log unexpected non-zero exits (0 = normal end, -15/-9 = killed by us).
            if proc.returncode not in (0, -signal.SIGTERM, -signal.SIGKILL):
                self.logger.warning(
                    "ffmpeg exited unexpectedly: returncode=%d sink=%s",
                    proc.returncode,
                    self._sink_name,
                )

        except asyncio.CancelledError:
            pass
        except Exception as err:
            self.logger.error("Playback error: %s", err, exc_info=True)
        finally:
            # Resume before killing: tidy, though SIGKILL reaches stopped processes.
            if _was_paused and proc is not None and proc.returncode is None:
                with suppress(ProcessLookupError, OSError):
                    proc.send_signal(signal.SIGCONT)
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await stderr_task
            if proc is not None and proc.returncode is None:
                with suppress(ProcessLookupError, OSError):
                    proc.kill()
                with suppress(Exception):
                    await proc.wait()
            self._attr_playback_state = PlaybackState.IDLE
            self._attr_current_media = None
            self._paused = False
            self.update_state()
            if self._playback_task is asyncio.current_task():
                self._playback_task = None

    async def _drain_stderr(self, proc: asyncio.subprocess.Process) -> None:
        """Consume ffmpeg stderr to prevent pipe buffer saturation; log at DEBUG level."""
        if proc.stderr is None:
            return
        try:
            async for line_bytes in proc.stderr:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "ffmpeg: %s", line_bytes.decode(errors="replace").rstrip()
                    )
        except asyncio.CancelledError:
            pass
        except Exception as err:
            self.logger.debug("ffmpeg stderr drain: %s", err)

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

    # --- Volume control (hardware sink via pactl) ---

    async def volume_set(self, volume_level: int) -> None:
        """Set volume via pactl on the PA sink (0-100)."""
        self._attr_volume_level = volume_level
        await self.mass.loop.run_in_executor(None, self._pactl_set_volume, volume_level)
        await self._save_state()
        self.update_state()

    async def volume_mute(self, muted: bool) -> None:
        """Mute/unmute via pactl on the PA sink."""
        self._attr_volume_muted = muted
        await self.mass.loop.run_in_executor(None, self._pactl_set_mute, muted)
        await self._save_state()
        self.update_state()

    def _pactl_set_volume(self, volume_level: int) -> None:
        """Blocking: set sink volume percentage via pactl."""
        import subprocess  # noqa: PLC0415

        subprocess.run(
            ["pactl", "set-sink-volume", self._sink_name, f"{volume_level}%"],
            timeout=3,
            check=False,
            capture_output=True,
        )

    def _pactl_set_mute(self, muted: bool) -> None:
        """Blocking: mute or unmute the sink via pactl."""
        import subprocess  # noqa: PLC0415

        subprocess.run(
            ["pactl", "set-sink-mute", self._sink_name, "1" if muted else "0"],
            timeout=3,
            check=False,
            capture_output=True,
        )

    async def apply_restored_volume(self) -> None:
        """Apply restored volume to the PA sink on startup.

        Clamps to DEFAULT_HARDWARE_VOLUME_CEILING so a corrupt or missing
        cache entry never causes full-blast output on the hardware sink.
        """
        volume = min(
            self._attr_volume_level or DEFAULT_PLAYER_VOLUME,
            DEFAULT_HARDWARE_VOLUME_CEILING,
        )
        self._attr_volume_level = volume
        await self.mass.loop.run_in_executor(None, self._pactl_set_volume, volume)
        if self._attr_volume_muted:
            await self.mass.loop.run_in_executor(None, self._pactl_set_mute, True)
        self.logger.debug(
            "Restored volume %d%% muted=%s on sink %s",
            volume,
            self._attr_volume_muted,
            self._sink_name,
        )

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

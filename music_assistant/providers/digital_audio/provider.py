"""Digital Audio player provider for Music Assistant."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from music_assistant.models.player_provider import PlayerProvider

from .constants import CONF_PA_SINK_NAME
from .player import DigitalAudioPlayer, get_player_uuid

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest
    from music_assistant.mass import MusicAssistant


class DigitalAudioProvider(PlayerProvider):
    """Player provider for direct multichannel PulseAudio output via ffmpeg -f pulse.

    Uses ffmpeg as a full PA client to write to the named hardware sink, bypassing
    the HA supervisor's 2-channel pa_simple proxy restriction.  Supports any PA sink:
    analog surround (5.1/7.1), HDMI audio, USB multichannel.  Requires no remap sinks
    or libpulse-simple.

    One provider instance = one PA sink = one MA player.  Use multiple instances for
    multiple hardware outputs (e.g. HDMI + analog surround simultaneously).
    """

    _player: DigitalAudioPlayer | None

    async def handle_async_init(self) -> None:
        """Verify ffmpeg with PulseAudio output support is available."""
        import shutil  # noqa: PLC0415
        import subprocess  # noqa: PLC0415

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError(
                "ffmpeg not found in PATH — is it installed in the MA container? "
                "(PR #3734 adds pulseaudio-utils; ffmpeg ships with the base image)"
            )
        result = subprocess.run(
            [ffmpeg_bin, "-formats"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if "pulse" not in result.stdout.lower():
            raise RuntimeError(
                "ffmpeg is not built with PulseAudio output support "
                "(-f pulse not available).  Check ffmpeg build flags in the MA image."
            )
        self._player = None

    async def loaded_in_mass(self) -> None:
        """Handle provider fully loaded in Music Assistant."""
        await self._register_player()

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/removal of the provider."""
        if self._player:
            with suppress(Exception):
                await self._player.stop_stream()
            self._player = None

    async def _register_player(self) -> None:
        """Register a single digital audio player from provider configuration."""
        sink_name = str(self.config.get_value(CONF_PA_SINK_NAME) or "")
        if not sink_name:
            self.logger.warning("No PA sink configured — skipping player registration")
            return

        # Query native sample rate, bit depth, and channel count from the sink.
        # channels == 0 means the sink was not found in pactl output.
        sample_rate, bit_depth, channels = await self.mass.loop.run_in_executor(
            None, _query_sink_format, sink_name
        )
        if channels == 0:
            self.logger.warning(
                "Sink '%s' not found in pactl list sinks — "
                "verify it exists and PulseAudio is running.  "
                "For HDMI sinks, the display must be physically connected so "
                "EDID negotiation can expose the sink.  "
                "Registering with 8ch fallback; playback may fail until the "
                "sink is available.",
                sink_name,
            )
            channels = 8  # 7.1 assumption; correct via re-registering once sink appears

        player_id = get_player_uuid(sink_name)
        # Derive a short display name from the sink's profile suffix, e.g.
        # "alsa_output.pci-0000_07_00.6.analog-surround-71" -> "analog-surround-71".
        display_name = f"Digital Audio ({sink_name.rsplit('.', 1)[-1]})"

        self._player = DigitalAudioPlayer(
            provider=self,
            player_id=player_id,
            sink_name=sink_name,
            display_name=display_name,
            channels=channels,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
        )
        await self._player.restore_state()
        await self._player.apply_restored_volume()
        await self.mass.players.register_or_update(self._player)

        self.logger.info(
            "Registered digital audio player: %s (%dch, %dHz, %dbit)",
            sink_name,
            channels,
            sample_rate,
            bit_depth,
        )

    async def cmd_volume_set(self, player_id: str, volume_level: int) -> None:
        """Set volume level (0-100) for the player."""
        if self._player and self._player.player_id == player_id:
            await self._player.volume_set(volume_level)

    async def cmd_volume_mute(self, player_id: str, muted: bool) -> None:
        """Mute/unmute the player."""
        if self._player and self._player.player_id == player_id:
            await self._player.volume_mute(muted)

    async def cmd_stop(self, player_id: str) -> None:
        """Send stop command to the player."""
        if self._player and self._player.player_id == player_id:
            await self._player.stop_stream()


def _query_sink_format(sink_name: str) -> tuple[int, int, int]:
    """
    Query native sample rate, bit depth, and channel count for a PA sink via pactl.

    :param sink_name: The PulseAudio sink name.
    :returns: Tuple of (sample_rate, bit_depth, channels).
              Falls back to (48000, 16, 0); channels=0 signals 'sink not found'.
    """
    import json  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    try:
        result = subprocess.run(
            ["pactl", "--format=json", "list", "sinks"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            for sink in json.loads(result.stdout):
                if sink.get("name") == sink_name:
                    # sample_specification format: "s32le 8ch 96000Hz"
                    spec_str: str = sink.get("sample_specification", "")
                    parts = spec_str.split()
                    fmt = parts[0]
                    channels = int(parts[1].replace("ch", ""))
                    sample_rate = int(parts[2].replace("Hz", ""))
                    bit_depth = int(
                        "".join(filter(str.isdigit, fmt.split("le")[0].split("be")[0]))
                    )
                    return sample_rate, bit_depth, channels
    except Exception:
        pass
    return 48000, 16, 0

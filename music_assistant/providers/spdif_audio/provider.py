"""S/PDIF Audio player provider for Music Assistant."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from music_assistant.models.player_provider import PlayerProvider

from .constants import CONF_AC3_BITRATE, CONF_PA_SINK_NAME, DEFAULT_AC3_BITRATE
from .player import SpdifAudioPlayer, get_player_uuid

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest
    from music_assistant.mass import MusicAssistant


class SpdifAudioProvider(PlayerProvider):
    """Player provider for optical/coax S/PDIF output via PulseAudio.

    Stereo content plays as plain PCM. 5.1 surround content is encoded to
    Dolby Digital (AC3) and wrapped in IEC 61937 framing for the receiver to
    decode as a passthrough bitstream. See player.py for the full design
    rationale, including why volume control is intentionally not exposed.
    """

    _player: SpdifAudioPlayer | None

    async def handle_async_init(self) -> None:
        """Verify ffmpeg (AC3 encoder, spdif muxer, pulse output) is available.

        Both the AC3 passthrough path's encode and write stages use ffmpeg —
        no pacat dependency. pulseaudio-utils on this project's Debian base
        (16.1+dfsg1-2+b1) installs pactl but not pacat, so the write stage
        uses a second ffmpeg process with -c:a copy (pure remux, zero
        processing) instead.
        """
        import shutil  # noqa: PLC0415
        import subprocess  # noqa: PLC0415

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError("ffmpeg not found in PATH — is it installed in the MA container?")

        if not shutil.which("ffprobe"):
            raise RuntimeError(
                "ffprobe not found in PATH — required to detect real source channel "
                "count before choosing the direct-PCM vs AC3-passthrough path. "
                "Normally ships alongside ffmpeg from the same build."
            )

        formats = subprocess.run(
            [ffmpeg_bin, "-formats"], capture_output=True, text=True, timeout=10, check=False,
        ).stdout.lower()
        if "pulse" not in formats:
            raise RuntimeError("ffmpeg is not built with PulseAudio output support (-f pulse).")
        if "spdif" not in formats:
            raise RuntimeError("ffmpeg is not built with the spdif muxer (-f spdif).")

        encoders = subprocess.run(
            [ffmpeg_bin, "-encoders"], capture_output=True, text=True, timeout=10, check=False,
        ).stdout.lower()
        if " ac3" not in encoders:
            raise RuntimeError("ffmpeg is not built with an AC3 encoder.")

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
        """Register a single S/PDIF player from provider configuration."""
        sink_name = str(self.config.get_value(CONF_PA_SINK_NAME) or "")
        if not sink_name:
            self.logger.warning("No PA sink configured — skipping player registration")
            return

        ac3_bitrate = str(self.config.get_value(CONF_AC3_BITRATE) or DEFAULT_AC3_BITRATE)
        player_id = get_player_uuid(sink_name)
        display_name = f"S/PDIF Audio ({sink_name.rsplit('.', 1)[-1]})"

        self._player = SpdifAudioPlayer(
            provider=self,
            player_id=player_id,
            sink_name=sink_name,
            display_name=display_name,
            ac3_bitrate=ac3_bitrate,
        )
        await self._player.restore_state()
        await self._player.apply_restored_volume()
        await self.mass.players.register_or_update(self._player)

        self.logger.info(
            "Registered S/PDIF player: %s (bitrate=%s)", sink_name, ac3_bitrate,
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

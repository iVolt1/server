"""S/PDIF Audio Out — provider registration."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from music_assistant.models.player_provider import PlayerProvider

from .constants import (
    CONF_ENCODING_FORMAT,
    CONF_PA_SINK_NAME,
    ENCODING_MAX_CHANNELS,
    UUID_NAMESPACE,
)
from .player import SPDIFPlayer

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest
    from music_assistant.mass import MusicAssistant

LOGGER = logging.getLogger(__name__)


class SPDIFAudioProvider(PlayerProvider):
    """Player provider that delivers IEC 61937 encoded audio to a PA IEC958 sink."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
        supported_features: set,
    ) -> None:
        super().__init__(mass, manifest, config)
        self._supported_features = supported_features
        self._player: SPDIFPlayer | None = None

    async def handle_async_init(self) -> None:
        """Async initialisation — register the player."""
        await self._register_player()

    async def unload(self, is_removed: bool = False) -> None:
        """Unload provider."""
        if self._player:
            self._player._attr_available = False
            self._player.update_state()
            await self._player.stop()

    async def _register_player(self) -> None:
        """Instantiate and register the SPDIFPlayer."""
        sink_name: str = self.config.get_value(CONF_PA_SINK_NAME)
        encoding: str = self.config.get_value(CONF_ENCODING_FORMAT)

        if not sink_name:
            LOGGER.warning("No PA sink configured — skipping player registration")
            return

        player_id = str(uuid.uuid5(UUID_NAMESPACE, sink_name))
        self._player = SPDIFPlayer(
            provider=self,
            player_id=player_id,
            sink_name=sink_name,
        )

        await self.mass.players.register_or_update(self._player)
        LOGGER.info(
            "Registered S/PDIF player '%s' — encoding=%s max_ch=%d",
            sink_name,
            encoding,
            ENCODING_MAX_CHANNELS.get(encoding, 6),
        )

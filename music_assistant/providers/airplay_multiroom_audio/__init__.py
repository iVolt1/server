"""Package entry point for the AirPlay Multiroom Audio provider.

setup() signature and pattern confirmed directly against the real
built-in AirPlay Receiver plugin's __init__.py: construct the provider
directly with (mass, manifest, config), no separate awaited setup() call
from here. The earlier version of this file guessed at a
construct-then-await-setup() pattern that was wrong -- almost certainly
why AirplayMultiroomProvider was missing an instance_id attribute, since
that's very likely set inside the base PlayerProvider's own __init__ from
manifest/config, which the wrong pattern never passed through at all.

Still unconfirmed: whether AirplayMultiroomProvider actually needs to
subclass a real PlayerProvider base class for this to work end to end
(almost certainly yes, given instance_id and presumably other attributes
come from that base __init__) -- see provider.py's own TODOs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .provider import AirplayMultiroomProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return AirplayMultiroomProvider(mass, manifest, config)
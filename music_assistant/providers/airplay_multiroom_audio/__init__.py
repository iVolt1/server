"""Package entry point for the AirPlay Multiroom Audio provider.

TODO: this file's shape (an async setup() returning the provider instance)
is a best-effort guess based on the common pattern of "manifest declares a
domain, MA dynamically imports that package and calls setup()" seen across
similar plugin-loading frameworks -- and is consistent with
get_config_entries() living as a PlayerProvider *instance* method rather
than a bare module function (something has to construct that instance
first). It is NOT confirmed against a real MA provider's actual __init__.py
in this session. Before relying on this: open an existing provider's
__init__.py (e.g. the built-in airplay provider, or local_audio if it's
still present) and compare the real setup() signature -- what arguments it
receives (mass instance? manifest? config?), and what it's expected to
return -- against what's sketched here, and correct as needed.
"""

from __future__ import annotations

from .provider import AirplayMultiroomProvider

# TODO: confirm real signature. This guess assumes MA's core calls
# setup(mass, manifest, config) and awaits a provider instance back, based
# on the general "dynamic import + setup() entry point" pattern -- but the
# actual parameter set, and whether config setup happens here or inside
# the provider's own async setup()/start() lifecycle method, needs
# verifying against real source before this is trustworthy.
async def setup(mass, manifest, config) -> AirplayMultiroomProvider:
    """Entry point MA's provider loader is expected to call."""
    provider = AirplayMultiroomProvider(mass)
    await provider.setup()
    return provider

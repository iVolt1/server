"""Fetch, verify, and cache the custom shairport-sync-pa binary.

No publicly-published shairport-sync build with --with-pa --with-soxr exists
to pull from an upstream registry (confirmed: MA's own bundled shairport-sync,
built in Dockerfile.base for the AirPlay Receiver plugin, is --with-stdout
--with-pipe only -- see the earlier conversation for how that was traced).

The practical equivalent of "pull it like addons do" is: this addon's own
repo builds the binary via CI (reusing the Dockerfile stage already written
and confirmed working -- alpine-builder, FROM ${BUILD_FROM}, --with-pa
--with-soxr, no --with-alsa) and publishes it as a GitHub Release asset, one
per architecture. This module downloads that asset on first run, verifies it
against a pinned SHA256 (same trust model as the Dockerfile's pinned git SHA
for shairport-sync itself), and caches it locally so subsequent starts don't
re-fetch.

TODO before this is usable: build_binaries.sh (or a GitHub Actions workflow)
needs to actually produce shairport-sync-pa-amd64 / shairport-sync-pa-aarch64
release assets from the confirmed-working alpine-builder Dockerfile stage,
and BINARY_RELEASES below needs real URLs + real sha256 digests filled in
from that build -- the values below are placeholders.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import platform
import stat
from pathlib import Path

import aiohttp

LOGGER = logging.getLogger(__name__)

# Pin to a specific release tag, not "latest" -- same reasoning as the
# Dockerfile pinning shairport-sync to a specific git SHA rather than
# building whatever HEAD happens to be. A moving target here would
# reintroduce the exact class of regression the addon's Dockerfile header
# comment describes (an unpinned build silently ignoring -p/port=).
RELEASE_TAG = "v0.1.0"  # TODO: set to your actual release tag
REPO = "iVolt1/hassio-apps"  # TODO: confirm final repo path

# TODO: fill in real sha256 digests once build_binaries.sh actually produces
# these assets. Get them with: sha256sum shairport-sync-pa-amd64
BINARY_RELEASES: dict[str, dict[str, str]] = {
    "x86_64": {
        "asset": "shairport-sync-pa-amd64",
        "sha256": "PLACEHOLDER_FILL_IN_AFTER_FIRST_RELEASE_BUILD",
    },
    "aarch64": {
        "asset": "shairport-sync-pa-aarch64",
        "sha256": "PLACEHOLDER_FILL_IN_AFTER_FIRST_RELEASE_BUILD",
    },
}


def _detect_arch() -> str:
    """Map Python's platform.machine() to the release asset naming."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    raise RuntimeError(f"Unsupported architecture for shairport-sync-pa: {machine}")


def _sha256_file(path: Path) -> str:
    """Compute the SHA256 of a file already on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def ensure_binary(cache_dir: Path) -> Path:
    """Ensure the shairport-sync-pa binary is present and verified in cache_dir.

    Returns the path to the verified, executable binary. Downloads and
    verifies it on first call; subsequent calls are a fast no-op if the
    cached copy already matches the pinned digest.

    Raises RuntimeError if the download fails, the digest doesn't match
    (do NOT run an unverified binary -- treat a mismatch the same as a
    failed `test "$(git rev-parse HEAD)" = "${SHAIRPORT_SHA}"` in the
    Dockerfile: hard stop, not a warning), or the architecture isn't
    supported.
    """
    arch = _detect_arch()
    release = BINARY_RELEASES.get(arch)
    if release is None:
        raise RuntimeError(f"No shairport-sync-pa release for architecture: {arch}")

    expected_sha256 = release["sha256"]
    asset_name = release["asset"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "shairport-sync-pa"

    # Fast path: already downloaded and still matches the pinned digest.
    if target.exists():
        actual = await asyncio.to_thread(_sha256_file, target)
        if actual == expected_sha256:
            return target
        LOGGER.warning(
            "Cached shairport-sync-pa digest mismatch (expected %s, got %s) "
            "-- re-downloading",
            expected_sha256,
            actual,
        )
        target.unlink()

    url = (
        f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{asset_name}"
    )
    LOGGER.info("Downloading shairport-sync-pa from %s", url)

    tmp_target = target.with_suffix(".tmp")
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"Failed to download shairport-sync-pa: HTTP {resp.status} for {url}"
                )
            with tmp_target.open("wb") as f:
                async for chunk in resp.content.iter_chunked(1024 * 1024):
                    f.write(chunk)

    actual_sha256 = await asyncio.to_thread(_sha256_file, tmp_target)
    if actual_sha256 != expected_sha256:
        tmp_target.unlink(missing_ok=True)
        raise RuntimeError(
            f"shairport-sync-pa SHA256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}. Refusing to install an unverified binary."
        )

    tmp_target.rename(target)
    target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    LOGGER.info("shairport-sync-pa installed and verified at %s", target)
    return target

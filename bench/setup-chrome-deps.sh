#!/usr/bin/env bash
# Installs the system libraries headless Chromium needs, without root.
#
# The usual fix is `playwright install-deps`, but that needs sudo and this box
# has no passwordless sudo. Instead we download the .deb packages and unpack
# them under bench/.cache/chrome-deps, then point LD_LIBRARY_PATH at them
# (bench/run.mjs does this automatically).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEBS="$HERE/.cache/chrome-debs"
ROOT="$HERE/.cache/chrome-deps"

mkdir -p "$DEBS" "$ROOT"
cd "$DEBS"

PACKAGES=(
	libglib2.0-0t64 libnspr4 libnss3 libatk1.0-0t64 libatk-bridge2.0-0t64
	libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0
	libasound2t64 libatspi2.0-0t64 libegl1 libgles2 libglvnd0 libgl1
	libx11-6 libxcb1 libxext6 libdrm2 libexpat1 libpcre2-8-0 libffi8
	libdbus-1-3 libxau6 libxdmcp6 libbsd0 libmd0 libselinux1 libsystemd0
	liblzma5 libzstd1 libcap2 libgcrypt20 libgpg-error0 libcups2t64
	libwayland-server0 libwayland-client0 libxkbcommon-x11-0 libxcb-render0
	libxcb-shm0 libxcb-glx0 libx11-xcb1 libmount1 libblkid1 libuuid1 libudev1
	libxrender1 libxi6 libcairo2 libpango-1.0-0 libpangocairo-1.0-0
	libharfbuzz0b libfontconfig1 libfreetype6 libpixman-1-0 libcairo-gobject2
	libgraphite2-3 libpng16-16t64 libbrotli1 libbz2-1.0 libavahi-common3
	libavahi-client3 libfribidi0 libthai0 libdatrie1
)

echo "downloading ${#PACKAGES[@]} packages…"
apt-get download "${PACKAGES[@]}" >/dev/null 2>&1 || true

echo "unpacking…"
for f in ./*.deb; do
	[ -f "$f" ] || continue
	dpkg-deb -x "$f" "$ROOT"
done

echo "done -> $ROOT"

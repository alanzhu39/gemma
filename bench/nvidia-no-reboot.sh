#!/usr/bin/env bash
# Builds and loads the NVIDIA kernel module WITHOUT rebooting.
#
# The blocker was not the driver: it was that Debian trixie ships headers only
# for 6.12.107+, while this machine runs 6.12.101. DKMS therefore had nothing to
# build against and silently produced no module. Upgrading the kernel would fix
# it but needs a reboot.
#
# The headers for the running kernel still exist in Debian's snapshot archive.
# This script installs those exact versions, builds the module for the kernel we
# are already running, and loads it in place. No reboot, no kernel change.
#
# Run as root:  sudo ./bench/nvidia-no-reboot.sh
set -euo pipefail

DKMS=/usr/sbin/dkms
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKGS=(
	"$HERE/.cache/nvidia-pkgs/hdr101.deb"
	"$HERE/.cache/nvidia-pkgs/linux-headers-6.12.101_deb13-common.deb"
	"$HERE/.cache/nvidia-pkgs/kbuild-amd64.deb"
)

if [ "$(id -u)" -ne 0 ]; then
	echo "must run as root: sudo $0" >&2
	exit 1
fi

KREL="$(uname -r)"
echo "running kernel: $KREL"

echo "==> 1/5 check the header packages are present and for the right architecture"
ARCH="$(dpkg --print-architecture)"
for p in "${PKGS[@]}"; do
	if [ ! -f "$p" ]; then
		echo "missing $p" >&2
		exit 1
	fi
	# Snapshot's binary index lists every architecture, and taking the first
	# entry silently downloaded an armel package once. Check before installing.
	pa="$(dpkg-deb -f "$p" Architecture)"
	if [ "$pa" != "$ARCH" ] && [ "$pa" != "all" ]; then
		echo "$p is for $pa, need $ARCH or all" >&2
		exit 1
	fi
	echo "    ok $(basename "$p") ($pa)"
done

echo "==> 2/5 install headers matching the RUNNING kernel (no upgrade)"
# linux-kbuild depends on pahole (used for BTF). Without it dpkg leaves the
# header packages unconfigured and the build silently has nothing to compile
# against, so install it first.
if ! command -v pahole >/dev/null 2>&1; then
	apt-get install -y pahole || {
		echo "could not install pahole (required by linux-kbuild)" >&2
		exit 1
	}
fi

dpkg -i "${PKGS[@]}" || {
	# A half-configured header package blocks the build, so fail loudly here
	# rather than at the compile step.
	echo "retrying dependency resolution" >&2
	apt-get -f install -y || {
		echo "dpkg failed; run 'sudo dpkg --configure -a' then retry" >&2
		exit 1
	}
}

# If anything is still unconfigured the build tree is incomplete, which shows up
# much later as a confusing compile error. Fail here instead.
if dpkg -l | grep -qE '^i[^i].*(linux-headers|linux-kbuild).*6\.12\.101'; then
	echo "header packages are still not configured:" >&2
	dpkg -l | grep -E 'linux-headers|linux-kbuild' >&2
	exit 1
fi

if [ ! -d "/lib/modules/$KREL/build" ]; then
	echo "/lib/modules/$KREL/build still missing - aborting before touching the module" >&2
	exit 1
fi
echo "    headers in place for $KREL"

echo "==> 3/5 build the NVIDIA module for $KREL"
"$DKMS" install nvidia/550.163.01 -k "$KREL" || {
	echo "--- build failed; log follows ---" >&2
	tail -40 "/var/lib/dkms/nvidia/550.163.01/build/make.log" >&2 || true
	exit 1
}
"$DKMS" status || true

KO=$(ls "/lib/modules/$KREL/updates/dkms/"nvidia*.ko* 2>/dev/null | head -1 || true)
if [ -z "$KO" ]; then
	echo "module not produced - not loading anything" >&2
	exit 1
fi
echo "    built $KO"

echo "==> 4/5 load it"
modprobe nvidia
sleep 1

echo "==> 5/5 verify"
if nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv 2>/dev/null; then
	echo
	echo "GPU is up. Next:"
	echo "  node bench/preflight.mjs      # expect shader-f16: yes"
	echo "  npm run bench:golden          # f16 hardware goldens"
else
	echo "module loaded but nvidia-smi still failing" >&2
	dmesg | tail -20 >&2 || true
	exit 1
fi

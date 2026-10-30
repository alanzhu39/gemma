#!/usr/bin/env bash
# Builds and loads the NVIDIA kernel module.
#
# Why the previous attempt failed:
#   - running kernel is 6.12.101, but Debian trixie only ships headers for
#     6.12.107, so DKMS had nothing to build against for the running kernel;
#   - `dkms` lives in /usr/sbin, which is not on the user PATH, so any
#     `dkms ...` invocation failed with "command not found" and the module was
#     never compiled.
#
# Result: no nvidia.ko, /dev/dri empty, nvidia-smi fails with
# "couldn't communicate with the NVIDIA driver", modprobe can't find
# nvidia-current.
#
# Run with sudo. Ends in a reboot, which is required: the module must exist for
# the kernel you are actually running, and the running kernel has no headers.
set -euo pipefail

DKMS=/usr/sbin/dkms

if [ "$(id -u)" -ne 0 ]; then
	echo "must run as root: sudo $0" >&2
	exit 1
fi

KREL="$(uname -r)"
echo "running kernel: $KREL"

echo "==> 1/5 install the kernel that has matching headers"
apt-get update
# trixie ships headers for 6.12.107; 6.12.101 has none, so we must move to 107.
apt-get install -y linux-image-6.12.107+deb13-amd64 linux-headers-6.12.107+deb13-amd64

echo "==> 2/5 verify DKMS is usable"
if [ ! -x "$DKMS" ]; then
	echo "dkms not found at $DKMS" >&2
	exit 1
fi
"$DKMS" --version

echo "==> 3/5 build the module against the new kernel"
# Build explicitly for the new kernel: the running one has no headers, so
# `dkms autoinstall` on its own would skip it.
"$DKMS" install nvidia/550.163.01 -k 6.12.107+deb13-amd64 || {
	echo "--- build failed; log follows ---" >&2
	tail -40 "/var/lib/dkms/nvidia/550.163.01/build/make.log" >&2 || true
	exit 1
}
"$DKMS" status

echo "==> 4/5 confirm the module exists"
ls -l "/lib/modules/6.12.107+deb13-amd64/updates/dkms/"nvidia*.ko* 2>/dev/null || {
	echo "module not produced - do not reboot yet" >&2
	exit 1
}

echo "==> 5/5 rebooting into 6.12.107"
echo "after reboot, verify with:  nvidia-smi && node bench/preflight.mjs"
reboot

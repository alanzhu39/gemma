#!/usr/bin/env bash
# Puts the GPU into a state where benchmarks measure the GPU, not its idle clock.
#
# Observed: `nvidia-smi` reports clocks.sm 139 MHz against a 1911 MHz max while
# idle, and it stays there during short benchmark runs. On Pascal the clock ramp
# takes on the order of 100 ms+, so a kernel that finishes in a few hundred
# milliseconds is largely measured at idle clock. That is why the driver swap
# bought only 2.5x when the clock ceiling changed by 13x.
#
# persistence mode keeps the driver initialised between runs; locking the
# graphics clock pins it. Both reset on reboot.
#
# Run as root:  sudo ./bench/set-gpu-performance.sh [clock_mhz|--reset]
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
	echo "must run as root: sudo $0" >&2
	exit 1
fi

if [ "${1:-}" = "--reset" ]; then
	nvidia-smi -rgc || true
	echo "graphics clock lock cleared; GPU will manage its own clocks"
	nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv
	exit 0
fi

MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1 | tr -d ' ')
TARGET="${1:-$MAX}"

echo "max SM clock: ${MAX} MHz; targeting ${TARGET} MHz"

# Persistence keeps the driver loaded, so consecutive runs do not each pay
# initialisation, and the clock does not fall back to idle between them.
nvidia-smi -pm 1
nvidia-smi -lgc "$TARGET"

echo
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,persistence_mode --format=csv
echo
echo "reset with: sudo $0 --reset"

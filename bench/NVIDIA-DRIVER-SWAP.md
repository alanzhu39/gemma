# Swapping nouveau for the NVIDIA proprietary driver

Why: nouveau cannot reclock Pascal (engine, memory and automatic reclocking are
all `TODO` in its support matrix), so the GTX 1060 idles at boot clocks — roughly
139 MHz against a 1709 MHz boost clock. It also does not expose `shader-f16`,
which the fp16 weight path requires. Both problems disappear with the
proprietary driver.

Verified preconditions on this machine:

- Bare metal (not a container), so a kernel module install will actually take.
- No display server running (`Xorg`/`Xwayland`/Wayland all absent), so swapping
  the GPU driver cannot break a session.
- `nouveau` refcount is 0 and `/sys/module/nouveau/holders` is empty, so it can
  be unloaded without a reboot.
- Legacy BIOS boot, no secure boot, so the module does not need signing.

## 1. Enable the non-free repo

The driver is in `non-free`, which is not enabled (only `non-free-firmware` is).
Add `non-free` to each source line:

```sh
sudo sed -i 's/ main non-free-firmware$/ main contrib non-free non-free-firmware/' \
  /etc/apt/sources.list
sudo apt update
```

Debian trixie ships `nvidia-graphics-drivers` 550.x, which supports GP106.

## 2. Blacklist nouveau and unload it

```sh
echo -e "blacklist nouveau\noptions nouveau modeset=0" \
  | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u
sudo modprobe -r nouveau
```

Confirm it is gone: `lsmod | grep nouveau` should print nothing.

## 3. Install the driver

```sh
sudo apt install -y nvidia-driver firmware-misc-nonfree
```

This pulls the kernel module plus the userspace Vulkan/GL libraries, including
the ICD that provides `shader-f16`.

## 4. Load it

```sh
sudo modprobe nvidia
nvidia-smi          # should now name the GTX 1060 and show clocks
```

If `modprobe` fails, the usual cause is a missing kernel header package:

```sh
sudo apt install -y linux-headers-$(uname -r)
sudo apt install --reinstall -y nvidia-kernel-dkms
```

## 5. Reboot (recommended, not strictly required)

The unload in step 2 avoids needing one, but a reboot is the cleanest way to
confirm the blacklist survives and that `nvidia` loads from a cold start. Safe
here — there is no display to lose.

## 6. Verify

```sh
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv
NVK_UNUSED=1 node bench/preflight.mjs
```

`bench/chrome.mjs` already sets `NVK_I_WANT_A_BROKEN_VULKAN_DRIVER=1`; that
variable is harmless with the proprietary ICD (it is only read by Mesa's NVK), so
no code change is needed. If anything breaks, that line is the first thing to
remove.

What you should see:

| | before (nouveau) | expected after |
| --- | --- | --- |
| `preflight` renderer | `NVK GP106` | `NVIDIA GeForce GTX 1060` |
| `shader-f16` | **NO** | **yes** |
| `precision` | `f32` | `f16` |
| SM clock | ~139 MHz | ~1700 MHz |

## 7. Re-establish the baseline

Everything measured so far is void (see `BASELINE-VOID.md`). After the swap:

```sh
node bench/preflight.mjs                        # must say shader-f16: yes
npm run bench:golden                            # new f16 hardware goldens
npm run bench -- --label baseline --repeats 5
cp "$(ls -t bench/results/baseline-*.json | head -1)" bench/baseline.json
```

Sanity-check the achieved bandwidth before trusting it. On this GPU the logits
projection should read 320 MiB of fp16 in roughly:

- 192 GB/s (nominal) → ~1.7 ms
- ~150 GB/s (realistic for GDDR5) → ~2.2 ms

If it is still hundreds of milliseconds, the driver did not take effect — check
the renderer string before investigating the kernel.

## Reverting

```sh
sudo apt remove --purge -y 'nvidia-*'
sudo rm /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u
sudo reboot
```

nouveau comes back automatically; nothing else on the system was changed.

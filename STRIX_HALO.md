# DS4 on Strix Halo — Setup Guide

Minimal setup for DS4 ROCm inference on a Strix Halo machine with 128 GB RAM
and Radeon 8060S (`gfx1151`).

## 1. Install ROCm

On Ubuntu 26.04 LTS (also works on Fedora 44 with equivalent packages):

```sh
sudo apt-get update
sudo apt-get install -y \
  hipcc rocminfo rocm-smi \
  libamdhip64-dev \
  libhipblas-dev libhipblaslt-dev \
  librocblas-dev \
  librocwmma-dev \
  libhipcub-dev
```

The backend uses rocWMMA. Install complete matching rocWMMA header tree:

```sh
git clone --depth 1 --branch rocm-7.1.0 https://github.com/ROCm/rocWMMA.git /tmp/rocWMMA-rocm-7.1.0
sudo mkdir -p /usr/local/include
sudo cp -a /tmp/rocWMMA-rocm-7.1.0/library/include/rocwmma /usr/local/include/
```

If ROCm is installed under `/usr` but tooling expects `/opt/rocm`:

```sh
sudo mkdir -p /opt/rocm/bin
sudo ln -sf /usr/bin/hipcc /opt/rocm/bin/hipcc
sudo ln -sfn /usr/lib/x86_64-linux-gnu /opt/rocm/lib
sudo ln -sfn /usr/include /opt/rocm/include
```

## 2. Enable ROCm access

```sh
sudo usermod -aG render,video "$USER"
```

Log out and back in, or reboot. Verify:

```sh
rocminfo | grep -A80 'Name:                    gfx1151'
```

## 3. Increase GPU-visible memory

Kernel parameters:

```text
amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856
```

On Ubuntu with GRUB:

```sh
sudo cp /etc/default/grub /etc/default/grub.bak
sudoedit /etc/default/grub
# Set:
# GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856"
sudo update-grub
sudo reboot
```

Verify after reboot:

```sh
cat /proc/cmdline
sudo dmesg | grep -Ei 'GTT|gttsize|TTM|VRAM'
rocminfo | grep -A80 'Name:                    gfx1151'
```

Expected:

```text
amdgpu:  126976M of GTT memory ready
rocminfo gfx1151 pool: 130023424 KB    (~124 GiB)
```

## 4. Build DS4

```sh
make strix-halo -j"$(nproc)"
```

`make rocm` is an alias for `make strix-halo`.

## 5. Use the right GGUF

Use the IQ2XXS/Q2K/Q8 imatrix GGUF:

```text
DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

Avoid mixed IQ2/IQ4 or IQ2/Q4 GGUFs on this machine — they put more memory
pressure on the ROCm path and can trigger system OOM.

## 6. Run DS4

```sh
./ds4 -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

The ROCm build uses the Strix Halo backend automatically.

---
title: "Tools: encoders & metrics"
description: "Supported encoders, decoders, and quality measurement tools available in the dev container."
---

## Image Encoders

### JPEG — `cjpeg` (libjpeg-turbo)

```bash
cjpeg -quality 85 input.ppm > output.jpg
```

- Quality range: 0–100 (typical: 75–95)
- Input: PPM format (use `convert` or Pillow to prepare)
- Sampling factor: `-sample 2x2` (4:2:0) or `-sample 1x1` (4:4:4)

### WebP — `cwebp`

```bash
cwebp -q 85 input.png -o output.webp
```

- Quality range: 0–100 (typical: 75–95)
- Method (`-m`): 0–6 (higher = slower, better compression)
- Sharp YUV conversion: `-sharp_yuv`
- Input: PNG or JPEG

### AVIF — `avifenc` (libaom backend)

```bash
avifenc -s 4 -q 60 input.png output.avif
```

- Quality (`-q`): 0–100 (typical: 50–85, very efficient at lower values)
- Speed (`-s`): 0–10 (higher = faster, lower quality; default: 6)
- YUV format: `-y 420`, `-y 444`

### JPEG XL — `cjxl`

```bash
cjxl input.png output.jxl -q 85
```

- Quality range: 0–100 (typical: 75–95)
- Effort (`-e`): 1–10 (higher = slower, better compression)
- Also supports distance mode (`-d`)

## Image Decoders

### AVIF — `avifdec`

```bash
avifdec input.avif output.png
```

Used by the quality measurement pipeline to decode AVIF files back to PNG for metric computation.

### JPEG XL — `djxl`

```bash
djxl input.jxl output.png
```

Used by the quality measurement pipeline to decode JXL files back to PNG for metric computation.

JPEG and WebP are decoded via Pillow (no separate CLI decoder needed).

## Quality Measurement Tools

### SSIMULACRA2

Perceptual quality metric designed specifically for lossy image compression. Higher is better.

```bash
ssimulacra2 original.png compressed.png
```

| Score | Quality |
|-------|---------|
| 90+   | Visually lossless |
| 70–90 | Excellent |
| 50–70 | Good |
| 30–50 | Medium |
| < 30  | Poor |

### Butteraugli — `butteraugli_main`

Perceptual distance metric that models the human visual system. Lower is better.

```bash
butteraugli_main reference.png distorted.png
```

| Score | Quality |
|-------|---------|
| < 1.0 | Excellent (imperceptible) |
| 1.0–2.0 | Good (minor artifacts) |
| 2.0–3.0 | Medium (noticeable) |
| > 3.0 | Poor |

Included in the JPEG XL static binaries.

### PSNR (via FFmpeg)

Peak Signal-to-Noise Ratio in dB. Higher is better.

```bash
ffmpeg -i original.png -i compressed.png -lavfi psnr -f null -
```

| Score | Quality |
|-------|---------|
| 40+ dB | Excellent |
| 30–40 dB | Good |
| 20–30 dB | Fair |
| < 20 dB | Poor |

### SSIM (via FFmpeg)

Structural Similarity Index (0–1). Higher is better.

```bash
ffmpeg -i original.png -i compressed.png -lavfi ssim -f null -
```

| Score | Quality |
|-------|---------|
| 0.95–1.0 | Excellent |
| 0.90–0.95 | Good |
| 0.80–0.90 | Fair |
| < 0.80 | Poor |

## Tool availability

All encoders, decoders and metric tools are pre-installed in the dev container and available on `PATH`. Check versions with:

```bash
cjpeg -version
cwebp -version
avifenc --version
cjxl --version
ffmpeg -version
ssimulacra2  # prints usage if no args
```

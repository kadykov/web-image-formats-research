# Reference: Image Encoding and Measurement Tools

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
- Sharp YUV conversion: `-sharp_yuv`
- Input: PNG or JPEG

### AVIF — `avifenc` (libaom backend)

```bash
avifenc -s 4 -q 60 input.png output.avif
```

- Quality (`-q`): 0–100 (typical: 50–85, very efficient at lower values)
- Speed (`-s`): 0–10 (higher = faster, lower quality; default: 6)
- Codec: libaom 3.11.0 (encoder), dav1d 1.5.0 (decoder)
- YUV format: `-y 420`, `-y 444`

### JPEG XL — `cjxl`

```bash
cjxl input.png output.jxl -q 85
```

- Quality range: 0–100 (typical: 75–95)
- Also supports distance mode (`-d`)
- Includes Butteraugli perceptual metric tool

## Quality Measurement Tools

### SSIMULACRA2

Perceptual quality metric. Higher is better.

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

Perceptual distance metric. Lower is better.

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

### VMAF (via FFmpeg)

Netflix's Video Multimethod Assessment Fusion metric.

```bash
ffmpeg -i compressed.png -i original.png -lavfi libvmaf -f null -
```

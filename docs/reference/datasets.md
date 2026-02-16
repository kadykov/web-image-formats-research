# Dataset Support and Roadmap

This document outlines the current and planned dataset support for the research project.

## Currently Supported

### DIV2K (2K Resolution)

**Status**: ✅ Fully implemented

The DIV2K dataset from ETH Zurich was the initial primary dataset for this project.

- **Resolution**: 2K (typically 2040×1356 or similar)
- **Format**: PNG (lossless, uncompressed)
- **Quality**: High-quality photographs
- **License**: Unknown - check DIV2K website
- **Source**: [DIV2K Challenge](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

**Available splits**:

- Validation: 100 images (~449MB)
- Training: 800 images (~3.5GB)

**Usage**:

```bash
just fetch div2k-valid  # Recommended for development
just fetch div2k-train  # For comprehensive research
```

**Pros**:

- Manageable size for testing
- High-quality reference images
- Widely used in image processing research
- Direct HTTP download (fast and reliable)

**Cons**:

- Limited to 2K resolution (may not fully test high-res optimizations)
- Relatively small number of images (especially validation set)

### LIU4K v1 (4K Resolution) ⭐

**Status**: ✅ Fully implemented (High-resolution dataset - **Recommended**)

Large-scale Ideal Ultra high definition 4K benchmark from Peking University. This is the original LIU4K benchmark.

- **Resolution**: 4K
- **Format**: PNG (lossless)
- **Quality**: High-quality diverse photographs
- **License**: CC BY-NC-ND 4.0 (Attribution required, non-commercial use, no derivatives)
- **Storage**: Google Drive (requires `gdown` library)
- **Source**: [LIU4K Benchmark](https://github.com/flyywh/Liu4K_benchmark)

**Available splits**:

- Validation: 80 images (~1.26GB)
- Training: 800 images (~10GB)

**Usage**:

```bash
just fetch liu4k-v1-valid  # Recommended for 4K research
just fetch liu4k-v1-train  # For comprehensive 4K training
```

**Pros**:

- High resolution (4K) - ideal for modern displays
- Lossless PNG format - no compression artifacts
- Manageable validation set size (1.26GB zip)
- Single-file zip archive (simpler than v2's multi-part archives)
- Automatic extraction with Python's zipfile module
- Original benchmark - widely cited

**Cons**:

- Google Drive storage (may have download quota limitations)
- CC BY-NC-ND 4.0 license (non-commercial, no derivatives)
- Smaller image count than v2 (but sufficient for research)
- Downloaded as zip archive (auto-extracted during fetch)

### LIU4K v2 (4K-6K Resolution)

**Status**: ✅ Fully implemented (Higher resolution, larger dataset)

Extended version of LIU4K with higher resolution and more images.

- **Resolution**: 4K-6K (most images between 4K-6K, minimum 3K)
- **Format**: PNG (lossless)
- **Quality**: Very high-quality diverse photographs
- **License**: CC BY-NC-ND 4.0 (Attribution required, non-commercial use, no derivatives)
- **Storage**: Google Drive - multi-part zip archives by category
- **Source**: [LIU4K Dataset v2](https://structpku.github.io/LIU4K_Dataset/LIU4K_v2.html)

**Available splits**:

- Validation: 400 images (~15GB as multi-part zips)
- Training: 1600 images (~60GB as multi-part zips)

**Usage**:

```bash
just fetch liu4k-v2-valid  # Advanced - requires extraction
just fetch liu4k-v2-train  # Very large download
```

**Pros**:

- Very high resolution (4K-6K) - excellent for high-res research
- Lossless PNG format
- Larger dataset (400/1600 images)
- Diverse content organized by category

**Cons**:

- Large download size (15GB validation, 60GB training)
- Multi-part zip archives require post-processing (automated)
- Google Drive storage (slower, quota limitations)
- CC BY-NC-ND 4.0 license (non-commercial only)

**Note**: LIU4K v2 datasets download as multi-part zip archives organized by category (Animal, Building, Capture, Mountain, Street). The system automatically extracts these archives after download.

### UHD-IQA (4K Resolution)

**Status**: ✅ Fully implemented

Ultra High Definition Image Quality Assessment benchmark from University of Konstanz.

- **Resolution**: UHD-1 (3840×2160 = 4K)
- **Format**: JPEG (lossy, sourced from Pixabay)
- **Quality**: High-quality photos with quality ratings (MOS)
- **License**: CC0 (Public Domain)
- **Storage**: Direct HTTP download
- **Source**: [UHD-IQA Database](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html)

**Available dataset**:

- Full: 6073 images (~10.7GB)
- Includes train, validation, and test splits
- Includes quality ratings and metadata (CSV files)

**Usage**:

```bash
just fetch uhd-iqa-full  # Full dataset with quality ratings
```

**Pros**:

- Large number of 4K images (6000+)
- Includes quality ratings for analysis
- Direct HTTP download (fast and reliable)
- Public domain (CC0)
- Comprehensive metadata

**Cons**:

- JPEG format (lossy) - contains compression artifacts from Pixabay
- Large download size (10.7GB)
- Not ideal for unbiased format comparison due to pre-existing artifacts

**Use Cases**:

- Large-scale 4K testing
- Quality assessment research
- Benchmarking with existing quality ratings
- Training ML models on quality prediction

## Dataset Comparison

| Feature | DIV2K | LIU4K v1 | LIU4K v2 | UHD-IQA |
|---------|-------|----------|----------|---------|
| **Resolution** | 2K | 4K | 4K-6K | 4K (3840×2160) |
| **Format** | PNG (lossless) | PNG (lossless) | PNG (lossless) | JPEG (lossy) |
| **Images (val)** | 100 | 80 | 400 | ~2000 |
| **Images (train)** | 800 | 800 | 1600 | ~4000 |
| **Size (val)** | 449MB | 1.26GB | 15GB | ~3.5GB |
| **Size (train)** | 3.5GB | 10GB | 60GB | ~7GB |
| **Storage** | Direct HTTP | Google Drive | Google Drive | Direct HTTP |
| **License** | Unknown | CC BY-NC-ND 4.0 | CC BY-NC-ND 4.0 | CC0 |
| **Post-process** | None | None | Extract zips | None |
| **Best for** | 2K research, dev | 4K research | 4K-6K large-scale | Large 4K dataset |

## Recommendations by Use Case

### Development & Testing
- **DIV2K Validation** (100 images, 2K, 449MB) - Fastest, most manageable
- **LIU4K v1 Validation** (80 images, 4K, 1.3GB) - Best for 4K dev/testing

### Production Research
- **2K Displays**: DIV2K Training (800 images, 2K, 3.5GB)
- **4K Displays**: LIU4K v1 Training (800 images, 4K, 10GB)
- **4K-6K Displays**: LIU4K v2 Validation or Training (400-1600 images, advanced)

### Large-Scale Studies
- **UHD-IQA Full** (6073 images, 4K, 10.7GB) - Despite JPEG, useful for scale
- **LIU4K v2 Training** (1600 images, 4K-6K, 60GB) - Very large, high-res

### Format Comparison Research
- **Prefer DIV2K or LIU4K v1/v2** (lossless) - Avoid pre-existing compression artifacts
- **Avoid UHD-IQA for format research** - JPEG artifacts may bias results

### License Considerations
- **Commercial use**: Use DIV2K or UHD-IQA (though DIV2K license unclear)
- **Non-commercial research**: All datasets available
- **Unrestricted/Public domain**: UHD-IQA only (CC0)

## Planned Support

### Flickr2K

**Status**: 🔄 Planned

Extension of DIV2K with 2650 additional 2K resolution images from Flickr -

. Could provide more 2K diversity.

- **Resolution**: 2K
- **Format**: PNG
- **Size**: ~7GB
- **Source**: [EDSR Project](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)

**Implementation notes**:

- Would add configuration entry to `datasets.json`
- Requires TAR archive support (already implemented)
- Could be combined with DIV2K for larger test set
- Lower priority now that we have LIU4K-v2 for higher resolution

### 8K+ Ultra High Resolution

**Status**: 🔍 Under investigation

**Challenge**: 8K datasets are typically very large (50GB+) and not practical for automated fetching.

**Requirements for ideal 8K dataset**:

- 8K resolution or higher (7680×4320+)
- Lossless format (PNG, TIFF)
- Manageable validation set (<5GB)
- No compression artifacts
- Diverse content

**Current limitations**:

- Most 8K datasets require manual requests/registration
- File sizes make automated fetching impractical
- Limited availability of freely licensed 8K content

**Potential future sources**:

- Camera manufacturer sample images (Canon, Nikon, Sony 8K samples)
- 8K video frame extraction (if suitable licensing)
- Scientific/satellite imagery at ultra-high resolution

### HuggingFace Datasets

**Status**: 💡 Concept stage

HuggingFace hosts numerous image datasets that could be useful.

**Implementation considerations**:

- Would require `datasets` library dependency
- Need to filter for high-resolution, high-quality images
- Many HF datasets are designed for ML training (labeled, preprocessed)
- Would add `fetch_huggingface()` method with dataset identifier

**Example datasets to evaluate**:

- `imagenet-1k` (if uncompressed high-res available)
- `coco` (if suitable resolution available)
- Photography-specific datasets

## Dataset Selection Criteria

When evaluating new datasets, we consider:

1. **Resolution**: Minimum 2K, preferably 4K or higher
2. **Quality**: No visible compression artifacts
3. **Format**: PNG or uncompressed TIFF preferred
4. **Size**: Manageable for development and CI
5. **Diversity**: Various scenes, lighting, content types
6. **Licensing**: Clear license for research use
7. **Accessibility**: Direct download without complex authentication

## Custom Dataset Curation

**Status**: 🤔 Under consideration

For maximum control and high-resolution testing, we could curate our own dataset:

**Advantages**:

- Precise resolution control (24MP target)
- Content diversity optimization
- Known quality and provenance
- Size optimization for research needs

**Challenges**:

- Time-intensive selection process
- Need clear selection criteria
- Copyright/licensing management
- Quality verification workflow

**Potential sources**:

- Unsplash API (free high-res, good licensing)
- Pexels API (free high-res)
- Flickr API (with appropriate licensing filters)
- Wikimedia Commons (public domain)

## Implementation Architecture

The `DatasetFetcher` class is designed to be extensible:

```python
class DatasetFetcher:
    def fetch_div2k(self, split: str) -> Path | None:
        """Implemented"""
        
    def fetch_flickr2k(self) -> Path | None:
        """Planned - similar to DIV2K"""
        
    def fetch_huggingface(self, dataset_id: str, split: str) -> Path | None:
        """Planned - requires HF datasets library"""
        
    def fetch_from_urls(self, urls: list[str], dataset_name: str) -> Path | None:
        """Planned - for custom curation"""
```

Each method follows the same pattern:

1. Check if dataset already exists
2. Download archive or files
3. Extract/organize files
4. Verify contents
5. Return dataset path

## Contributing

If you know of high-quality, high-resolution image datasets suitable for compression research, please:

1. Check the selection criteria above
2. Verify licensing permits research use
3. Open an issue with dataset details
4. Consider submitting a PR with implementation

## References

- [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- [Flickr2K Dataset](https://cv.snu.ac.kr/research/EDSR/)
- [HuggingFace Datasets](https://huggingface.co/datasets)
- [Unsplash API](https://unsplash.com/developers)
- [Pexels API](https://www.pexels.com/api/)

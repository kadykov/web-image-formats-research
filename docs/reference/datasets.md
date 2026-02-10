# Dataset Support and Roadmap

This document outlines the current and planned dataset support for the research project.

## Currently Supported

### DIV2K

**Status**: ✅ Fully implemented

The DIV2K dataset from ETH Zurich is the primary dataset for this project during development.

- **Resolution**: 2K (typically 2040×1356 or similar)
- **Format**: PNG (uncompressed)
- **Quality**: High-quality photographs
- **Source**: [DIV2K Challenge](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

**Available splits**:

- Validation: 100 images (~500MB)
- Training: 800 images (~4GB)

**Usage**:

```bash
just fetch-div2k-valid  # Recommended for development
just fetch-div2k-train  # For comprehensive research
```

**Pros**:

- Manageable size for testing
- High-quality reference images
- Widely used in image processing research

**Cons**:

- Limited to 2K resolution (may not test high-res encoder optimizations)
- Relatively small number of images

## Planned Support

### Flickr2K

**Status**: 🔄 Planned

Extension of DIV2K with 2650 additional 2K resolution images from Flickr.

- **Resolution**: 2K
- **Format**: PNG
- **Size**: ~7GB
- **Source**: [EDSR Project](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)

**Implementation notes**:

- Would add `fetch_flickr2k()` method to `DatasetFetcher`
- Requires TAR archive support (already implemented)
- Could be combined with DIV2K for larger test set

### High-Resolution Datasets (4K+)

**Status**: 🔍 Under investigation

**Challenge**: Most high-resolution datasets are very large and designed for computer vision tasks rather than compression research.

**Requirements for ideal dataset**:

- 4K or higher resolution (ideally 24MP)
- High-quality photographs (not synthetic)
- Manageable size (<10GB for validation set)
- No compression artifacts
- Diverse content (landscapes, portraits, objects, etc.)

**Potential sources being evaluated**:

- Camera manufacturer sample images
- Open photo platforms (Unsplash, Pexels) with licensing
- Scientific/satellite imagery datasets
- Curated subset of larger computer vision datasets

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

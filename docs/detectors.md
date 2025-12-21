# Detectors

Each detector looks for different types of manipulation artifacts. Here's what they do and when they're useful.

## Color Stats

**What it checks:** RGB channel correlations

**How it works:** In real photos, color channels are correlated (R-G around 0.9, G-B around 0.9, R-B around 0.85). That's just how light works. When you edit colors, these relationships often break.

**Good for:**

- Color grading inconsistencies
- Spliced regions from different lighting
- Unnatural color manipulation

**Bad for:**

- Grayscale images
- Images that are naturally weird (intentional artistic color effects)

**Example:** If you paste a person from a sunny photo into an overcast scene, the RGB correlations in the pasted region won't match.

---

## Neighbor Consistency

**What it checks:** Pixel-to-neighbor relationships

**How it works:** Predicts each pixel from its 8 neighbors. In natural images, this works pretty well. Splice boundaries and edited regions often fail this test.

**Good for:**

- Copy-paste detection
- Splice boundaries
- Inpainting artifacts

**Bad for:**

- Genuine edges and textures (can false-positive)

**Example:** When you copy-paste an object, the boundary pixels don't match their new neighbors as well as they should.

---

## FFT/DCT

**What it checks:** Frequency domain patterns

**How it works:** Natural images follow a ~1/f^2 power law in frequency space. Resampling, JPEG compression, and synthetic generation mess this up. Also catches periodic patterns that shouldn't be there.

**Good for:**

- Detecting resizing/rotation
- JPEG compression inconsistencies
- AI-generated content (sometimes)
- Periodic artifacts from generation

**Bad for:**

- Small patches (needs at least 16x16)
- Genuinely textured regions

**Example:** If part of an image was resized, it'll have different frequency characteristics than the rest.

---

## Residual Energy

**What it checks:** High-frequency noise levels

**How it works:** Removes low-frequency content with Gaussian blur, measures what's left. Real photos have sensor noise and fine texture. Over-edited or AI-generated regions are often too smooth.

**Good for:**

- Detecting smoothing/blurring
- AI-generated backgrounds (often too clean)
- Over-processed regions

**Bad for:**

- Naturally smooth areas (sky, walls)
- Out-of-focus regions

**Example:** AI-generated backgrounds often lack the natural sensor noise present in real photos.

---

## DCT Co-occurrence

**What it checks:** JPEG compression patterns

**How it works:** Analyzes 8x8 DCT block coefficients. Images with consistent JPEG history have consistent patterns. Splicing from different sources or double compression creates mismatches.

**Good for:**

- Detecting regions from different images
- Double JPEG compression
- Mixed compression quality

**Bad for:**

- Uncompressed images (less useful)
- Images with complex JPEG history

**Example:** If you paste a high-quality JPEG face onto a low-quality background, the DCT patterns won't match.

---

## PRNU (Stub)

**What it checks:** Sensor noise patterns

**Status:** ! Currently a placeholder - just uses gradient magnitude

**How it should work:** Real PRNU extracts camera sensor fingerprints using wavelet denoising. Each camera has a unique noise pattern. Manipulated regions either lack this or have mismatched patterns.

**Current implementation:** Just measures gradient magnitude as a proxy for high-frequency content.

**TODO:** Replace with proper wavelet-based PRNU extraction

**Why it matters:** When implemented properly, PRNU is one of the most reliable forensic techniques. But it's also computationally expensive.

---

## JPEG Ghost

**What it checks:** JPEG recompression artifacts

**How it works:** Recompresses the patch at various JPEG quality levels (50-95) and measures the difference from the original. If a region was originally compressed at quality Q, recompressing at Q produces minimal change. Different regions from different JPEG sources will have different "ghost" patterns.

**Good for:**

- Spliced images from multiple JPEG sources
- Copy-paste from different quality images
- Detecting manipulation in JPEG images

**Bad for:**

- Uncompressed source images
- Images with heavy post-processing
- AI-generated images (they haven't been through JPEG)

**Example:** If you paste a face from a high-quality JPEG (Q=90) into a low-quality background (Q=70), the JPEG ghost detector will show inconsistent patterns.

---

## GAN Fingerprint

**What it checks:** Neural network upsampling artifacts

**How it works:** GANs and AI generators often use transposed convolutions for upsampling, which leaves checkerboard patterns in the frequency domain. This detector looks at FFT patterns for:

- Periodic peaks at quarter/half frequencies (from 2x/4x upsampling)
- Spectral slope deviation from natural 1/f pattern
- Unusual frequency symmetry

**Good for:**

- Detecting AI-generated images (DALL-E, Midjourney, Stable Diffusion)
- GAN-generated faces
- Neural network-based upscaling artifacts

**Bad for:**

- Real photographs
- Traditional edits (copy-paste, color correction)
- Very small patches

**Example:** GAN-generated faces often have telltale periodic patterns in their frequency spectrum that real camera images don't produce.

---

## Noise Consistency

**What it checks:** Unnatural noise patterns

**How it works:** Real cameras produce consistent sensor noise across the entire image - typically Gaussian/Poisson distributed and correlated across RGB channels. AI-generated images often have:

- Regions that are too smooth (no noise)
- Inconsistent noise levels in different areas
- Wrong noise distribution (not Gaussian)
- Uncorrelated noise between color channels

This detector measures noise variance, distribution shape (skewness, kurtosis), regional consistency, and cross-channel correlation.

**Good for:**

- AI-generated images (often too clean)
- Heavily processed/inpainted regions
- Synthetic backgrounds

**Bad for:**

- High-ISO photos (legitimate heavy noise)
- Professionally denoised photos
- Very smooth natural scenes (sky, water)

**Example:** An AI-generated person might have perfectly smooth skin with no sensor noise, while the background has some noise - this inconsistency is suspicious.

---

## Benford's Law

**What it checks:** First-digit distribution in DCT coefficients

**How it works:** Benford's Law says that in real-world data, the first digit isn't evenly distributed - 1 appears more than 2, 2 more than 3, etc. DCT coefficients from real photos follow this pattern. Edited and AI images often don't.

**Good for:**

- General manipulation detection
- AI-generated images
- Any kind of synthetic content

**Bad for:**

- Very small patches
- Heavily textured images

**Research:** Achieves ~90% F1-score according to published studies. One of the most mathematically robust forensic techniques.

---

## CFA Artifact (Demosaicing)

**What it checks:** Presence of Bayer pattern demosaicing artifacts

**How it works:** Real cameras have Bayer sensors - each pixel only sees one color (R, G, or B). The camera fills in the missing colors through interpolation (demosaicing), which creates specific patterns between pixels. AI images are made directly in RGB, so they don't have these patterns.

**Good for:**

- AI-generated images (most reliable single detector)
- Any synthetic content
- Distinguishing real cameras from computer graphics

**Bad for:**

- Images with heavy post-processing
- Specific camera brands with unusual CFA patterns

**Why it's powerful:** AI can't fake what happens inside a real camera sensor. This physical difference is really hard to get around - probably the best single feature for catching AI images.

---

## LBP Texture

**What it checks:** Local texture patterns

**How it works:** LBP compares each pixel to its 8 neighbors - brighter neighbor = 1, darker = 0. This gives an 8-bit pattern per pixel. Real images have certain patterns that show up often; AI images tend to have more uniform or weird distributions.

**Good for:**

- Deepfakes and AI-generated faces
- General AI detection
- Texture-based manipulation

**Bad for:**

- Smooth natural scenes
- Very small patches

**Research:** LBP-based methods achieve 83-99% accuracy on deepfake datasets.

---

## Chromatic Aberration

**What it checks:** Lens chromatic aberration consistency

**How it works:** Real lenses can't focus all colors to the same spot - red and blue shift slightly in opposite directions, especially near the edges. AI images don't go through real optics, so they either have no color fringing or it looks wrong.

**Good for:**

- AI-generated images
- Detecting spliced regions with different CA patterns
- Source identification

**Bad for:**

- Images with CA already corrected
- Low-contrast images with few edges

---

## PRNU Wavelet

**What it checks:** Sensor noise patterns using wavelet denoising

**How it works:** Each camera sensor has unique manufacturing imperfections that create a Photo Response Non-Uniformity (PRNU) - a consistent noise pattern in all images from that camera. This detector extracts the noise residual using wavelet-based denoising and analyzes its characteristics. AI images lack real sensor noise patterns.

**Good for:**

- AI-generated images (no real sensor noise)
- Heavily processed images
- Source camera identification

**Bad for:**

- High-ISO photos (strong legitimate noise)
- Images with artificial noise added

---

## Comparison

| Detector | Speed | Best For | Weakness | AI Detection |
|----------|-------|----------|----------|--------------|
| color_stats | Fast | Color edits | Grayscale images | Low |
| neighbor_consistency | Fast | Splice boundaries | Genuine edges | Low |
| fft_dct | Slow | Resampling, JPEG | Small patches | Medium |
| residual_energy | Medium | Smoothing, AI gen | Naturally smooth areas | Medium |
| dct_cooccurrence | Medium | JPEG inconsistencies | Uncompressed images | Low |
| ela | Medium | JPEG manipulation | Uncompressed images | Medium |
| lighting_consistency | Medium | Composites, splicing | Multiple light sources | Medium |
| copy_move | Slow | Duplicated regions | Small patches | Medium |
| jpeg_ghost | Slow | JPEG splicing | Uncompressed/AI images | Low |
| gan_fingerprint | Medium | AI upsampling | Real photos | High |
| noise_consistency | Medium | AI-generated/smooth | High-ISO photos | High |
| **benford_law** | Medium | All manipulations | Small patches | **Very High** |
| **cfa_artifact** | Fast | AI detection | Post-processed | **Highest** |
| **lbp_texture** | Medium | Deepfakes, AI | Smooth scenes | **Very High** |
| **chromatic_aberration** | Medium | AI, splicing | CA-corrected | High |
| **prnu_wavelet** | Slow | AI, source ID | High-ISO | High |

## Using them together

The ensemble combines all detectors using z-score normalization and weighted averaging. This helps because:

- Different manipulations leave different traces
- One detector's false positive might be corrected by others
- You can weight detectors based on what you trust more

Example config:

```json
{
  "enabled_detectors": ["color_stats", "fft_dct", "neighbor_consistency"],
  "detector_weights": {
    "fft_dct": 2.0,           // trust frequency analysis more
    "color_stats": 1.0,
    "neighbor_consistency": 0.5   // less reliable in this dataset
  }
}
```

## Adding your own

See `CONTRIBUTING.md` for how to add a new detector. It's pretty straightforward - just implement the `score()` method and register it. With auto-discovery, you just drop a file in `priorpatch/detectors/` and it's automatically loaded.

Some ideas for new detectors:

- Shadow direction consistency
- Perspective/vanishing point consistency
- Reflection analysis
- Metadata consistency checking

## References

These algorithms are based on various forensics papers:

- Popescu & Farid: Copy-move forgery detection, CFA artifact detection
- Lukas et al: PRNU-based camera identification (IEEE Trans. IFS, 2006)
- Farid: Frequency analysis for manipulation detection
- Mahdian & Saic: Using inconsistencies for detection
- Fu et al: Benford's Law for JPEG coefficients (2007)
- Johnson & Farid: Chromatic aberration for forgery detection (2006)
- Ojala et al: Local Binary Patterns for texture classification
- Bonettini et al: Benford's Law for GAN detection (2020)

Full references in academic papers about image forensics.

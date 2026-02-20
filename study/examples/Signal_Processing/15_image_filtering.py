#!/usr/bin/env python3
"""
Image Signal Processing: 2D DFT, Frequency-Domain and Spatial Filtering
=========================================================================

Images are 2-dimensional signals.  Every technique from 1-D signal processing
has a 2-D counterpart, and the connections are direct and elegant.

2-D Discrete Fourier Transform (DFT)
--------------------------------------
For an M×N image f(m, n):

    F(u, v) = sum_{m=0}^{M-1}  sum_{n=0}^{N-1}
              f(m, n) * exp(-j*2*pi*(u*m/M + v*n/N))

    - DC component F(0,0) = mean pixel value * M*N
    - After fftshift: low frequencies are at the centre
    - Convolution theorem: filtering in spatial domain ≡ multiplication in freq domain

2-D Filtering
--------------
Frequency domain:
    F_filtered(u, v) = H(u, v) * F(u, v)   (element-wise)
    then ifft2 → spatial result

Spatial domain:
    g(m, n) = f(m, n) ** h(m, n)            (2-D convolution)

Both are equivalent for linear shift-invariant (LSI) filters.  The frequency-
domain approach is often more intuitive for designing ideal brick-wall filters,
while the spatial domain is faster for small kernels.

Author: Educational example for Signal Processing
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


# ============================================================================
# SYNTHETIC TEST IMAGE
# ============================================================================

def make_test_image(size=256):
    """
    Create a synthetic test image combining:
        - Checkerboard pattern (broad spectrum, high-frequency edges)
        - Gaussian-blurred circles (low-frequency smooth features)
        - Additive Gaussian noise

    Args:
        size (int): Image size (size × size pixels)

    Returns:
        ndarray: Float64 image in [0, 1], shape (size, size)
    """
    img = np.zeros((size, size), dtype=float)

    # Checkerboard: 16×16 pixel squares
    sq = size // 16
    for i in range(16):
        for j in range(16):
            if (i + j) % 2 == 0:
                img[i*sq:(i+1)*sq, j*sq:(j+1)*sq] = 0.6

    # Smooth circles on top
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    for r, amp in [(60, 0.4), (30, -0.3), (15, 0.3)]:
        mask = (x - cx) ** 2 + (y - cy) ** 2 < r ** 2
        img[mask] += amp

    # Gaussian blur on the circles to make them smooth
    img = ndimage.gaussian_filter(img, sigma=3)

    # Add noise
    rng = np.random.default_rng(42)
    img += 0.05 * rng.standard_normal(img.shape)

    # Clip to [0, 1]
    img = np.clip(img, 0, 1)
    return img


# ============================================================================
# 2-D DFT UTILITIES
# ============================================================================

def compute_2d_spectrum(img):
    """
    Compute 2-D FFT and return the log-magnitude spectrum (shifted to centre).

    Args:
        img (ndarray): 2-D real image

    Returns:
        F     (ndarray): Complex DFT coefficients (shifted)
        mag   (ndarray): Log-magnitude spectrum for display
    """
    F_raw = np.fft.fft2(img)
    F = np.fft.fftshift(F_raw)
    mag = np.log1p(np.abs(F))      # log(1 + |F|) avoids log(0)
    return F, mag


def freq_axes(img):
    """
    Return normalised frequency axes (cycles/pixel) for a shifted 2-D DFT.

    Returns:
        fu, fv (ndarray): 1-D frequency vectors, each in [-0.5, 0.5].
    """
    M, N = img.shape
    fu = np.fft.fftshift(np.fft.fftfreq(M))
    fv = np.fft.fftshift(np.fft.fftfreq(N))
    return fu, fv


# ============================================================================
# FREQUENCY-DOMAIN FILTERS
# ============================================================================

def ideal_lowpass(img, cutoff):
    """
    Ideal (brick-wall) circular lowpass filter in the frequency domain.

    H(u, v) = 1 if sqrt(u^2 + v^2) <= cutoff, else 0

    Very sharp cutoff → significant Gibbs ringing in spatial domain.

    Args:
        img    (ndarray): Input image
        cutoff (float)  : Cutoff radius in normalised frequency [0, 0.5]

    Returns:
        filtered (ndarray): Filtered image (real part of IFFT)
        H        (ndarray): Filter mask (shifted), for display
    """
    F_raw = np.fft.fft2(img)
    F = np.fft.fftshift(F_raw)

    fu, fv = freq_axes(img)
    UU, VV = np.meshgrid(fu, fv, indexing='ij')
    dist = np.sqrt(UU ** 2 + VV ** 2)
    H = (dist <= cutoff).astype(float)

    F_filtered = np.fft.ifftshift(F * H)
    filtered = np.real(np.fft.ifft2(F_filtered))
    return filtered, H


def gaussian_lowpass(img, sigma_freq):
    """
    Gaussian lowpass filter in the frequency domain.

    H(u, v) = exp(-(u^2 + v^2) / (2*sigma^2))

    Smooth roll-off → no ringing.  The spatial-domain kernel is also Gaussian.

    Args:
        img        (ndarray): Input image
        sigma_freq (float)  : Standard deviation in normalised frequency

    Returns:
        filtered (ndarray): Filtered image
        H        (ndarray): Gaussian filter mask (shifted)
    """
    F_raw = np.fft.fft2(img)
    F = np.fft.fftshift(F_raw)

    fu, fv = freq_axes(img)
    UU, VV = np.meshgrid(fu, fv, indexing='ij')
    dist2 = UU ** 2 + VV ** 2
    H = np.exp(-dist2 / (2 * sigma_freq ** 2))

    F_filtered = np.fft.ifftshift(F * H)
    filtered = np.real(np.fft.ifft2(F_filtered))
    return filtered, H


def ideal_highpass(img, cutoff):
    """
    Ideal highpass filter: H_hp = 1 - H_lp.

    Removes low-frequency content; enhances edges and fine details.

    Args:
        img    (ndarray): Input image
        cutoff (float)  : Cutoff radius in normalised frequency

    Returns:
        filtered (ndarray): Filtered image
        H        (ndarray): Highpass filter mask
    """
    F_raw = np.fft.fft2(img)
    F = np.fft.fftshift(F_raw)

    fu, fv = freq_axes(img)
    UU, VV = np.meshgrid(fu, fv, indexing='ij')
    dist = np.sqrt(UU ** 2 + VV ** 2)
    H = (dist > cutoff).astype(float)

    F_filtered = np.fft.ifftshift(F * H)
    filtered = np.real(np.fft.ifft2(F_filtered))
    return filtered, H


# ============================================================================
# SPATIAL-DOMAIN FILTERS
# ============================================================================

def sobel_edges(img):
    """
    Sobel edge detection using 3×3 derivative kernels.

    Gx detects horizontal changes, Gy detects vertical changes.
    Edge magnitude: G = sqrt(Gx^2 + Gy^2)

    Args:
        img (ndarray): Grayscale image

    Returns:
        edges (ndarray): Edge magnitude image
    """
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)
    Ky = Kx.T    # transposing Kx gives the vertical kernel

    Gx = ndimage.convolve(img, Kx)
    Gy = ndimage.convolve(img, Ky)
    edges = np.hypot(Gx, Gy)
    return edges


def gaussian_blur_spatial(img, sigma):
    """
    Gaussian blur via direct spatial convolution (SciPy implementation).

    The 2-D Gaussian kernel:
        h(m, n) = (1/(2*pi*sigma^2)) * exp(-(m^2+n^2)/(2*sigma^2))

    Args:
        img   (ndarray): Input image
        sigma (float)  : Blur radius in pixels

    Returns:
        ndarray: Blurred image
    """
    return ndimage.gaussian_filter(img, sigma=sigma)


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def show_spectrum_pair(ax_img, ax_spec, image, title_img, title_spec, cmap='gray'):
    """Display an image and its log-magnitude spectrum side-by-side on given axes."""
    ax_img.imshow(image, cmap=cmap, vmin=0, vmax=1)
    ax_img.set_title(title_img)
    ax_img.axis('off')

    _, mag = compute_2d_spectrum(image)
    ax_spec.imshow(mag, cmap='hot')
    ax_spec.set_title(title_spec)
    ax_spec.axis('off')


# ============================================================================
# DEMO SECTIONS
# ============================================================================

def demo_spectrum(img):
    """Visualise the test image and its 2-D DFT magnitude spectrum."""
    print("=" * 60)
    print("SECTION 1: 2-D DFT Magnitude Spectrum")
    print("=" * 60)

    F, mag = compute_2d_spectrum(img)
    dc = np.abs(np.fft.fft2(img)[0, 0])
    print(f"  Image size : {img.shape}")
    print(f"  Mean pixel : {img.mean():.3f}")
    print(f"  DC magnitude F(0,0) = {dc:.1f}  (≈ mean × {img.shape[0]} × {img.shape[1]} = "
          f"{img.mean()*img.shape[0]*img.shape[1]:.1f})")
    print(f"  Log spectrum range  : [{mag.min():.2f}, {mag.max():.2f}]")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Test Image and its 2-D DFT Magnitude Spectrum",
                 fontsize=12, fontweight='bold')

    axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("(a) Synthetic Test Image\n(checkerboard + circles + noise)")
    axes[0].axis('off')

    axes[1].imshow(mag, cmap='hot')
    axes[1].set_title("(b) Log-Magnitude Spectrum |F(u,v)|\n"
                      "(DC at centre; bright = high energy)")
    axes[1].axis('off')

    # Annotate the spectrum image
    cx, cy = np.array(img.shape) // 2
    axes[1].plot(cy, cx, 'c+', markersize=12, markeredgewidth=2,
                 label='DC (u=v=0)')
    axes[1].legend(loc='upper right', fontsize=8)

    # 1-D cross-section through the spectrum
    mid = mag.shape[0] // 2
    axes[2].plot(mag[mid, :], 'C0', linewidth=1.2)
    axes[2].set_title("(c) Horizontal Cross-section of Spectrum\n"
                      "at v=0; symmetric about centre")
    axes[2].set_xlabel("Column index (u)")
    axes[2].set_ylabel("Log magnitude")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("15_spectrum.png", dpi=120)
    print("  Saved: 15_spectrum.png")
    plt.show()


def demo_lowpass_filters(img):
    """Compare ideal and Gaussian lowpass filters in the frequency domain."""
    print("\n" + "=" * 60)
    print("SECTION 2: 2-D Lowpass Filters (frequency domain)")
    print("=" * 60)

    cutoff = 0.08    # normalised frequency (0.5 = Nyquist)
    sigma_g = 0.05

    lp_ideal, H_ideal = ideal_lowpass(img, cutoff=cutoff)
    lp_gauss, H_gauss = gaussian_lowpass(img, sigma_freq=sigma_g)

    print(f"  Ideal LP  cutoff   : {cutoff} (cycles/pixel)")
    print(f"  Gaussian LP sigma  : {sigma_g} (cycles/pixel)")
    print(f"  Ideal LP output range  : [{lp_ideal.min():.3f}, {lp_ideal.max():.3f}]")
    print(f"  Gaussian LP output range: [{lp_gauss.min():.3f}, {lp_gauss.max():.3f}]")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("2-D Lowpass Filters: Ideal vs Gaussian\n"
                 "Top row: filter masks | Bottom row: filtered images",
                 fontsize=12, fontweight='bold')

    # Row 0: filter masks
    axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    _, orig_mag = compute_2d_spectrum(img)
    axes[0, 1].imshow(orig_mag, cmap='hot')
    axes[0, 1].set_title("Original Spectrum")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(H_ideal, cmap='gray')
    axes[0, 2].set_title(f"Ideal LP Mask\n(cutoff={cutoff})")
    axes[0, 2].axis('off')

    axes[0, 3].imshow(H_gauss, cmap='gray')
    axes[0, 3].set_title(f"Gaussian LP Mask\n(σ={sigma_g})")
    axes[0, 3].axis('off')

    # Row 1: filtered results
    lp_ideal_clipped = np.clip(lp_ideal, 0, 1)
    lp_gauss_clipped = np.clip(lp_gauss, 0, 1)

    axes[1, 0].imshow(lp_ideal_clipped, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title("Ideal LP Result\n(note: Gibbs ringing)")
    axes[1, 0].axis('off')

    _, mag_ideal = compute_2d_spectrum(lp_ideal_clipped)
    axes[1, 1].imshow(mag_ideal, cmap='hot')
    axes[1, 1].set_title("Ideal LP Spectrum\n(high freqs removed)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(lp_gauss_clipped, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title("Gaussian LP Result\n(smooth, no ringing)")
    axes[1, 2].axis('off')

    _, mag_gauss = compute_2d_spectrum(lp_gauss_clipped)
    axes[1, 3].imshow(mag_gauss, cmap='hot')
    axes[1, 3].set_title("Gaussian LP Spectrum")
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig("15_lowpass_filters.png", dpi=120)
    print("  Saved: 15_lowpass_filters.png")
    plt.show()

    return lp_ideal, lp_gauss


def demo_highpass_and_edges(img):
    """
    Apply highpass filtering in the frequency domain and compare with
    spatial-domain Sobel edge detection.
    """
    print("\n" + "=" * 60)
    print("SECTION 3: Highpass Filter vs Spatial Sobel Edge Detection")
    print("=" * 60)

    hp_img, H_hp = ideal_highpass(img, cutoff=0.05)
    sobel_img = sobel_edges(img)

    # Normalise for display
    hp_display = np.abs(hp_img)
    hp_display = hp_display / hp_display.max()
    sobel_display = sobel_img / sobel_img.max()

    print(f"  HP filter output  max : {hp_img.max():.4f}")
    print(f"  Sobel edges output max: {sobel_img.max():.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Highpass Filtering (Frequency Domain) vs Sobel Edge Detection (Spatial)",
                 fontsize=12, fontweight='bold')

    axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title("(a) Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(H_hp, cmap='gray')
    axes[0, 1].set_title("(b) Ideal HP Mask\n(zeros at centre = DC blocked)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(hp_display, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title("(c) Ideal HP Output\n(edges + fine detail)")
    axes[0, 2].axis('off')

    # Spectrum of HP output
    _, mag_hp = compute_2d_spectrum(hp_display)
    axes[1, 0].imshow(mag_hp, cmap='hot')
    axes[1, 0].set_title("(d) HP Output Spectrum\n(DC gap visible at centre)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(sobel_display, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title("(e) Sobel Edge Magnitude\n(spatial 3×3 kernel)")
    axes[1, 1].axis('off')

    # Overlay comparison
    axes[1, 2].imshow(img, cmap='gray', vmin=0, vmax=1, alpha=0.6)
    axes[1, 2].imshow(sobel_display, cmap='Reds', alpha=0.5, vmin=0.1, vmax=1)
    axes[1, 2].set_title("(f) Sobel Edges Overlaid on Original")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig("15_highpass_edges.png", dpi=120)
    print("  Saved: 15_highpass_edges.png")
    plt.show()


def demo_freq_vs_spatial_blur(img):
    """
    Demonstrate equivalence of frequency-domain Gaussian LP filter and
    spatial-domain Gaussian blur, and show the effect of increasing blur.
    """
    print("\n" + "=" * 60)
    print("SECTION 4: Frequency Domain vs Spatial Domain Blur Comparison")
    print("=" * 60)

    # Gaussian LP in frequency domain (sigma_freq → sigma_spatial via uncertainty)
    # Relationship: sigma_spatial * sigma_freq ≈ 1/(2*pi)
    sigma_freq = 0.04
    sigma_spatial = 1.0 / (2 * np.pi * sigma_freq)   # theoretical equivalence
    print(f"  Gaussian LP sigma_freq={sigma_freq:.3f}  ↔  "
          f"spatial sigma≈{sigma_spatial:.1f} pixels")

    blurred_freq, _ = gaussian_lowpass(img, sigma_freq=sigma_freq)
    blurred_spatial = gaussian_blur_spatial(img, sigma=sigma_spatial)

    diff = np.abs(blurred_freq - blurred_spatial)
    print(f"  Max pixel difference (freq vs spatial): {diff.max():.6f}")
    print("  (small difference confirms near-equivalence for this image size)")

    # Show progression of spatial blur
    sigmas = [1, 4, 10]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Frequency Domain vs Spatial Domain Gaussian Blur",
                 fontsize=12, fontweight='bold')

    # Top row: spatial blur at increasing sigma
    for ax, sigma in zip(axes[0], sigmas):
        blurred = gaussian_blur_spatial(img, sigma=sigma)
        ax.imshow(np.clip(blurred, 0, 1), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Spatial Blur σ={sigma} px")
        ax.axis('off')

    # Bottom row: their spectra
    for ax, sigma in zip(axes[1], sigmas):
        blurred = gaussian_blur_spatial(img, sigma=sigma)
        _, mag = compute_2d_spectrum(blurred)
        ax.imshow(mag, cmap='hot')
        ax.set_title(f"Spectrum (σ={sigma} px)\n"
                     f"Larger σ → narrower spectrum")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("15_blur_comparison.png", dpi=120)
    print("  Saved: 15_blur_comparison.png")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Image Signal Processing: 2D DFT and Filtering")
    print("=" * 60)
    print("Key relationships:")
    print("  Convolution theorem : f*h  <->  F·H   (spatial ↔ frequency)")
    print("  Large σ spatial      → narrow Gaussian in frequency (smooth image)")
    print("  Ideal LP (brick wall) → sinc ripple in spatial domain (Gibbs)")
    print("  Gaussian LP           → Gaussian in spatial domain (no ringing)")
    print()

    size = 256
    print(f"Creating synthetic {size}×{size} test image...")
    img = make_test_image(size=size)
    print(f"  Image stats: min={img.min():.3f}, max={img.max():.3f}, "
          f"mean={img.mean():.3f}")

    demo_spectrum(img)
    demo_lowpass_filters(img)
    demo_highpass_and_edges(img)
    demo_freq_vs_spatial_blur(img)

    print("\nDone.  Four PNG files saved.")
    print("\nKey takeaways:")
    print("  - 2-D DFT: low freqs at centre (after fftshift), "
          "high freqs at edges")
    print("  - Ideal LP: sharp mask → Gibbs ringing in spatial domain")
    print("  - Gaussian LP: smooth mask → smooth blur, no ringing")
    print("  - Highpass = 1 - Lowpass: emphasises edges and fine detail")
    print("  - Frequency and spatial domain filtering are theoretically "
          "equivalent (convolution theorem)")

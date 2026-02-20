# Signal Processing Examples

This directory contains 15 Python examples covering the fundamentals and applications of signal processing. All examples use NumPy, SciPy, and Matplotlib.

## Requirements

```bash
pip install numpy scipy matplotlib
```

## Files Overview

### 1. `01_signals_classification.py` - Signals and Systems
**Concepts:**
- Basic signal types: sinusoidal, exponential, unit step, unit impulse, rectangular pulse
- Signal classification: continuous vs discrete, periodic vs aperiodic, deterministic vs random
- Even/odd decomposition
- Energy and power computation
- Time operations: shifting, scaling, reversal

**Run:** `python 01_signals_classification.py`

---

### 2. `02_convolution.py` - LTI Systems and Convolution
**Concepts:**
- Discrete convolution implemented from scratch
- Verification against `np.convolve`
- LTI system impulse response and output computation
- Moving average filter as convolution
- Step-by-step convolution visualization
- Convolution properties: commutativity, associativity, distributivity
- RC circuit impulse response

**Run:** `python 02_convolution.py`

---

### 3. `03_fourier_series.py` - Fourier Series
**Concepts:**
- Fourier series coefficients for square, sawtooth, triangular waves
- Signal reconstruction using partial sums
- Gibbs phenomenon visualization
- Magnitude and phase spectra
- Parseval's theorem verification
- Convergence analysis (RMSE vs number of harmonics)

**Run:** `python 03_fourier_series.py`

---

### 4. `04_fourier_transform.py` - Continuous Fourier Transform
**Concepts:**
- Numerical CTFT approximation for rectangular pulse, Gaussian, exponential decay
- Magnitude and phase spectra plotting
- Fourier transform properties: linearity, time shift, frequency shift (modulation)
- Parseval's theorem verification
- Rect-sinc duality
- Energy spectral density

**Run:** `python 04_fourier_transform.py`

---

### 5. `05_sampling_aliasing.py` - Sampling and Reconstruction
**Concepts:**
- Sampling at different rates (oversampling, critical, undersampling)
- Aliasing demonstration when violating Nyquist criterion
- Whittaker-Shannon sinc interpolation for signal reconstruction
- Spectral analysis before and after sampling
- Anti-aliasing filter design and application

**Run:** `python 05_sampling_aliasing.py`

---

### 6. `06_fft_analysis.py` - DFT and FFT
**Concepts:**
- Manual DFT via matrix multiplication vs `np.fft.fft`
- Computation time comparison (DFT O(N²) vs FFT O(N log N))
- Zero-padding for frequency resolution improvement
- Windowing effects: rectangular, Hanning, Hamming, Blackman
- Spectral leakage analysis
- Multi-tone signal detection

**Run:** `python 06_fft_analysis.py`

---

### 7. `07_z_transform.py` - Z-Transform
**Concepts:**
- Common Z-transform pairs verification
- Pole-zero diagram plotting with unit circle
- System stability analysis from pole locations
- Inverse Z-transform via partial fraction expansion
- 2nd-order IIR system analysis
- `scipy.signal.tf2zpk` and `zpk2tf` usage

**Run:** `python 07_z_transform.py`

---

### 8. `08_filter_response.py` - Digital Filter Fundamentals
**Concepts:**
- FIR vs IIR filter comparison
- Magnitude response, phase response, group delay
- Moving average (FIR) and 1st-order recursive (IIR) filters
- Impulse response and step response
- Noise filtering application with SNR measurement
- `scipy.signal.freqz` for frequency response

**Run:** `python 08_filter_response.py`

---

### 9. `09_fir_design.py` - FIR Filter Design
**Concepts:**
- Window method: Hamming, Kaiser windows
- Bandpass filter design
- Optimal design: Parks-McClellan algorithm (`scipy.signal.remez`)
- Window method vs optimal design comparison
- Kaiser window parameter estimation
- Filter application to multi-tone test signal

**Run:** `python 09_fir_design.py`

---

### 10. `10_iir_design.py` - IIR Filter Design
**Concepts:**
- Butterworth, Chebyshev Type I/II, Elliptic filter design
- Bilinear transform method
- Frequency response comparison across filter families
- Filter order vs transition band sharpness tradeoff
- Minimum order comparison using `buttord`, `cheb1ord`, `ellipord`
- Pole-zero diagrams for each filter type

**Run:** `python 10_iir_design.py`

---

### 11. `11_multirate.py` - Multirate Processing
**Concepts:**
- Decimation with and without anti-aliasing filter
- Interpolation with and without interpolation filter
- Aliasing and imaging artifacts
- Rational rate conversion (44100 Hz ↔ 48000 Hz)
- `scipy.signal.decimate` and `resample_poly`
- Spectral analysis at each processing stage

**Run:** `python 11_multirate.py`

---

### 12. `12_spectral_estimation.py` - Spectral Analysis
**Concepts:**
- Periodogram and its variance problem
- Bartlett's method (non-overlapping segments)
- Welch's method (overlapping, windowed segments)
- AR parametric estimation via Yule-Walker equations
- Resolution vs variance tradeoff
- Peak detection for frequency estimation

**Run:** `python 12_spectral_estimation.py`

---

### 13. `13_adaptive_lms.py` - Adaptive Filters
**Concepts:**
- LMS (Least Mean Squares) algorithm implementation
- NLMS (Normalized LMS) algorithm implementation
- System identification application
- Noise cancellation (Widrow noise canceller)
- Learning curve analysis (MSE vs iteration)
- Step size effect on convergence and stability
- LMS vs NLMS comparison under non-stationary input

**Run:** `python 13_adaptive_lms.py`

---

### 14. `14_spectrogram_wavelet.py` - Time-Frequency Analysis
**Concepts:**
- STFT spectrogram computation (`scipy.signal.stft`)
- Chirp signal analysis
- Time-frequency resolution tradeoff (window length effect)
- Continuous Wavelet Transform (CWT) with Morlet wavelet
- Spectrogram vs CWT scalogram comparison
- Multi-component signal analysis (stationary + transient)

**Run:** `python 14_spectrogram_wavelet.py`

---

### 15. `15_image_filtering.py` - Image Signal Processing
**Concepts:**
- Synthetic test image generation
- 2D DFT computation and visualization (log magnitude spectrum)
- Frequency domain filtering: ideal and Gaussian lowpass/highpass
- Spatial domain filtering: Sobel edge detection, Gaussian blur
- Frequency vs spatial domain filtering comparison
- Convolution theorem verification in 2D

**Run:** `python 15_image_filtering.py`

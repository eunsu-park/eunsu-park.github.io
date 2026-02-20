"""
Fourier Analysis - Series, Transforms, and Spectral Analysis

This script demonstrates:
- Fourier series coefficients computation
- Fast Fourier Transform (FFT)
- Spectral analysis of signals
- Filtering in frequency domain
- Parseval's theorem verification
- Windowing techniques
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def fourier_series_coefficients(f, T, n_harmonics):
    """
    Compute Fourier series coefficients for periodic function f with period T
    f(t) = a0/2 + sum(an*cos(nωt) + bn*sin(nωt))
    """
    omega = 2 * np.pi / T
    N = 1000  # Number of integration points

    t = np.linspace(0, T, N)
    dt = T / N

    # a0 (DC component)
    a0 = (2 / T) * np.sum(f(t)) * dt

    # an and bn coefficients
    an = np.zeros(n_harmonics)
    bn = np.zeros(n_harmonics)

    for n in range(1, n_harmonics + 1):
        an[n-1] = (2 / T) * np.sum(f(t) * np.cos(n * omega * t)) * dt
        bn[n-1] = (2 / T) * np.sum(f(t) * np.sin(n * omega * t)) * dt

    return a0, an, bn


def reconstruct_from_fourier(t, T, a0, an, bn):
    """Reconstruct signal from Fourier coefficients"""
    omega = 2 * np.pi / T
    signal = a0 / 2 * np.ones_like(t)

    for n in range(len(an)):
        signal += an[n] * np.cos((n + 1) * omega * t)
        signal += bn[n] * np.sin((n + 1) * omega * t)

    return signal


def parseval_theorem_check(f, T, a0, an, bn):
    """
    Verify Parseval's theorem:
    (1/T) ∫|f(t)|² dt = (a0²/4) + (1/2)∑(an² + bn²)
    """
    # Left side: time domain energy
    N = 1000
    t = np.linspace(0, T, N)
    dt = T / N
    time_energy = (1 / T) * np.sum(np.abs(f(t))**2) * dt

    # Right side: frequency domain energy
    freq_energy = (a0**2) / 4
    for n in range(len(an)):
        freq_energy += 0.5 * (an[n]**2 + bn[n]**2)

    return time_energy, freq_energy


def apply_window(signal, window_type='hanning'):
    """Apply window function to signal"""
    N = len(signal)

    if window_type == 'hanning':
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / N))
    elif window_type == 'hamming':
        window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / N)
    elif window_type == 'blackman':
        n = np.arange(N)
        window = 0.42 - 0.5 * np.cos(2*np.pi*n/N) + 0.08 * np.cos(4*np.pi*n/N)
    elif window_type == 'rectangular':
        window = np.ones(N)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

    return signal * window, window


def lowpass_filter(signal, cutoff_freq, sampling_freq):
    """Apply ideal lowpass filter in frequency domain"""
    N = len(signal)
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/sampling_freq)

    # Create filter
    filter_mask = np.abs(freqs) <= cutoff_freq
    fft_filtered = fft_signal * filter_mask

    # Inverse FFT
    filtered_signal = np.fft.ifft(fft_filtered).real

    return filtered_signal, freqs, filter_mask


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

print("=" * 70)
print("FOURIER ANALYSIS - SERIES, TRANSFORMS, AND SPECTRAL ANALYSIS")
print("=" * 70)

# Test 1: Fourier series for square wave
print("\n1. FOURIER SERIES - SQUARE WAVE")
print("-" * 70)
T = 2 * np.pi
square_wave = lambda t: np.where(np.mod(t, T) < T/2, 1, -1)

a0, an, bn = fourier_series_coefficients(square_wave, T, n_harmonics=10)
print(f"Period T = {T:.4f}")
print(f"DC component a0 = {a0:.6f}")
print(f"\nFirst 5 Fourier coefficients:")
for n in range(5):
    print(f"  a{n+1} = {an[n]:8.6f}, b{n+1} = {bn[n]:8.6f}")

# Analytical result for square wave: bn = 4/(nπ) for odd n
print(f"\nAnalytical (odd n): b_n = 4/(nπ)")
for n in [1, 3, 5]:
    analytical = 4 / (n * np.pi)
    print(f"  b{n} analytical = {analytical:.6f}")

# Test 2: Fourier series for sawtooth wave
print("\n2. FOURIER SERIES - SAWTOOTH WAVE")
print("-" * 70)
T = 2 * np.pi
sawtooth = lambda t: (np.mod(t, T) / T) * 2 - 1

a0, an, bn = fourier_series_coefficients(sawtooth, T, n_harmonics=10)
print(f"DC component a0 = {a0:.6f}")
print(f"\nFirst 5 Fourier coefficients:")
for n in range(5):
    print(f"  a{n+1} = {an[n]:8.6f}, b{n+1} = {bn[n]:8.6f}")

# Analytical: bn = -2/(nπ) for all n
print(f"\nAnalytical: b_n = -2/(nπ)")
for n in [1, 2, 3]:
    analytical = -2 / (n * np.pi)
    print(f"  b{n} analytical = {analytical:.6f}")

# Test 3: FFT of composite signal
print("\n3. FAST FOURIER TRANSFORM (FFT)")
print("-" * 70)
# Create composite signal: 5Hz + 15Hz + 25Hz
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*15*t) + 0.3*np.sin(2*np.pi*25*t)

# Add noise
np.random.seed(42)
signal_noisy = signal + 0.1 * np.random.randn(len(signal))

# Compute FFT
fft_result = np.fft.fft(signal_noisy)
freqs = np.fft.fftfreq(len(signal_noisy), 1/fs)
magnitude = np.abs(fft_result) / len(signal_noisy)

# Find peaks
positive_freqs = freqs[:len(freqs)//2]
positive_magnitude = magnitude[:len(magnitude)//2]
peak_indices = np.where(positive_magnitude > 0.1)[0]
peak_freqs = positive_freqs[peak_indices]

print(f"Sampling frequency: {fs} Hz")
print(f"Signal components: 5Hz, 15Hz, 25Hz")
print(f"\nDetected peaks in FFT:")
for freq in sorted(peak_freqs):
    if freq > 0:
        idx = np.argmin(np.abs(positive_freqs - freq))
        mag = positive_magnitude[idx]
        print(f"  {freq:.1f} Hz (magnitude: {mag:.3f})")

# Test 4: Parseval's theorem
print("\n4. PARSEVAL'S THEOREM")
print("-" * 70)
T = 2 * np.pi
square_wave = lambda t: np.where(np.mod(t, T) < T/2, 1, -1)
a0, an, bn = fourier_series_coefficients(square_wave, T, n_harmonics=20)

time_energy, freq_energy = parseval_theorem_check(square_wave, T, a0, an, bn)
print(f"Energy in time domain: {time_energy:.6f}")
print(f"Energy in frequency domain: {freq_energy:.6f}")
print(f"Relative difference: {abs(time_energy - freq_energy)/time_energy * 100:.2f}%")

# Test 5: Windowing
print("\n5. WINDOWING EFFECTS")
print("-" * 70)
# Signal: single frequency
fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2*np.pi*10*t)

window_types = ['rectangular', 'hanning', 'hamming', 'blackman']
print(f"Signal: 10 Hz sine wave")
print(f"\nSpectral leakage (sidelobe levels):")

for window_type in window_types:
    windowed_signal, window = apply_window(signal, window_type)
    fft_windowed = np.fft.fft(windowed_signal)
    magnitude = np.abs(fft_windowed) / len(windowed_signal)

    # Find main lobe and first sidelobe
    main_lobe = np.max(magnitude)
    # Look away from main peak
    main_peak_idx = np.argmax(magnitude[:len(magnitude)//2])
    sidelobe_region = magnitude[main_peak_idx+20:len(magnitude)//2]
    if len(sidelobe_region) > 0:
        sidelobe = np.max(sidelobe_region)
        sidelobe_db = 20 * np.log10(sidelobe / main_lobe)
        print(f"  {window_type:12s}: {sidelobe_db:6.1f} dB")

# Test 6: Filtering
print("\n6. LOWPASS FILTERING")
print("-" * 70)
# Signal with multiple frequencies
fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)
signal = (np.sin(2*np.pi*5*t) +
          np.sin(2*np.pi*50*t) +
          np.sin(2*np.pi*120*t))

print(f"Original signal: 5Hz + 50Hz + 120Hz")

# Apply lowpass filter at 30Hz
cutoff = 30
filtered_signal, freqs, filter_mask = lowpass_filter(signal, cutoff, fs)

# Check frequency content
fft_original = np.fft.fft(signal)
fft_filtered = np.fft.fft(filtered_signal)

print(f"Lowpass filter cutoff: {cutoff} Hz")
print(f"\nFrequency content after filtering:")

for test_freq in [5, 50, 120]:
    idx = np.argmin(np.abs(freqs[:len(freqs)//2] - test_freq))
    original_mag = np.abs(fft_original[idx])
    filtered_mag = np.abs(fft_filtered[idx])
    attenuation = 20 * np.log10(filtered_mag / original_mag) if filtered_mag > 1e-6 else -100
    print(f"  {test_freq:3d} Hz: {attenuation:6.1f} dB")

# Test 7: Gibbs phenomenon
print("\n7. GIBBS PHENOMENON")
print("-" * 70)
T = 2 * np.pi
square_wave = lambda t: np.where(np.mod(t, T) < T/2, 1, -1)

print("Square wave reconstruction with different harmonics:")
for n_harm in [5, 10, 50]:
    a0, an, bn = fourier_series_coefficients(square_wave, T, n_harmonics=n_harm)

    # Reconstruct at discontinuity
    t_disc = np.array([T/2 - 0.01, T/2, T/2 + 0.01])
    reconstructed = reconstruct_from_fourier(t_disc, T, a0, an, bn)

    overshoot = (reconstructed[0] - 1) / 1 * 100  # Before discontinuity
    print(f"  n={n_harm:3d} harmonics: overshoot ≈ {abs(overshoot):.1f}%")

print(f"\nGibbs overshoot approaches ~9% as n→∞")

# Visualization
if HAS_MATPLOTLIB:
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    fig = plt.figure(figsize=(15, 12))

    # Plot 1: Square wave Fourier series
    ax1 = plt.subplot(3, 3, 1)
    T = 2 * np.pi
    t_plot = np.linspace(0, 2*T, 1000)
    square_wave = lambda t: np.where(np.mod(t, T) < T/2, 1, -1)

    ax1.plot(t_plot, square_wave(t_plot), 'k-', linewidth=2, label='Original', alpha=0.3)

    for n_harm in [1, 3, 10]:
        a0, an, bn = fourier_series_coefficients(square_wave, T, n_harmonics=n_harm)
        reconstructed = reconstruct_from_fourier(t_plot, T, a0, an, bn)
        ax1.plot(t_plot, reconstructed, label=f'n={n_harm}', linewidth=1.5)

    ax1.set_xlabel('t')
    ax1.set_ylabel('f(t)')
    ax1.set_title('Fourier Series: Square Wave')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.5, 1.5)

    # Plot 2: Sawtooth wave
    ax2 = plt.subplot(3, 3, 2)
    sawtooth = lambda t: (np.mod(t, T) / T) * 2 - 1

    ax2.plot(t_plot, sawtooth(t_plot), 'k-', linewidth=2, label='Original', alpha=0.3)

    for n_harm in [1, 5, 20]:
        a0, an, bn = fourier_series_coefficients(sawtooth, T, n_harmonics=n_harm)
        reconstructed = reconstruct_from_fourier(t_plot, T, a0, an, bn)
        ax2.plot(t_plot, reconstructed, label=f'n={n_harm}', linewidth=1.5)

    ax2.set_xlabel('t')
    ax2.set_ylabel('f(t)')
    ax2.set_title('Fourier Series: Sawtooth Wave')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: FFT spectrum
    ax3 = plt.subplot(3, 3, 3)
    fs = 1000
    t = np.linspace(0, 1, fs, endpoint=False)
    signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*15*t) + 0.3*np.sin(2*np.pi*25*t)
    signal_noisy = signal + 0.1 * np.random.randn(len(signal))

    fft_result = np.fft.fft(signal_noisy)
    freqs = np.fft.fftfreq(len(signal_noisy), 1/fs)
    magnitude = np.abs(fft_result) / len(signal_noisy)

    ax3.plot(freqs[:len(freqs)//2], magnitude[:len(magnitude)//2], 'b-', linewidth=1.5)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('FFT Spectrum: 5Hz + 15Hz + 25Hz')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 50)

    # Plot 4: Window functions
    ax4 = plt.subplot(3, 3, 4)
    N = 256
    signal_unit = np.ones(N)

    for window_type in ['rectangular', 'hanning', 'hamming', 'blackman']:
        _, window = apply_window(signal_unit, window_type)
        ax4.plot(window, label=window_type, linewidth=1.5)

    ax4.set_xlabel('Sample')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Window Functions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Windowing effect on spectrum
    ax5 = plt.subplot(3, 3, 5)
    fs = 1000
    t = np.linspace(0, 1, fs, endpoint=False)
    signal = np.sin(2*np.pi*10*t)

    for window_type in ['rectangular', 'hanning', 'blackman']:
        windowed_signal, _ = apply_window(signal, window_type)
        fft_windowed = np.fft.fft(windowed_signal)
        freqs = np.fft.fftfreq(len(windowed_signal), 1/fs)
        magnitude_db = 20 * np.log10(np.abs(fft_windowed) / len(windowed_signal) + 1e-10)

        ax5.plot(freqs[:len(freqs)//2], magnitude_db[:len(magnitude_db)//2],
                label=window_type, linewidth=1.5, alpha=0.7)

    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Magnitude (dB)')
    ax5.set_title('Spectral Leakage with Different Windows')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 50)
    ax5.set_ylim(-80, 10)

    # Plot 6: Filtering demonstration
    ax6 = plt.subplot(3, 3, 6)
    fs = 1000
    t = np.linspace(0, 0.2, 200)
    signal = (np.sin(2*np.pi*5*t) +
              np.sin(2*np.pi*50*t) +
              np.sin(2*np.pi*120*t))

    filtered_signal, _, _ = lowpass_filter(signal, 30, fs)

    ax6.plot(t, signal, 'b-', linewidth=1.5, alpha=0.5, label='Original')
    ax6.plot(t, filtered_signal, 'r-', linewidth=2, label='Filtered (30Hz)')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Amplitude')
    ax6.set_title('Lowpass Filtering')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Plot 7: Frequency spectrum before/after filter
    ax7 = plt.subplot(3, 3, 7)
    t_full = np.linspace(0, 1, fs, endpoint=False)
    signal_full = (np.sin(2*np.pi*5*t_full) +
                   np.sin(2*np.pi*50*t_full) +
                   np.sin(2*np.pi*120*t_full))
    filtered_full, freqs, _ = lowpass_filter(signal_full, 30, fs)

    fft_orig = np.fft.fft(signal_full)
    fft_filt = np.fft.fft(filtered_full)
    mag_orig = np.abs(fft_orig) / len(signal_full)
    mag_filt = np.abs(fft_filt) / len(filtered_full)

    ax7.plot(freqs[:len(freqs)//2], mag_orig[:len(mag_orig)//2],
            'b-', linewidth=1.5, alpha=0.5, label='Original')
    ax7.plot(freqs[:len(freqs)//2], mag_filt[:len(mag_filt)//2],
            'r-', linewidth=2, label='Filtered')
    ax7.axvline(x=30, color='k', linestyle='--', alpha=0.5, label='Cutoff')
    ax7.set_xlabel('Frequency (Hz)')
    ax7.set_ylabel('Magnitude')
    ax7.set_title('Frequency Domain: Before/After Filter')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, 150)

    # Plot 8: Gibbs phenomenon
    ax8 = plt.subplot(3, 3, 8)
    T = 2 * np.pi
    t_gibbs = np.linspace(T/2 - 0.5, T/2 + 0.5, 500)
    square_wave = lambda t: np.where(np.mod(t, T) < T/2, 1, -1)

    ax8.plot(t_gibbs, square_wave(t_gibbs), 'k-', linewidth=3, label='Original', alpha=0.3)

    for n_harm in [5, 10, 50]:
        a0, an, bn = fourier_series_coefficients(square_wave, T, n_harmonics=n_harm)
        reconstructed = reconstruct_from_fourier(t_gibbs, T, a0, an, bn)
        ax8.plot(t_gibbs, reconstructed, label=f'n={n_harm}', linewidth=1.5)

    ax8.axhline(y=1.09, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax8.set_xlabel('t')
    ax8.set_ylabel('f(t)')
    ax8.set_title('Gibbs Phenomenon (9% overshoot)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(-1.3, 1.3)

    # Plot 9: Parseval's theorem
    ax9 = plt.subplot(3, 3, 9)
    harmonics = range(1, 51)
    time_energies = []
    freq_energies = []

    for n_harm in harmonics:
        a0, an, bn = fourier_series_coefficients(square_wave, T, n_harmonics=n_harm)
        time_e, freq_e = parseval_theorem_check(square_wave, T, a0, an, bn)
        time_energies.append(time_e)
        freq_energies.append(freq_e)

    ax9.plot(harmonics, time_energies, 'b-', linewidth=2, label='Time domain')
    ax9.plot(harmonics, freq_energies, 'r--', linewidth=2, label='Frequency domain')
    ax9.set_xlabel('Number of harmonics')
    ax9.set_ylabel('Energy')
    ax9.set_title("Parseval's Theorem Verification")
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Mathematical_Methods/06_fourier.png', dpi=150)
    print("Saved visualization: 06_fourier.png")
    plt.close()

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)

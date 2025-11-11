# generate_flux_models_final.py
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

# ==============================================================================
# === Reference Model Data & Coefficients ===
# ==============================================================================

# CygA Model (Baars et al. 1977, A&A, 61, 99)
CYGA_FREQS_BAARS = np.array([10.05, 12.6, 14.7, 16.7, 20.0, 22.25, 25.0, 26.3, 38.0, 81.5, 152.0, 320.0, 550.0, 625.0, 710.0, 800.0, 1117., 1304., 1415., 2000., 2290., 2740., 3380., 3960., 5680., 6660., 10700., 13490., 22285., 34900.])
CYGA_FLUX_BAARS = np.array([13500, 21900, 31700, 26600, 27000, 29100, 31500, 29600, 25500, 16300, 10500, 5870, 4140, 3400, 3100, 2670, 1900, 1690, 1500, 1000, 935, 710, 615, 515, 317, 265, 148, 102, 60.2, 36])
CYGA_ERR_PERCENT = np.array([11, 14, 14, 14, 14, 6, 14, 5.2, 4.2, 4.2, 4.2, 6, 4, 4, 4, 4, 3, 3, 3, 6, 6, 6, 6, 6, 6, 3.8, 4.5, 3.8, 3.8, 20])

# Definitive Cas A data for Epoch 2023.5
CASA_FREQS_2023_MHZ = np.array([
    7.99685322, 8.9687719, 9.97791914, 11.01200996, 12.00997509, 12.99452444, 14.00388081, 15.03186164,
    16.00870855, 17.04864632, 18.01382656, 18.95869005, 20.11044065, 21.08192238, 22.10016544, 23.07652558,
    24.0960202, 25.06165791, 26.06599314, 27.11057667, 28.08640313, 29.09713212, 30.02620504, 31.10674056,
    32.09998037, 33.12468205, 34.18235484, 35.13515066, 35.97282563, 37.12115701, 38.15615939, 39.06585976,
    39.99724876, 41.11244158, 42.09262417, 43.09617579, 44.12398972, 46.98566408, 46.07193346, 45.17631631,
    47.91751645, 48.86747769, 49.83627185, 50.82465945, 51.83264944, 52.86063058, 54.12131999, 54.97816013,
    55.84814018, 57.40528341, 58.77391192, 60.17517059, 62.0956208, 63.57607402, 65.09182348, 66.64371069,
    69.8598974, 73.51915686, 78.90454561, 87.04557109
])
CASA_FLUX_2023_JY = np.array([
    5136.489246, 8645.572948, 12439.80605, 16002.48400, 19573.95795, 22261.41164, 24344.55298, 25887.57351,
    27222.20425, 28148.62322, 28944.50463, 29431.58117, 29759.15478, 29923.09943, 29919.88142, 29749.82131,
    29580.72779, 29248.56408, 28920.13024, 28595.38441, 28274.53858, 27801.12831, 27489.44055, 27029.17545,
    26726.14232, 26278.89290, 25984.27144, 25549.66586, 25122.55454, 24702.14040, 24425.41535, 24017.09801,
    23615.60647, 23351.05328, 22960.69595, 22576.86419, 22324.14756, 21462.67107, 21705.44097, 22074.25974,
    21222.61649, 20868.02706, 20519.36214, 20289.85823, 20062.92126, 19838.52253, 19616.45783, 19397.22652,
    19073.30682, 18859.63862, 18544.36384, 18234.35948, 17829.06669, 17531.01988, 17237.95549, 16949.79024,
    16479.88454, 15933.36181, 15232.66278, 14399.63915
])
CASA_ERR_2023_PERCENT = np.full_like(CASA_FREQS_2023_MHZ, 10.0)

# --- NEW: Virgo A Polynomial Coefficients ---
VIRA_COEFFS = {'a0': 2.4466, 'a1': -0.8116, 'a2': -0.0483}

# ==============================================================================
# === Telescope & Model Configuration ===
# ==============================================================================
N_CHANNELS = 3072
CHANNEL_WIDTH_HZ = 23926.0
START_FREQ_HZ = 13398000.0
REFERENCE_EPOCH_CASA = 2023.5
REFERENCE_EPOCH_TAUA = 1985.5

# ==============================================================================
# === Model Fitting & Calculation Logic ===
# ==============================================================================
def fit_polynomial_model(ref_freqs_mhz, ref_flux_jy, ref_err_percent, order):
    """Fits a polynomial in log-log space and returns the model."""
    log_freq_mhz = np.log10(ref_freqs_mhz)
    log_flux = np.log10(ref_flux_jy)
    sigma_log = ref_err_percent / (100.0 * np.log(10))
    weights = 1.0 / (sigma_log**2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        poly_model = np.poly1d(np.polyfit(log_freq_mhz, log_flux, order, w=weights))
    return poly_model

def get_flux_from_poly_fit(poly_model, freqs_hz):
    """Evaluates a polynomial model (from polyfit) at the given frequencies."""
    log_freq_mhz = np.log10(np.atleast_1d(freqs_hz) / 1e6)
    log_s_jy = poly_model(log_freq_mhz)
    return 10**log_s_jy

def get_flux_from_log_poly(freqs_hz, coeffs):
    """Calculates flux from a log-polynomial with frequency in GHz."""
    freqs_ghz = freqs_hz / 1e9
    log_freq_ghz = np.log10(freqs_ghz)
    log_s_jy = coeffs['a0'] + (coeffs['a1'] * log_freq_ghz) + (coeffs['a2'] * (log_freq_ghz**2))
    return 10**log_s_jy

# ==============================================================================
def plot_model(source_name, freqs_hz, flux_jy, ref_freqs_mhz=None, ref_flux_jy=None, ref_err_percent=None, epoch=None):
    """Generates and saves a diagnostic plot for a given flux model."""
    plt.figure(figsize=(10, 8))
    epoch_str = f' (Epoch {epoch:.1f})' if epoch else ''
    plt.plot(freqs_hz / 1e6, flux_jy, label=f'High-Resolution Model{epoch_str}', color='black', zorder=2)
    if ref_freqs_mhz is not None and ref_flux_jy is not None:
        sort_idx = np.argsort(ref_freqs_mhz)
        ref_freqs_mhz_sorted, ref_flux_jy_sorted = ref_freqs_mhz[sort_idx], ref_flux_jy[sort_idx]
        errors = (ref_err_percent[sort_idx] / 100.0) * ref_flux_jy_sorted if ref_err_percent is not None else 0
        plt.errorbar(ref_freqs_mhz_sorted, ref_flux_jy_sorted, yerr=errors, fmt='o', label=f'Reference Data{epoch_str}', color='red', zorder=3, alpha=0.7)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10, 100)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Flux Density (Jy)')
    plt.title(f'Intrinsic Flux Density Model for {source_name}')
    plt.legend()
    
    output_filename = f'{source_name}_flux_model.png'
    plt.savefig(output_filename)
    print(f"-> Saved diagnostic plot to '{output_filename}'")
    plt.close()

def generate_models():
    """Generates and saves the intrinsic flux models and plots."""
    print("\n--- Generating Reference Intrinsic Flux Models ---")
    
    frequencies_hz = START_FREQ_HZ + np.arange(N_CHANNELS) * CHANNEL_WIDTH_HZ
    freqs_mhz = frequencies_hz / 1e6
    print(f"Frequency range: {freqs_mhz[0]:.3f} MHz to {freqs_mhz[-1]:.3f} MHz")

    print("\nProcessing Cygnus A...")
    cyga_model = fit_polynomial_model(CYGA_FREQS_BAARS, CYGA_FLUX_BAARS, CYGA_ERR_PERCENT, 8)
    cyga_flux_jy = get_flux_from_poly_fit(cyga_model, frequencies_hz)
    plot_model('CygA', frequencies_hz, cyga_flux_jy, CYGA_FREQS_BAARS, CYGA_FLUX_BAARS, CYGA_ERR_PERCENT)

    print("\nProcessing Cassiopeia A...")
    casa_model = fit_polynomial_model(CASA_FREQS_2023_MHZ, CASA_FLUX_2023_JY, CASA_ERR_2023_PERCENT, 10)
    casa_flux_jy_ref = get_flux_from_poly_fit(casa_model, frequencies_hz)
    plot_model('CasA', frequencies_hz, casa_flux_jy_ref, CASA_FREQS_2023_MHZ, CASA_FLUX_2023_JY, CASA_ERR_2023_PERCENT, epoch=REFERENCE_EPOCH_CASA)

    # --- ADJUSTED: Using the log-polynomial model for Virgo A ---
    print("\nProcessing Virgo A...")
    vira_flux_jy = get_flux_from_log_poly(frequencies_hz, VIRA_COEFFS)
    plot_model('VirA', frequencies_hz, vira_flux_jy)
    
    # CORRECTED: Save reference flux at 1985.5 without secular correction
    print("\nProcessing Taurus A (Crab Nebula)...")
    taua_flux_jy_ref = 970.0 * (freqs_mhz / 1000.0)**(-0.30)
    plot_model('TauA', frequencies_hz, taua_flux_jy_ref, epoch=REFERENCE_EPOCH_TAUA)

    output_filename = 'primary_calibrator_flux_models.npz'
    np.savez_compressed(
        output_filename,
        frequencies_hz=frequencies_hz,
        cyga_flux_jy=cyga_flux_jy,
        vira_flux_jy=vira_flux_jy,
        casa_flux_jy_ref=casa_flux_jy_ref,
        taua_flux_jy_ref=taua_flux_jy_ref,
        reference_epoch=np.array([REFERENCE_EPOCH_CASA]),
        taua_reference_epoch=np.array([REFERENCE_EPOCH_TAUA])
    )
    print(f"\n-> Successfully saved all models to '{output_filename}'")

if __name__ == "__main__":
    generate_models()

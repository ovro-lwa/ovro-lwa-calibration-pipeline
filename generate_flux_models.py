# generate_flux_models.py
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

# ==============================================================================
# === Original Reference Model Data ===
# ==============================================================================

# CygA Model (Baars et al. 1977, A&A, 61, 99)
CYGA_FREQS_BAARS = np.array([10.05, 12.6, 14.7, 16.7, 20.0, 22.25, 25.0, 26.3, 38.0, 81.5, 152.0, 320.0, 550.0, 625.0, 710.0, 800.0, 1117., 1304., 1415., 2000., 2290., 2740., 3380., 3960., 5680., 6660., 10700., 13490., 22285., 34900.])
CYGA_FLUX_BAARS = np.array([13500, 21900, 31700, 26600, 27000, 29100, 31500, 29600, 25500, 16300, 10500, 5870, 4140, 3400, 3100, 2670, 1900, 1690, 1500, 1000, 935, 710, 615, 515, 317, 265, 148, 102, 60.2, 36])
CYGA_ERR_PERCENT = np.array([11, 14, 14, 14, 14, 6, 14, 5.2, 4.2, 4.2, 4.2, 6, 4, 4, 4, 4, 3, 3, 3, 6, 6, 6, 6, 6, 6, 3.8, 4.5, 3.8, 3.8, 20])

# Cas A data from CasA_Stanislav_2023.5.txt
CASA_FREQS_2023_MHZ = np.array([8.001307527, 9.001135841548793, 10.014082918876746, 11.01895069750386, 11.99243878, 13.015309723339062, 14.009759759938195, 14.997998343803246, 16.01174501528519, 17.001089021005935, 18.051302252895553, 19.062109223842608, 20.074488694434926, 21.118231138492828, 22.271062891701007, 23.527899044083483, 24.85554734, 26.294797739337344, 27.72004826689174, 29.30433016185675, 30.84928621588123, 32.47547426617627, 34.187385128870126, 35.98955112293226, 37.88646645632371, 39.88336024860401, 41.98547232356894, 44.19837603211344, 46.52774487309058, 48.979965868115755, 51.56126967404626, 54.27871102826093, 57.13938503760226, 60.15073677508702, 63.32081207, 66.03863156140054, 70.19708868480869, 76.79265824178734])
CASA_FLUX_2023_JY = np.array([5097.7122310115965, 8648.844717815098, 12366.268160561069, 15981.50078110863, 19483.766851967703, 22147.78454088155, 24122.217738679767, 25867.45064556309, 26994.18065146048, 28060.983217455072, 28831.60751857888, 29394.232407881773, 29620.418885690797, 29823.760930112374, 29733.379816210778, 29519.774107174224, 29197.88955502757, 28744.829003672134, 28284.852943155267, 27625.985176743357, 27083.577294653256, 26407.233492203704, 25747.779664595244, 25112.410708472355, 24362.303560255234, 23633.168230557858, 22911.48684706928, 22210.495755959884, 21467.90750735395, 20780.38287, 20064.498021258056, 19402.009439646317, 18765.35206993919, 18127.786143209152, 17516.28407051788, 17082.51191946973, 16293.17391201305, 15547.263079574572])
CASA_ERR_2023_PERCENT = np.full_like(CASA_FREQS_2023_MHZ, 10.0)

# ==============================================================================
# === Telescope & Model Configuration ===
# ==============================================================================
N_CHANNELS = 3072
CHANNEL_WIDTH_HZ = 23926.0
START_FREQ_HZ = 13398000.0
REFERENCE_EPOCH = 2025.0

# ==============================================================================
# === NEW: Self-Contained Model Fitting Logic ===
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

def get_flux_from_model(poly_model, freqs_hz):
    """Evaluates a polynomial model at the given frequencies."""
    log_freq_mhz = np.log10(np.atleast_1d(freqs_hz) / 1e6)
    log_s_jy = poly_model(log_freq_mhz)
    return 10**log_s_jy

# ==============================================================================

def plot_model(source_name, freqs_hz, flux_jy, ref_freqs_mhz=None, ref_flux_jy=None, ref_err_percent=None):
    """Generates and saves a diagnostic plot for a given flux model."""
    plt.figure(figsize=(10, 8))
    plt.plot(freqs_hz / 1e6, flux_jy, label=f'High-Resolution Model (Epoch {REFERENCE_EPOCH:.1f})', color='black', zorder=2)
    if ref_freqs_mhz is not None and ref_flux_jy is not None:
        errors = (ref_err_percent / 100.0) * ref_flux_jy if ref_err_percent is not None else 0
        plt.errorbar(ref_freqs_mhz, ref_flux_jy, yerr=errors, fmt='o', label='Original Reference Data', color='red', zorder=3)

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
    print(f"?? Saved diagnostic plot to '{output_filename}'")
    plt.close()

def generate_models():
    """Generates and saves the intrinsic flux models and plots."""
    print("\n--- Generating Reference Intrinsic Flux Models ---")
    
    # Define the high-resolution frequency array for the instrument
    frequencies_hz = START_FREQ_HZ + np.arange(N_CHANNELS) * CHANNEL_WIDTH_HZ
    print(f"Frequency range: {frequencies_hz[0]/1e6:.3f} MHz to {frequencies_hz[-1]/1e6:.3f} MHz")

    # --- Fit and Plot Models ---
    print("\nProcessing Cygnus A...")
    cyga_model = fit_polynomial_model(CYGA_FREQS_BAARS, CYGA_FLUX_BAARS, CYGA_ERR_PERCENT, 8)
    cyga_flux_jy = get_flux_from_model(cyga_model, frequencies_hz)
    plot_model('CygA', frequencies_hz, cyga_flux_jy, CYGA_FREQS_BAARS, CYGA_FLUX_BAARS, CYGA_ERR_PERCENT)

    print("\nProcessing Cassiopeia A...")
    casa_model = fit_polynomial_model(CASA_FREQS_2023_MHZ, CASA_FLUX_2023_JY, CASA_ERR_2023_PERCENT, 10)
    casa_flux_jy_ref = get_flux_from_model(casa_model, frequencies_hz)
    plot_model('CasA', frequencies_hz, casa_flux_jy_ref, CASA_FREQS_2023_MHZ, CASA_FLUX_2023_JY, CASA_ERR_2023_PERCENT)

    # For VirA and TauA, we still use the direct formulas as they are simple power laws
    print("\nProcessing Virgo A...")
    log_s_jy = 5.024 + (-0.856 * np.log10(frequencies_hz/1e6))
    vira_flux_jy = 10**log_s_jy
    plot_model('VirA', frequencies_hz, vira_flux_jy)
    
    print("\nProcessing Taurus A (Crab Nebula)...")
    s_ref_nu = 970.0 * ((frequencies_hz/1e6) / 1000.0)**(-0.30)
    taua_flux_jy_ref = s_ref_nu * (1 + (-0.16) / 100.0)**(REFERENCE_EPOCH - 1985.5)
    plot_model('TauA', frequencies_hz, taua_flux_jy_ref)

    # --- Save all arrays to a single .npz file ---
    output_filename = 'primary_calibrator_flux_models.npz'
    np.savez_compressed(
        output_filename,
        frequencies_hz=frequencies_hz,
        cyga_flux_jy=cyga_flux_jy,
        vira_flux_jy=vira_flux_jy,
        casa_flux_jy_ref=casa_flux_jy_ref,
        taua_flux_jy_ref=taua_flux_jy_ref,
        reference_epoch=np.array([REFERENCE_EPOCH]),
        taua_reference_epoch=np.array([1985.5])
    )
    print(f"\n?? Successfully saved all models to '{output_filename}'")

if __name__ == "__main__":
    generate_models()

# qa.py
import os
import sys
import glob
import logging
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz
from astropy.wcs import WCS
# --- MODIFIED: Added new imports ---
from astropy.visualization import (
    PercentileInterval, ImageNormalize, LogStretch, AsinhStretch,
    make_lupton_rgb, LinearStretch
)
# --- END MODIFICATION ---
import astropy.units as u
import warnings


# ReportLab imports for PDF generation
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    _reportlab_available = True
except ImportError:
    _reportlab_available = False

# ---- ReportLab md5 compatibility (Python<3.9 / OpenSSL quirks) ----
_REPORTLAB_MD5_PATCHED = False

def _patch_reportlab_md5(logger):
    global _REPORTLAB_MD5_PATCHED
    if _REPORTLAB_MD5_PATCHED or not _reportlab_available:
        return

    try:
        import hashlib
        from reportlab.pdfbase import pdfdoc
        from reportlab.lib import utils as rl_utils

        def _safe_md5(*args, **kwargs):
            kwargs.pop("usedforsecurity", None)
            return hashlib.md5(*args, **kwargs)

        # Patch only if the called function rejects usedforsecurity=
        try:
            pdfdoc.md5(usedforsecurity=False)
        except TypeError:
            logger.info("Patching reportlab.pdfbase.pdfdoc.md5 to ignore 'usedforsecurity'.")
            pdfdoc.md5 = _safe_md5

        try:
            rl_utils.md5(b"test", usedforsecurity=False)
        except TypeError:
            logger.info("Patching reportlab.lib.utils.md5 to ignore 'usedforsecurity'.")
            rl_utils.md5 = _safe_md5

        _REPORTLAB_MD5_PATCHED = True

    except ImportError as e:
        logger.warning(f"ReportLab present but patch imports failed: {e}")
    except Exception as e:
        logger.warning(f"Could not apply ReportLab md5 patches (PDFs may still fail): {e}", exc_info=True)

# Matplotlib configuration
try:
    import matplotlib
    # Set non-interactive backend
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import ScalarFormatter
    import matplotlib.dates as mdates
    _plotting_available = True # Re-set this flag
except ImportError:
    _plotting_available = False # Keep this from pipeline_utils
    pass # Handled by pipeline_utils._plotting_available

# Import pipeline utilities and config
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import pipeline_utils
import pipeline_config as config

# CASA Imports (checked via pipeline_utils)
CASA_TASKS_AVAILABLE = pipeline_utils.CASA_TASKS_AVAILABLE
CASACORE_TABLES_AVAILABLE = pipeline_utils.CASACORE_TABLES_AVAILABLE

# Initialize logger
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = pipeline_utils.get_logger('QA')
_patch_reportlab_md5(logger)


# Attempt to import casacore.tables specifically for QA table reading
if CASACORE_TABLES_AVAILABLE:
    try:
        import casacore.tables
    except ImportError:
        # Fallback if pipeline_utils detected pyrap.tables instead
        logger.warning("casacore.tables not found, attempting fallback for table access.")
        if pipeline_utils.CASA_IMPORTS.get('table_tools_module'):
             # Create a namespace alias for compatibility if needed
             class CompatibilityTables:
                 table = pipeline_utils.CASA_IMPORTS['table_tools_module'].table
             casacore = CompatibilityTables()
        else:
             CASACORE_TABLES_AVAILABLE = False


# Global storage for loaded flux models
FLUX_MODELS = None

# ==============================================================================
# === Flux Model Loading and Calculation (Centralized) ===
# ==============================================================================

def load_flux_models(model_path=config.FLUX_MODEL_NPZ_PATH):
    """Loads the static NPZ flux models into the global FLUX_MODELS variable."""
    global FLUX_MODELS
    # Check if already loaded
    if FLUX_MODELS is not None:
        return True

    if not model_path or not os.path.exists(model_path):
        logger.error(f"Flux model NPZ file not found or path not configured: {model_path}")
        return False

    logger.info(f"Loading reference flux models from {model_path}")
    try:
        # Load the NPZ file containing numpy arrays
        FLUX_MODELS = np.load(model_path, allow_pickle=True)
        # Check for essential keys
        if 'frequencies_hz' not in FLUX_MODELS:
            raise KeyError("Key 'frequencies_hz' missing in the NPZ file.")
        return True
    except Exception as e:
        logger.error(f"Failed to load flux models from NPZ: {e}", exc_info=True)
        FLUX_MODELS = None
        return False

def get_expected_flux(source, freqs_hz, obs_epoch):
    """
    Retrieves the expected intrinsic flux density from the loaded NPZ models,
    applying secular decrease corrections based on the observation epoch.
    """
    # Ensure models are loaded
    if FLUX_MODELS is None:
        if not load_flux_models():
            return np.full_like(freqs_hz, np.nan, dtype=float)

    # Ensure freqs_hz is an array
    freqs_hz = np.atleast_1d(freqs_hz)
    model_freqs_hz = FLUX_MODELS['frequencies_hz']

    # Determine the appropriate flux array key based on the source
    flux_key = None
    if source == 'CygA':
        flux_key = 'cyga_flux_jy'
    elif source == 'VirA':
        flux_key = 'vira_flux_jy'
    elif source == 'CasA':
        flux_key = 'casa_flux_jy_ref'
    elif source == 'TauA':
        flux_key = 'taua_flux_jy_ref'
    else:
        # Source not found in primary models
        logger.debug(f"Flux model data not found for source: {source}")
        return np.full_like(freqs_hz, np.nan, dtype=float)

    if flux_key not in FLUX_MODELS:
        logger.error(f"Key '{flux_key}' missing in NPZ file for source {source}.")
        return np.full_like(freqs_hz, np.nan, dtype=float)

    model_flux_ref = FLUX_MODELS[flux_key]

    # Interpolate the reference flux at the observed frequencies
    # Use log-log interpolation as radio spectra are typically power laws
    try:
        # Ensure inputs are positive before taking log
        if np.any(freqs_hz <= 0) or np.any(model_freqs_hz <= 0) or np.any(model_flux_ref <= 0):
             # Fallback to linear interpolation if log interpolation fails due to non-positive values
             logger.warning(f"Non-positive values detected in frequencies or fluxes for {source}. Falling back to linear interpolation.")
             interpolated_flux_ref = np.interp(freqs_hz, model_freqs_hz, model_flux_ref)
        else:
            log_obs_freqs = np.log10(freqs_hz)
            log_model_freqs = np.log10(model_freqs_hz)
            log_model_flux_ref = np.log10(model_flux_ref)

            # Numpy interp requires x-coordinates to be increasing
            if not np.all(np.diff(log_model_freqs) > 0):
                 # Sort if they are not increasing
                 sort_idx = np.argsort(log_model_freqs)
                 log_model_freqs = log_model_freqs[sort_idx]
                 log_model_flux_ref = log_model_flux_ref[sort_idx]

            # Interpolate
            interpolated_log_flux_ref = np.interp(log_obs_freqs, log_model_freqs, log_model_flux_ref)
            interpolated_flux_ref = 10**interpolated_log_flux_ref

    except Exception as e:
        logger.warning(f"Error during flux interpolation for {source}: {e}")
        return np.full_like(freqs_hz, np.nan, dtype=float)

    # Apply Secular Decrease (if applicable)
    if source == 'CasA':
        ref_epoch = FLUX_MODELS['reference_epoch'][0]
        # CasA decrease rate: -0.46% per year
        correction_factor = (1 + -0.46 / 100.0)**(obs_epoch - ref_epoch)
        return interpolated_flux_ref * correction_factor
    elif source == 'TauA':
        # Use specific TauA reference epoch if available, otherwise fallback
        ref_epoch = FLUX_MODELS.get('taua_reference_epoch', FLUX_MODELS['reference_epoch'])[0]
        # TauA decrease rate: -0.16% per year
        correction_factor = (1 + -0.16 / 100.0)**(obs_epoch - ref_epoch)
        return interpolated_flux_ref * correction_factor
    else:
        # No correction needed for CygA and VirA
        return interpolated_flux_ref


# ==============================================================================
# === Delay Calibration QA ===
# ==============================================================================

def read_delays_from_table(table_path):
    """Reads delay (K) values from a CASA calibration table using casacore.tables."""
    if not CASACORE_TABLES_AVAILABLE:
        logger.error("casacore.tables (or compatible) not available. Cannot read delays.")
        return None

    if not os.path.exists(table_path):
        logger.error(f"Delay table not found: {table_path}")
        return None

    try:
        # Use the available table module (casacore.tables or the compatibility layer)
        table_access = casacore.tables.table if hasattr(casacore, 'tables') else casacore.table

        with table_access(table_path, ack=False) as tb:
            # FPARAM holds the solutions for gaintype='K' (delays in ns)
            solutions = tb.getcol('FPARAM')
            antennas = tb.getcol('ANTENNA1')
            # flags = tb.getcol('FLAG') # Optional: use flags for weighted mean

        # Handle dimensions: Shape is typically (N_pol, N_chan, N_rows) or (N_rows, N_chan, N_pol)
        # For delay solutions (K), N_chan is usually 1.

        if solutions.ndim == 3:
             # Try to identify the channel dimension (usually size 1)
             if solutions.shape[1] == 1:
                  # Layout: (N_pol, 1, N_rows) -> Squeeze to (N_pol, N_rows)
                  solutions = solutions[:, 0, :]
             elif solutions.shape[0] == 1:
                   # Layout: (1, N_chan, N_rows) -> Squeeze to (N_chan, N_rows)
                   solutions = solutions[0, :, :]
             else:
                  logger.warning(f"Unexpected 3D shape for FPARAM: {solutions.shape}. Attempting to average over channels.")
                  # Fallback: Average over the middle dimension (assumed channels)
                  solutions = np.mean(solutions, axis=1)
        elif solutions.ndim != 2:
             logger.error(f"Unexpected shape for FPARAM column: {solutions.shape}. Expected 2 or 3 dimensions.")
             return None

        # Now solutions should be 2D. Ensure consistent orientation (N_rows x N_pols)
        # Assuming N_pol <= 4 and N_rows > N_pol
        if solutions.shape[0] < solutions.shape[1] and solutions.shape[0] <= 4:
             solutions = solutions.T # Transpose if (N_pol x N_rows)

        delay_data = {}
        # Iterate over rows (corresponding to antennas/spws combinations)
        for i in range(solutions.shape[0]):
            ant_id = antennas[i]

            # Average the delays across polarizations (e.g., XX and YY)
            pol_delays = solutions[i, :]
            avg_delay = np.mean(pol_delays)

            # Store data, potentially accumulating across different SPWs
            if ant_id not in delay_data:
                delay_data[ant_id] = []
            delay_data[ant_id].append(avg_delay)

        # Final averaging across SPWs if multiple entries exist per antenna
        final_delays = {ant_id: np.mean(data) for ant_id, data in delay_data.items()}
        return final_delays

    except Exception as e:
        logger.error(f"Error reading delays from {table_path}: {e}", exc_info=True)
        return None

def compare_and_diagnose_delays(new_delays, ref_delays, context):
    """Compares delays and runs diagnostics."""
    # Use the specific diagnostic logger
    diag_logger = pipeline_utils.get_logger('QA.Diagnostics')

    if not ref_delays:
        diag_logger.info("Reference delays not available for comparison.")
        return {}, []

    # Find antennas present in both datasets
    common_ants = sorted(list(set(new_delays.keys()) & set(ref_delays.keys())))

    if not common_ants:
        diag_logger.warning("No common antennas found between new and reference tables.")
        return {}, []

    differences = {}
    problematic_delays = {}
    threshold_ns = config.DELAY_DIFF_THRESHOLD_NS

    for ant_id in common_ants:
        diff = new_delays[ant_id] - ref_delays[ant_id]
        differences[ant_id] = diff
        # Check if the absolute difference exceeds the threshold
        if abs(diff) > threshold_ns:
            diag_logger.warning(f"Antenna {ant_id}: Problematic delay difference of {diff:.1f} ns detected (Threshold: {threshold_ns} ns).")
            problematic_delays[ant_id] = diff

    # Run SNAP2 diagnostics if issues were found
    if problematic_delays:
        # Load mapping from context if available, otherwise load from utils
        antenna_mapping = context.get('antenna_mapping')
        if not antenna_mapping:
             antenna_mapping = pipeline_utils.load_antenna_mapping()

        if antenna_mapping:
            pipeline_utils.run_snap2_diagnostics(problematic_delays, antenna_mapping)
        else:
            diag_logger.error("Could not load antenna mapping. Skipping SNAP2 diagnostics.")
    else:
        diag_logger.info("No significant delay differences detected compared to reference.")

    return differences, common_ants

def plot_delays(new_delays, ref_delays, differences, common_ants, new_label, ref_label, output_filename):
    """Generates the delay comparison plot."""
    if not _plotting_available:
        return False

    logger.info(f"Generating delay comparison plot: {os.path.basename(output_filename)}")

    # --- CSV DATA PREPARATION ---
    # Combine all delay data into a single DataFrame for saving.
    all_antennas = sorted(list(set(new_delays.keys()) | set(ref_delays.keys() if ref_delays else [])))
    df_data = pd.DataFrame(index=all_antennas).sort_index()
    df_data.index.name = 'antenna_id'
    
    df_data['delay_new_ns'] = pd.Series(new_delays)
    if ref_delays:
        df_data['delay_ref_ns'] = pd.Series(ref_delays)
    if differences:
        df_data['difference_ns'] = pd.Series(differences)
    
    # --- PLOTTING PREPARATION ---
    ants_new = sorted(new_delays.keys())
    delays_new_vals = [new_delays[ant] for ant in ants_new]

    if ref_delays:
        ants_ref = sorted(ref_delays.keys())
        delays_ref_vals = [ref_delays[ant] for ant in ants_ref]
    else:
        ants_ref = []
        delays_ref_vals = []

    if common_ants:
        diffs_vals = [differences[ant] for ant in common_ants]
        rms_diff = np.sqrt(np.mean(np.square(diffs_vals)))
    else:
        diffs_vals = []
        rms_diff = np.nan

    # --- PLOTTING ---
    if ref_delays:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
        ax2 = None

    if ants_ref:
        ax1.scatter(ants_ref, delays_ref_vals, label=f'{ref_label} (Ref)', alpha=0.6, marker='o', s=50)

    marker_style = 'x' if ref_delays else 'o'
    if ants_new:
        ax1.scatter(ants_new, delays_new_vals, label=f'{new_label} (New)', alpha=0.8, marker=marker_style, s=60, color='red')

    ax1.set_ylabel('Delay (ns)')
    ax1.set_title('Delay Calibration Comparison' if ref_delays else 'Delay Calibration Results')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    if ax2:
        if common_ants:
            ax2.scatter(common_ants, diffs_vals, color='purple', marker='s', s=50)

        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        thresh = config.DELAY_DIFF_THRESHOLD_NS
        ax2.axhline(y=thresh, color='orange', linestyle='--', linewidth=1.5, label=f'+/- {thresh} ns threshold')
        ax2.axhline(y=-thresh, color='orange', linestyle='--', linewidth=1.5)

        ax2.set_xlabel('Antenna Index (Correlator Number)')
        ax2.set_ylabel('Delay Difference (New - Ref) (ns)')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='lower right')

        textstr = f'RMS Difference: {rms_diff:.2f} ns\nCommon Antennas: {len(common_ants)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        ax_bottom = ax2
    else:
         ax1.set_xlabel('Antenna Index (Correlator Number)')
         ax_bottom = ax1

    max_ant_new = max(ants_new) if ants_new else 0
    ax_bottom.set_xlim(-5, max(max_ant_new + 5, config.N_ANTENNAS))

    plt.tight_layout()
    if ax2:
        plt.subplots_adjust(hspace=0.1)

    try:
        # Save plot
        plt.savefig(output_filename, dpi=150)
        
        # --- SAVE CSV ---
        csv_filename = output_filename.replace('.png', '.csv')
        df_data.to_csv(csv_filename)
        logger.info(f"Saved plot data to {os.path.basename(csv_filename)}")
        return True
    except Exception as e:
        logger.error(f"Error saving delay plot or CSV: {e}", exc_info=True)
        return False
    finally:
        plt.close(fig)

def analyze_and_plot_delays(context, delay_table_path, ref_delay_table_path):
    """Orchestrates the delay QA: Read, Compare, Diagnose, Plot."""
    new_delays = read_delays_from_table(delay_table_path)
    if not new_delays: return False

    if ref_delay_table_path and os.path.exists(ref_delay_table_path):
        ref_delays = read_delays_from_table(ref_delay_table_path)
        ref_label = os.path.basename(ref_delay_table_path) if ref_delays else "N/A (Load Failed)"
    else:
        logger.warning("Reference delay table path not provided or file missing. Skipping comparison.")
        ref_delays, ref_label = None, "N/A"

    differences, common_ants = compare_and_diagnose_delays(new_delays, ref_delays, context)

    time_identifier = context.get('time_identifier', 'unknown_time')
    # MODIFIED: Save plot to main qa_dir
    plot_filename = os.path.join(context['qa_dir'], f"QA_plot_delays_{time_identifier}.png")
    new_label = os.path.basename(delay_table_path)

    if plot_delays(new_delays, ref_delays, differences, common_ants, new_label, ref_label, plot_filename):
        logger.info("Delay QA plotting completed.")
        return True
    else:
        logger.error("Delay QA plotting failed.")
        return False

# ==============================================================================
# === Imaging QA Helper Functions ===
# ==============================================================================

def get_freq_from_fits(header):
    freq_hz = header.get('CRVAL3', header.get('CRVAL4', header.get('FREQ')))
    if freq_hz is not None:
        return float(freq_hz)
    # --- MODIFICATION: Check for 2D image freq ---
    elif 'FREQ' in header:
        return float(header['FREQ'])
    elif 'CRVAL1' in header and header.get('CTYPE1', '').upper() == 'FREQ':
        return float(header['CRVAL1'])
    elif 'CRVAL2' in header and header.get('CTYPE2', '').upper() == 'FREQ':
        return float(header['CRVAL2'])
    # --- END MODIFICATION ---
    else:
        logger.warning("Frequency information (CRVAL3/4 or FREQ) not found in FITS header.")
        return np.nan

def get_time_from_fits(header, context):
     obs_time_str = header.get('DATE-OBS')
     fallback_time = context.get('obs_info', {}).get('obs_mid_time', Time.now())
     if not obs_time_str:
          logger.warning("DATE-OBS not found in FITS header. Using context midpoint as fallback.")
          return fallback_time
     else:
          try:
              return Time(obs_time_str, scale='utc', location=config.OVRO_LOCATION)
          except Exception as e:
              logger.warning(f"Error parsing DATE-OBS '{obs_time_str}': {e}. Using fallback.")
              return fallback_time

def generate_image_snapshot(fits_path_or_hdu, output_jpg_path, stats, fit_results, 
                            cmap='inferno', stretch=AsinhStretch(a=0.1), vmin=None, vmax=None):
    """
    MODIFIED: Accepts either a FITS file path or a pre-opened HDU.
    MODIFIED: Accepts cmap, stretch, vmin, vmax arguments for all-sky images.
    """
    if not _plotting_available: return False
    
    hdu = None
    try:
        # Determine if input is a path or an HDU
        if isinstance(fits_path_or_hdu, str):
            hdul = fits.open(fits_path_or_hdu)
            hdu = hdul[0]
            fits_name = os.path.basename(fits_path_or_hdu)
            close_hdu = True
        elif isinstance(fits_path_or_hdu, fits.PrimaryHDU):
            hdu = fits_path_or_hdu
            fits_name = "FITS Plane"
            close_hdu = False
        else:
            logger.error("Invalid input to generate_image_snapshot. Must be file path or HDU.")
            return False

        data = hdu.data.squeeze()
        if data.ndim != 2:
            logger.warning(f"FITS data is not 2D (shape: {data.shape}) for {fits_name}. Skipping snapshot.")
            return False

        fig = plt.figure(figsize=(8, 10))

        try:
            wcs = WCS(hdu.header).celestial
            ax = fig.add_subplot(111, projection=wcs)
            use_wcs = True
        except Exception as e:
            logger.debug(f"Could not initialize WCS projection for {fits_name}: {e}")
            ax = fig.add_subplot(111)
            use_wcs = False

        # --- MODIFICATION: Use provided vmin/vmax or calculate ---
        if vmin is None or vmax is None:
            try:
                # Avoid divide-by-zero or warnings on all-NaN slices
                if np.all(np.isnan(data)):
                    vmin_calc, vmax_calc = 0, 1
                else:
                    vmin_calc, vmax_calc = PercentileInterval(99.5).get_limits(data)
            except Exception:
                vmin_calc, vmax_calc = np.nanmin(data), np.nanmax(data)
                if not np.isfinite(vmin_calc): vmin_calc = 0
                if not np.isfinite(vmax_calc): vmax_calc = 1
            if vmin is None: vmin = vmin_calc
            if vmax is None: vmax = vmax_calc
        # --- END MODIFICATION ---

        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch)
        im = ax.imshow(data, cmap=cmap, origin='lower', norm=norm)

        if use_wcs:
            ax.coords[0].set_axislabel('Right Ascension')
            ax.coords[1].set_axislabel('Declination')
            ax.coords.grid(True, color='white', ls='dotted', alpha=0.5)
        else:
            ax.set_xlabel('Pixel X')
            ax.set_ylabel('Pixel Y')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Flux Density (Jy/beam)')
        plt.title(fits_name, fontsize=9, wrap=True)

        stats_text = "--- Image Statistics (imstat) ---\n"
        rms_val = stats.get('rms', [np.nan])[0] if stats else np.nan
        max_val = stats.get('max', [np.nan])[0] if stats else np.nan
        stats_text += f"RMS: {rms_val:.3f} Jy/beam\nPeak: {max_val:.2f} Jy/beam\n"

        if fit_results and fit_results.get('converged', False) and 'results' in fit_results and 'component0' in fit_results['results']:
            comp = fit_results['results']['component0']
            stats_text += "\n--- Fitted Parameters (imfit) ---\n"
            try:
                flux_val, flux_err = comp['flux']['value'][0], comp['flux']['error'][0]
                stats_text += f"Int Flux: {flux_val:.2f} +/- {flux_err:.2f} Jy\n"
            except (KeyError, IndexError, TypeError): stats_text += "Int Flux: N/A\n"
            try:
                ra_val, dec_val = comp['shape']['direction']['m0']['value'], comp['shape']['direction']['m1']['value']
                pos_coord = SkyCoord(ra=ra_val*u.rad, dec=dec_val*u.rad, frame='icrs')
                stats_text += f"Pos: {pos_coord.to_string('hmsdms', precision=1)}\n"
            except (KeyError, TypeError): stats_text += "Pos: N/A\n"
            try:
                bmaj, bmin, unit = comp['shape']['majoraxis']['value'], comp['shape']['minoraxis']['value'], comp['shape']['majoraxis']['unit']
                stats_text += f"Size: {bmaj:.1f}x{bmin:.1f} {unit}"
            except (KeyError, TypeError): stats_text += "Size: N/A"
        elif fit_results:
            stats_text += "\n--- Fitted Parameters (imfit) ---\nConverged: No or data missing"

        fig.text(0.5, 0.05, stats_text, ha="center", fontsize=9, bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.8), verticalalignment='bottom')
        plt.tight_layout(rect=[0, 0.2, 1, 0.95])
        # MODIFICATION: Save as JPG not PNG
        output_jpg_path = output_jpg_path.replace('.png', '.jpg')
        fig.savefig(output_jpg_path, dpi=120)
        return True
    except Exception as e:
        logger.error(f"Error generating JPG snapshot for {fits_name}: {e}", exc_info=True)
        return False
    finally:
        if 'hdul' in locals() and close_hdu:
            hdul.close()
        plt.close('all')

        
def run_bdsf(fits_path):
    """ Run source detection with pybdsf.
    fits_path is the path to the all-sky FITS image
    """

    import bdsf  # TODO: we should move this outside the function
    img = bdsf.process_image(fits_path)
    fluxes = np.array([src.total_flux for src in img.sources])
    print(f"Found {len(fluxes)} sources with fluxes from {fluxes.min()} to {fluxes.max()}")
    if write_csv:
        csv_path = fits_path.replace('.fits', '.pybdsf.gaul')
        print(f"Writing {csv_path}")
        img.write_catalog(format='csv')
        return csv_path
# TODO: maybe define some QA metric on source list
#    else:
#        return fluxes
    
        
def run_imfit_imstat(fits_file, context, is_cube=False):
    """
    MODIFIED: Now accepts `is_cube` hint to run imstat correctly.
    MODIFIED: Calls imstat with region='' and box='' to avoid WCBox error.
    """
    try:
        with fits.open(fits_file) as hdul:
            if not np.any(np.isfinite(hdul[0].data)):
                logger.warning(f"Skipping imfit/imstat for {os.path.basename(fits_file)}: FITS image contains no valid data.")
                return None, None
    except Exception as e:
        logger.error(f"Could not read or validate FITS file {os.path.basename(fits_file)}: {e}")
        return None, None

    stats, fit_results = None, None
    if CASA_TASKS_AVAILABLE:
        logger.debug(f"Running imfit/imstat on {os.path.basename(fits_file)}")
        imstat_task, imfit_task = pipeline_utils.get_casa_task('imstat'), pipeline_utils.get_casa_task('imfit')
        if imstat_task and imfit_task:
            try:
                # --- MODIFICATION: Set region/box to empty strings ---
                imstat_params = {
                    'imagename': fits_file,
                    'region': '',
                    'box': ''
                }
                if is_cube:
                    imstat_params['axes'] = [0,1] # Get stats per channel
                
                stats = imstat_task(**imstat_params)
                # --- END MODIFICATION ---

                fit_results = imfit_task(imagename=fits_file) # imfit on cube fits each channel
                
                # Check convergence: 'converged' is now a list of bools if it was a cube
                if not fit_results or not 'converged' in fit_results:
                    logger.debug(f"imfit failed for {os.path.basename(fits_file)}.")
                elif not np.any(fit_results.get('converged', [])):
                     logger.debug(f"imfit did not converge on any channel for {os.path.basename(fits_file)}.")
            except Exception as e:
                logger.error(f"Error running CASA tasks on {fits_file}: {e}", exc_info=True)
                stats, fit_results = None, None
    else:
        logger.warning("CASA tasks not available. Skipping imfit/imstat.")
        
    # --- MODIFICATION: Snapshot generation for cubes/images ---
    base_name = os.path.basename(fits_file).replace('.fits', '').replace('-image', '')
    jpg_filename = f"QA_snapshot_{base_name}.jpg"
    fits_dir = os.path.dirname(fits_file)
    output_jpg_path = os.path.join(fits_dir, jpg_filename)

    # Generate a snapshot of the *first plane* or the 2D image
    try:
        with fits.open(fits_file) as hdul:
            first_plane_data = hdul[0].data.squeeze()
            if first_plane_data.ndim > 2:
                # Handle 4D (t, chan, y, x) or 3D (chan, y, x)
                while first_plane_data.ndim > 2:
                    first_plane_data = first_plane_data[0] # Select first plane
            
            # Create a minimal header for the 2D plane
            wcs = WCS(hdul[0].header).celestial
            plane_header = wcs.to_header()
            plane_header['NAXIS'] = 2
            
            # Create a fake 2D FITS HDU to pass to the snapshot function
            plane_hdu = fits.PrimaryHDU(data=first_plane_data, header=plane_header)
            
            # Get stats for the first plane only
            plane_stats = None
            if stats:
                if 'rms' in stats and isinstance(stats['rms'], np.ndarray) and stats['rms'].size > 0:
                    plane_stats = {'rms': [stats['rms'][0]], 'max': [stats['max'][0]]}
                elif 'rms' in stats: # 2D image stats
                    plane_stats = stats
            
            # Get fit results for the first plane only
            plane_fit = None
            if fit_results and isinstance(fit_results.get('converged'), (list, np.ndarray)) and fit_results.get('converged', [False])[0]:
                # Cube fit: get first channel
                if '0' in fit_results['results']['component0']:
                    plane_fit = {'converged': True, 'results': {'component0': fit_results['results']['component0']['0']}}
                else:
                    plane_fit = fit_results # Fallback
            elif fit_results and fit_results.get('converged', False): # Handle 2D image fit
                plane_fit = fit_results

            # Generate the snapshot
            generate_image_snapshot(plane_hdu, output_jpg_path, plane_stats, plane_fit)
            logger.debug(f"Generated QA snapshot (first plane/MFS): {jpg_filename} in {os.path.basename(fits_dir)}")

    except Exception as e:
        logger.error(f"Could not generate snapshot for {fits_file}: {e}", exc_info=True)


    return fit_results, stats

# ==============================================================================
# === Imaging QA Analysis (Spectrum) ===
# ==============================================================================

# --- MODIFIED: Reverted to per-channel FITS logic ---
def analyze_spectrum_images(qa_dir, source, time_identifier, context):
    
    spectrum_img_dir = context['web_spec_dir']
    # REVERT: Look for per-channel files
    pattern = os.path.join(spectrum_img_dir, f"{source}_spectrum_{time_identifier}-????-image.fits")
    fits_files = sorted(glob.glob(pattern))
    
    if not fits_files:
        logger.info(f"No spectrum per-channel FITS files found for {source} in {spectrum_img_dir}.")
        return pd.DataFrame()

    logger.info(f"Analyzing {len(fits_files)} per-channel spectrum images for {source}...")
    results = []
    
    for fits_file in tqdm(fits_files, desc=f"{source} Spectrum Analysis"):
        match = re.search(r'-(\d{4})-image\.fits', fits_file)
        if not match: continue
        channel_idx = int(match.group(1))

        # Snapshot will be saved in web_spec_dir by this function
        fit_results, stats = run_imfit_imstat(fits_file, context, is_cube=False)
        if not stats: continue

        try:
            freq_hz = get_freq_from_fits(fits.getheader(fits_file))
        except Exception as e:
            logger.error(f"Error reading FITS header for {fits_file}: {e}")
            continue

        flux_imfit, flux_err_imfit, fit_converged = np.nan, np.nan, False
        fit_ra_rad, fit_dec_rad = np.nan, np.nan
        if fit_results and fit_results.get('converged', False):
            try:
                comp = fit_results['results']['component0']
                flux_imfit, flux_err_imfit = comp['flux']['value'][0], comp['flux']['error'][0]
                fit_ra_rad, fit_dec_rad = comp['shape']['direction']['m0']['value'], comp['shape']['direction']['m1']['value']
                fit_converged = True
            except (KeyError, IndexError, TypeError):
                logger.debug(f"imfit converged but failed to parse results for {os.path.basename(fits_file)}")
        
        results.append({
            'channel_idx': channel_idx, 'freq_hz': freq_hz, 'freq_mhz': freq_hz / 1e6 if freq_hz else np.nan,
            'flux_imfit': flux_imfit, 'flux_err_imfit': flux_err_imfit,
            'fit_ra_rad': fit_ra_rad, 'fit_dec_rad': fit_dec_rad,
            'fit_converged': fit_converged,
            'flux_imstat': stats['max'][0], 'flux_err_imstat': stats['rms'][0],
            'rms': stats['rms'][0]
        })
        
    # --- NEW: Delete FITS files ---
    logger.info(f"Deleting {len(fits_files)} spectrum FITS files for {source}...")
    for f in fits_files:
        try: os.remove(f)
        except Exception as e: logger.warning(f"Failed to delete {f}: {e}")
    # Also delete the MFS file
    mfs_fits = glob.glob(os.path.join(spectrum_img_dir, f"{source}_spectrum_{time_identifier}-MFS-image.fits"))
    for f in mfs_fits:
        try: os.remove(f)
        except Exception as e: logger.warning(f"Failed to delete MFS file {f}: {e}")
    # --- END NEW ---
        
    return pd.DataFrame(results)

def plot_calibrator_spectrum(df_spec, source, qa_dir, time_identifier, beam_interpolator, context, flux_type='imstat'):
    if df_spec.empty or not _plotting_available: return
    logger.info(f"Plotting {flux_type.upper()} spectrum for {source}...")
    
    fig = None # Initialize fig to None
    try:
        flux_col_raw, flux_err_col_raw = f'flux_{flux_type}', f'flux_err_{flux_type}'
        df_plot = df_spec.dropna(subset=[flux_col_raw]).sort_values(by='freq_hz').copy()
        if df_plot.empty:
            logger.warning(f"No valid data for flux type '{flux_type}' for {source}. Skipping plot.")
            return

        freqs_mhz = df_plot['freq_hz'].values / 1e6
        beam_attenuation = pipeline_utils.calculate_beam_attenuation(context, source, freqs_mhz, beam_interpolator)
        df_plot['beam_attenuation'] = beam_attenuation
        df_plot[f'flux_corrected'] = np.where(beam_attenuation > 1e-6, df_plot[flux_col_raw] / beam_attenuation, np.nan)
        df_plot[f'flux_err_corrected'] = np.where(beam_attenuation > 1e-6, np.abs(df_plot[flux_err_col_raw]) / beam_attenuation, np.nan)
        
        obs_time = context.get('obs_info', {}).get('obs_mid_time')
        obs_epoch = obs_time.decimalyear if obs_time else Time.now().decimalyear
        df_plot['flux_expected'] = get_expected_flux(source, df_plot['freq_hz'].values, obs_epoch)

        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Use np.abs() on the error columns to prevent crash from negative imfit errors
        ax.errorbar(df_plot['freq_mhz'], df_plot[flux_col_raw], yerr=np.abs(df_plot[flux_err_col_raw]), fmt='s', color='gray', alpha=0.6, label='Raw (Apparent)', capsize=4)
        
        # --- MODIFICATION: Fix NameError ---
        # Was: yerr=np.abs(df_plot[flux_err_corrected])
        ax.errorbar(df_plot['freq_mhz'], df_plot['flux_corrected'], yerr=np.abs(df_plot[f'flux_err_corrected']), fmt='o', color='blue', label='Beam Corrected', capsize=5)
        # --- END MODIFICATION ---

        if df_plot['flux_expected'].notna().any():
            ax.plot(df_plot['freq_mhz'], df_plot['flux_expected'], color='red', linestyle='--', linewidth=2, label='Intrinsic Model')

        ax.set_title(f'{source} Spectrum ({flux_type.upper()}) - {time_identifier} (Epoch {obs_epoch:.2f})')
        ax.set(xscale='log', yscale='log', xlabel='Frequency (MHz)', ylabel='Flux Density (Jy)', xlim=(10, 100))
        ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.xaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_major_formatter(ScalarFormatter())
        plt.tight_layout()

        # MODIFIED: Save plot directly in main qa_dir
        plot_filename = os.path.join(qa_dir, f'QA_plot_spectrum_{source}_{time_identifier}_{flux_type}.png')
        plt.savefig(plot_filename, dpi=150)
        logger.info(f"Saved spectrum plot to {os.path.basename(plot_filename)}")
        
        csv_filename = plot_filename.replace('.png', '.csv')
        df_plot.to_csv(csv_filename, index=False, float_format='%.6f')
        logger.info(f"Saved plot data to {os.path.basename(csv_filename)}")

    except Exception as e:
        logger.error(f"Failed to generate spectrum plot for {source} ({flux_type}): {e}", exc_info=True)
    finally:
        if fig:
            plt.close(fig)

# ==============================================================================
# === Imaging QA Analysis (Scintillation / Time-Resolved Spectra) ===
# ==============================================================================

def analyze_scintillation_spectra(qa_dir, source, time_identifier, context):
    """Analyzes time-resolved spectral images (flux vs freq for each integration)."""
    # MODIFIED: Search in web_scint_dir
    scintillation_img_dir = context['web_scint_dir']
    pattern = os.path.join(scintillation_img_dir, f"{source}_scintillation_{time_identifier}-t????-????-image.fits")
    fits_files = sorted(glob.glob(pattern))

    if not fits_files:
        logger.info(f"No time-resolved spectral FITS files found for {source} in {scintillation_img_dir}.")
        return pd.DataFrame()

    results = []
    for fits_file in tqdm(fits_files, desc=f"{source} Time-Resolved Spectra Analysis"):
        match = re.search(r'-t(\d{4})-(\d{4})-image\.fits', fits_file)
        if not match:
            continue
        time_idx, channel_idx = int(match.group(1)), int(match.group(2))

        # MODIFIED: Snapshot will be saved in web_scint_dir by this function
        # Pass is_cube=False
        fit_results, stats = run_imfit_imstat(fits_file, context, is_cube=False)
        if not stats: continue
        
        try:
            header = fits.getheader(fits_file)
            freq_hz = get_freq_from_fits(header)
            obs_time = get_time_from_fits(header, context)
        except Exception as e:
            logger.error(f"Error reading FITS header for {fits_file}: {e}")
            continue

        flux, flux_err = (stats['max'][0], stats['rms'][0])
        # MODIFICATION: Need to handle single-plane imfit result
        if fit_results and fit_results.get('converged', False):
            try:
                # Check for nested channel results (from cube) vs single result
                if '0' in fit_results['results']['component0']:
                    comp = fit_results['results']['component0']['0'] # Not a cube, but imfit may still nest
                else:
                    comp = fit_results['results']['component0']
                flux = comp['flux']['value'][0]
                flux_err = comp['flux']['error'][0]
            except (KeyError, IndexError, TypeError): pass

        results.append({
            'time_idx': time_idx,
            'channel_idx': channel_idx,
            'time_utc': obs_time.datetime,
            'freq_hz': freq_hz,
            'freq_mhz': freq_hz / 1e6 if freq_hz else np.nan,
            'flux_jy': flux,
            'flux_err_jy': flux_err,
        })
        
    # --- NEW: Delete FITS files ---
    logger.info(f"Deleting {len(fits_files)} scintillation FITS files for {source}...")
    for f in fits_files:
        try: os.remove(f)
        except Exception as e: logger.warning(f"Failed to delete {f}: {e}")
    # --- END NEW ---

    return pd.DataFrame(results)

def plot_scintillation_spectra(df_scint, source, qa_dir, time_identifier):
    """Plots flux density vs frequency for each integration."""
    if df_scint.empty or not _plotting_available: return

    logger.info(f"Plotting time-resolved spectra for {source}...")
    
    fig = None
    try:
        df_plot = df_scint.sort_values(by=['time_idx', 'freq_mhz']).copy()
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot a separate line (spectrum) for each integration
        for time_idx, group in df_plot.groupby('time_idx'):
            ax.plot(group['freq_mhz'], group['flux_jy'], marker='.', linestyle='-', label=f'Int {time_idx}')

        ax.set_title(f'{source} Time-Resolved Spectra - {time_identifier}')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Flux Density (Jy)')
        ax.legend(title="Integration", fontsize='small', ncol=2) # Adjust legend display
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_yscale('log')
        plt.tight_layout()

        # MODIFIED: Save plot directly in main qa_dir
        plot_filename = os.path.join(qa_dir, f'QA_plot_scintillation_spectra_{source}_{time_identifier}.png')
        plt.savefig(plot_filename, dpi=150)
        logger.info(f"Saved time-resolved spectra plot to {os.path.basename(plot_filename)}")
        
        csv_filename = plot_filename.replace('.png', '.csv')
        df_plot.to_csv(csv_filename, index=False, float_format='%.6f')
        logger.info(f"Saved plot data to {os.path.basename(csv_filename)}")
    except Exception as e:
        logger.error(f"Failed to generate time-resolved spectra plot for {source}: {e}", exc_info=True)
    finally:
        if fig:
            plt.close(fig)

def plot_positional_offsets(results_dfs, context):
    """
    Plots the 2D positional offset (RA vs Dec) of fitted components.
    """
    if not _plotting_available: return
    qa_dir = context['qa_dir']
    time_identifier = context.get('time_identifier', 'unknown_time')
    logger.info("Generating 2D positional offset summary plot...")
    
    fig = None
    try:
        all_offsets_data = []
        for source, df in results_dfs.items():
            if df.empty or 'fit_ra_rad' not in df.columns: continue

            true_coord = config.PRIMARY_SOURCES.get(source, {}).get('skycoord')
            if not true_coord:
                logger.warning(f"True coordinates for {source} not found. Skipping offset calculation.")
                continue
                
            df_fit = df[df['fit_converged']].copy()
            if df_fit.empty: continue

            # Calculate RA and Dec offsets
            fitted_coords = SkyCoord(ra=df_fit['fit_ra_rad'].values*u.rad, dec=df_fit['fit_dec_rad'].values*u.rad, frame='icrs')
            
            # RA offset needs cos(Dec) correction
            delta_ra = (fitted_coords.ra - true_coord.ra) * np.cos(true_coord.dec)
            delta_dec = fitted_coords.dec - true_coord.dec
            
            df_fit['delta_ra_arcsec'] = delta_ra.to(u.arcsec).value
            df_fit['delta_dec_arcsec'] = delta_dec.to(u.arcsec).value
            df_fit['source'] = source
            all_offsets_data.append(df_fit)

        if not all_offsets_data:
            logger.info("No converged imfit positions found to generate offset plot.")
            return

        df_plot = pd.concat(all_offsets_data)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for source_name, group in df_plot.groupby('source'):
            ax.scatter(group['delta_ra_arcsec'], group['delta_dec_arcsec'], label=source_name, alpha=0.7, s=20)

        # Add crosshairs at the true position (0,0)
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('RA Offset (arcsec)')
        ax.set_ylabel('Dec Offset (arcsec)')
        ax.set_title(f'Calibrator 2D Positional Offsets - {time_identifier}')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal', adjustable='box') # Ensure RA/Dec scales are equal
        plt.tight_layout()
        
        # MODIFIED: Save plot directly in main qa_dir
        plot_filename = os.path.join(qa_dir, f'QA_plot_positional_offsets_{time_identifier}.png')
        plt.savefig(plot_filename, dpi=150)
        logger.info(f"Saved positional offset plot to {os.path.basename(plot_filename)}")
        
        csv_filename = plot_filename.replace('.png', '.csv')
        df_plot.to_csv(csv_filename, index=False, float_format='%.6f')
        logger.info(f"Saved plot data to {os.path.basename(csv_filename)}")
    except Exception as e:
        logger.error(f"Failed to generate positional offset plot for {e}", exc_info=True)
    finally:
        if fig:
            plt.close(fig)

# ==============================================================================
# === NEW 3-Color Image Generation ===
# ==============================================================================
def generate_3color_image(context):
    """
    Generates individual all-sky snapshots with 'viridis' cmap and
    a combined 3-color (RGB) JPG.
    """
    if not _plotting_available:
        logger.warning("Matplotlib/Astropy not available. Skipping 3-color image generation.")
        return
        
    logger.info("Generating 3-color all-sky image and individual snapshots...")
    time_identifier = context.get('time_identifier', 'unknown_time')
    allsky_dir = context['web_allsky_dir']
    qa_dir = context['qa_dir']

    # --- FIX 3: Modified 'bands' dictionary ---
    # Define band mappings
    bands = {
        "18-41MHz": {
            "file": os.path.join(allsky_dir, f"FullSky_zenith_18-41MHz_{time_identifier}-MFS-image.fits"),
            "stretch": LinearStretch(), "vmin": -60, "vmax": 420
        },
        "41-64MHz": {
            "file": os.path.join(allsky_dir, f"FullSky_zenith_41-64MHz_{time_identifier}-MFS-image.fits"),
            "stretch": LinearStretch(), "vmin": -20, "vmax": 140
        },
        "64-82MHz": {
            "file": os.path.join(allsky_dir, f"FullSky_zenith_64-82MHz_{time_identifier}-MFS-image.fits"),
            "stretch": LinearStretch(), "vmin": -15, "vmax": 105
        }
    }
    # --- END FIX 3 ---
    
    rgb_arrays = {}
    
    for band_name, info in bands.items():
        fits_file = info['file']
        if not os.path.exists(fits_file):
            logger.warning(f"FITS file for {band_name} not found: {fits_file}")
            continue
            
        try:
            # --- 1. Generate individual 'viridis' snapshot ---
            jpg_path = os.path.join(allsky_dir, f"QA_snapshot_FullSky_zenith_{band_name}_{time_identifier}.jpg")
            
            # --- FIX 3: Pass vmin/vmax from info dict ---
            generate_image_snapshot(
                fits_file, jpg_path, stats={}, fit_results=None,
                cmap='viridis', stretch=info['stretch'],
                vmin=info['vmin'], vmax=info['vmax']
            )
            
            # --- 2. Load data for 3-color image ---
            with fits.open(fits_file) as hdul:
                data = hdul[0].data.squeeze()
                if data.ndim != 2:
                    logger.warning(f"Data for {band_name} is not 2D, skipping 3-color.")
                    continue
                    
                # --- FIX 3: Apply normalization using ImageNormalize ---
                # This scales data to [0, 1] for make_lupton_rgb
                norm = ImageNormalize(vmin=info['vmin'], vmax=info['vmax'], stretch=info['stretch'])
                rgb_arrays[band_name] = norm(data)
                
        except Exception as e:
            logger.error(f"Failed to process all-sky image {fits_file}: {e}", exc_info=True)

    # --- 3. Create 3-color image ---
    r = rgb_arrays.get("18-41MHz")
    g = rgb_arrays.get("41-64MHz")
    b = rgb_arrays.get("64-82MHz")

    if r is not None and g is not None and b is not None:
        try:
            # Ensure all arrays have the same shape (WSClean should guarantee this)
            if r.shape != g.shape or r.shape != b.shape:
                logger.error("All-sky image dimensions do not match. Cannot create 3-color image.")
                return

            rgb_image = make_lupton_rgb(r, g, b, stretch=1, Q=0)
            
            # Save the 3-color JPG
            output_3color_path = os.path.join(qa_dir, f"QA_3color_allsky_{time_identifier}.jpg")
            plt.imsave(output_3color_path, rgb_image, origin='lower')
            logger.info(f"Saved 3-color all-sky image to {os.path.basename(output_3color_path)}")
            
        except Exception as e:
            logger.error(f"Failed to generate 3-color RGB image: {e}", exc_info=True)
    else:
        logger.warning("Missing one or more bands (R, G, B). Cannot create 3-color image.")


# ==============================================================================
# === QA Orchestration ===
# ==============================================================================

# --- NEW HELPER FOR PDF WARNINGS ---
def _map_channel_to_subband(chan_idx, context):
    """Maps a global channel index back to its sub-band name."""
    try:
        obs_info = context.get('obs_info', {})
        unique_freqs_detected = obs_info.get('unique_freqs_detected')
        # --- MODIFICATION: Dynamically detect averaging ---
        total_chans = obs_info.get('total_channels_concatenated')
        num_spw = obs_info.get('num_spw_concatenated')
        if not (unique_freqs_detected and total_chans and num_spw and num_spw > 0):
             nchan_per_spw = config.NCHAN_PER_SUBBAND # Fallback
        else:
             nchan_per_spw = total_chans // num_spw
        # --- END MODIFICATION ---

        if not unique_freqs_detected:
            return f"Chan {chan_idx}"

        # Parse and sort detected frequencies to get SPW order
        freq_mapping = []
        for freq_str in unique_freqs_detected:
            match = re.match(r'(\d+)MHz', freq_str)
            if match:
                freq_val = int(match.group(1))
                freq_mapping.append((freq_val, freq_str))
        freq_mapping.sort(key=lambda x: x[0])
        
        spw_id = chan_idx // nchan_per_spw
        
        if spw_id < len(freq_mapping):
            return freq_mapping[spw_id][1] # Return the name, e.g., "41MHz"
        else:
            return f"SPW {spw_id}"
    except Exception:
        return f"Chan {chan_idx}"

# --- MODIFIED: Add results_dfs parameter and warning logic ---
def generate_pdf_report(context, results_dfs):
    if not _reportlab_available:
        logger.warning("ReportLab library not found. Skipping PDF report generation.")
        return

    logger.info("--- Generating PDF Summary Report ---")
    # MODIFIED: Get directories from context
    qa_dir = context['qa_dir']
    ood_dir = context['ood_dir']
    web_allsky_dir = context['web_allsky_dir']
    web_spec_dir = context['web_spec_dir']
    
    time_identifier = context.get('time_identifier', 'unknown_time')
    # MODIFIED: Save PDF to ood_dir
    pdf_path = os.path.join(ood_dir, f'QA_Report_{time_identifier}.pdf') 

    try:
        doc = SimpleDocTemplate(pdf_path)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("OVRO-LWA Calibration Pipeline QA Report", styles['h1']))
        story.append(Spacer(1, 0.2*inch))
        
        obs_info = context.get('obs_info', {})
        story.append(Paragraph(f"<b>Observation:</b> {time_identifier}", styles['Normal']))
        story.append(Paragraph(f"<b>LST:</b> {obs_info.get('lst_hour', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Midpoint UTC:</b> {obs_info.get('obs_mid_time', 'N/A')}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

        # --- NEW: Add Fit Warnings Section ---
        fit_warnings = []
        
        # Get a set of all sub-bands that were processed
        all_subbands_processed = set()
        if results_dfs:
            any_df = next(iter(results_dfs.values()), pd.DataFrame())
            if not any_df.empty:
                all_subbands_processed = set(any_df['channel_idx'].apply(
                    lambda x: _map_channel_to_subband(x, context)
                ))

        # Create a sorted list of sub-band names (e.g., "18MHz", "23MHz"...)
        try:
            sorted_subbands = sorted(all_subbands_processed, key=lambda x: int(re.match(r'(\d+)MHz', x).group(1)))
        except:
             sorted_subbands = sorted(all_subbands_processed)


        for subband in sorted_subbands:
            sources_in_subband = []
            sources_with_no_fit = []
            
            for source, df in results_dfs.items():
                if df.empty:
                    # If DF is empty, we can't know which subbands were processed for it
                    # This case is handled by the "no valid spectral data" log message
                    continue

                # Find channels corresponding to this sub-band
                subband_chans = df['channel_idx'].apply(
                    lambda x: _map_channel_to_subband(x, context)
                ) == subband
                
                if not subband_chans.any(): # Source wasn't in this sub-band
                    continue

                sources_in_subband.append(source)
                
                # Check if *any* fit converged in this sub-band for this source
                if not df[subband_chans]['fit_converged'].any():
                    sources_with_no_fit.append(source)

            # Now issue warnings based on the logic
            if len(sources_with_no_fit) == len(sources_in_subband) and len(sources_in_subband) > 0:
                # Warning 2: All sources failed in this sub-band
                warn_msg = (f"<b>No source has a converged image fit in sub-band {subband}.</b> "
                            f"Calibration for this sub-band may have serious errors, please use with caution.")
                fit_warnings.append(warn_msg)
            elif len(sources_with_no_fit) > 0:
                # Warning 1: At least one source failed
                for source in sources_with_no_fit:
                    warn_msg = f"<b>Source {source}</b> does not have a converged image fit in <b>sub-band {subband}</b>."
                    fit_warnings.append(warn_msg)

        if fit_warnings:
            story.append(Paragraph("?? FIT WARNINGS", styles['h2']))
            for warning in fit_warnings:
                story.append(Paragraph(f"<font color='red'>{warning}</font>", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        # --- END NEW ---

        story.append(Paragraph("Run Summary", styles['h2']))
        log_summary = []
        try:
            with open(context['log_filepath'], 'r') as f:
                for line in f:
                    # Capture key success/failure messages and warnings
                    if 'Pipeline finished with status:' in line:
                        log_summary.append(line.strip())
                    elif 'CRITICAL:' in line and 'SNAP2 Board Group' in line:
                         log_summary.append(f"<font color='red'><b>CRITICAL:</b> {line.strip()}</font>")
                    elif ('WARNING:' in line or 'ERROR:' in line) and ('failed' in line or 'exceeds threshold' in line):
                         # Make warnings more visible
                         color = 'red' if 'ERROR' in line else 'orange'
                         level = 'ERROR' if 'ERROR' in line else 'WARNING'
                         log_summary.append(f"<font color='{color}'><b>{level}:</b> {line.strip()}</font>")
        except Exception as e:
            log_summary.append(f"Could not read log file for summary: {e}")

        # Limit number of summary lines to avoid excessive length
        max_summary_lines = 15
        if len(log_summary) > max_summary_lines:
            log_summary = log_summary[:max_summary_lines] + ["... (see full log for more details)"]
            
        for item in log_summary:
            story.append(Paragraph(item, styles['Code']))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("Recommendations", styles['h2']))
        story.append(Paragraph("Review all plots for quality. If pipeline status is 'SUCCESS', the bandpass and delay calibration tables are likely reliable. Check for any major warnings above.", styles['Normal']))
        story.append(PageBreak())

        # --- Include Plots and Snapshots ---
        story.append(Paragraph("QA Plots and Snapshots", styles['h1']))
        
        # MODIFIED: Gather images from new locations
        
        # 1. 3-Color All Sky Image (first, as requested)
        story.append(Paragraph("All Sky 3-Color Image", styles['h2']))
        all_sky_3color = os.path.join(qa_dir, f"QA_3color_allsky_{time_identifier}.jpg")
        if not os.path.exists(all_sky_3color):
            story.append(Paragraph("3-color all-sky image not found.", styles['Normal']))
        else:
            try:
                img = Image(all_sky_3color, width=7*inch, height=7*inch, kind='proportional')
                story.append(img)
            except Exception as e:
                 logger.error(f"Error processing image {all_sky_3color} for PDF: {e}")
                 story.append(Paragraph(f"<i>Error loading image: {os.path.basename(all_sky_3color)}</i>", styles['Normal']))
        story.append(PageBreak())

        # 2. Individual All Sky Images
        story.append(Paragraph("All Sky Individual Band Images", styles['h2']))
        all_sky_snapshots = sorted(glob.glob(os.path.join(web_allsky_dir, 'QA_snapshot_FullSky*.jpg')))
        if not all_sky_snapshots:
            story.append(Paragraph("No all-sky snapshots found.", styles['Normal']))
        for img_path in all_sky_snapshots:
            try:
                img = Image(img_path, width=7*inch, height=7*inch, kind='proportional')
                story.append(KeepTogether([
                    Paragraph(os.path.basename(img_path), styles['h3']),
                    img,
                    Spacer(1, 0.2*inch)
                ]))
            except Exception as e:
                 logger.error(f"Error processing image {img_path} for PDF: {e}")
                 story.append(Paragraph(f"<i>Error loading image: {os.path.basename(img_path)}</i>", styles['Normal']))
        story.append(PageBreak())

        # 3. Gather Summary Plots (PNGs directly in qa_dir)
        story.append(Paragraph("Summary Plots", styles['h2']))
        plot_files = sorted(glob.glob(os.path.join(qa_dir, 'QA_plot_*.png')))
        
        # 4. Gather Snapshots (JPGs from web_spec_dir)
        # --- MODIFICATION: Removed web_scint_dir, re-added web_spec_dir ---
        story.append(Paragraph("Per-Channel Spectrum Snapshots", styles['h2']))
        snapshot_patterns = [
            os.path.join(web_spec_dir, 'QA_snapshot_*.jpg')
        ]
        snapshot_files = []
        for pattern in snapshot_patterns:
            snapshot_files.extend(glob.glob(pattern))
        snapshot_files = sorted(snapshot_files)
        
        all_other_images = plot_files + snapshot_files
        
        if not all_other_images:
            story.append(Paragraph("No summary plots or snapshots found.", styles['Normal']))
            
        for img_path in all_other_images:
            if not os.path.exists(img_path): continue

            elements_to_keep = []
            elements_to_keep.append(Paragraph(os.path.basename(img_path), styles['h3']))
            
            try:
                img = Image(img_path)
                img_width, img_height = img.imageWidth, img.imageHeight
                aspect = img_height / float(img_width)
                if aspect > 1.2: # Tall image (like snapshots)
                     new_width = 5.5 * inch
                else: # Wide image (like plots)
                     new_width = 7 * inch
                img.drawWidth = new_width
                img.drawHeight = new_width * aspect
                elements_to_keep.append(img)
            except Exception as e:
                 logger.error(f"Error processing image {img_path} for PDF: {e}")
                 elements_to_keep.append(Paragraph(f"<i>Error loading image: {os.path.basename(img_path)}</i>", styles['Normal']))

            elements_to_keep.append(Spacer(1, 0.2*inch))
            story.append(KeepTogether(elements_to_keep))

            if img_path in plot_files:
                 story.append(PageBreak())

        doc.build(story)
        logger.info(f"Successfully generated PDF report: {os.path.basename(pdf_path)}")

    except Exception as e:
        logger.exception(f"Failed to generate PDF report: {e}")


def run_qa(context):
    """Orchestrates the entire Post-Imaging QA workflow."""
    logger.info("Starting Post-Imaging QA.")
    # MODIFIED: Get main qa_dir, not subdirs
    qa_dir = context['qa_dir']
    time_identifier = context.get('time_identifier', 'unknown_time')
    modeled_sources = context.get('model_sources', [])

    if not load_flux_models():
        logger.warning("Proceeding with QA without reference flux models.")

    beam_interpolator = pipeline_utils.get_beam_interpolator(config.BEAM_MODEL_H5)
    if beam_interpolator is None:
        logger.warning("Proceeding with QA without beam correction.")

    results_dfs = {} # Store spectrum analysis results per source

    if modeled_sources:
        for source in modeled_sources:
            logger.info(f"--- QA Processing Source: {source} ---")

            # 1. Analyze frequency-domain spectrum images (now looks in web_spec_dir)
            # --- MODIFIED: This function is now reverted ---
            df_spec = analyze_spectrum_images(qa_dir, source, time_identifier, context)
            results_dfs[source] = df_spec # Store for combined offset plot

            if df_spec.empty:
                logger.info(f"No valid spectral data points found for {source} after analysis. Skipping spectrum plots.")
            else:
                # MODIFIED: Pass main qa_dir for saving plots
                plot_calibrator_spectrum(df_spec, source, qa_dir, time_identifier, beam_interpolator, context, flux_type='imstat')
                if df_spec['fit_converged'].any():
                    plot_calibrator_spectrum(df_spec, source, qa_dir, time_identifier, beam_interpolator, context, flux_type='imfit')
                else:
                    logger.info(f"Skipping imfit plot for {source} as no fits converged.")
            
            # (FITS Deletion for spectrum is now *inside* analyze_spectrum_images)

            # 2. Analyze time-resolved spectra (now looks in web_scint_dir)
            df_scint_spectra = analyze_scintillation_spectra(qa_dir, source, time_identifier, context)
            if not df_scint_spectra.empty:
                # MODIFIED: Pass main qa_dir for saving plots
                plot_scintillation_spectra(df_scint_spectra, source, qa_dir, time_identifier)
            
            # (FITS Deletion for scintillation is now *inside* analyze_scintillation_spectra)

    else:
         logger.info("No sources modeled. Skipping targeted Imaging QA.")

    # 3. Generate combined positional offset plot (using results_dfs)
    if results_dfs:
        # MODIFIED: Pass context (which contains qa_dir)
        plot_positional_offsets(results_dfs, context)
        
    # 4. Generate snapshot of the MFS Full Sky image
    # MODIFIED: Do not run imstat/imfit. Call new 3-color function instead.
    generate_3color_image(context)


    # 5. Generate final report (looks for plots/snapshots in correct locations)
    # --- MODIFICATION: Pass results_dfs ---
    generate_pdf_report(context, results_dfs)
    logger.info("Post-Imaging QA finished.")
    return True

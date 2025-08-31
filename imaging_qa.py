# imaging_qa.py
import os
import sys
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import pipeline utilities and config
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import pipeline_utils
import pipeline_config as config

# Initialize logger
logger = pipeline_utils.get_logger('ImagingQA')

# CASA Imports (needed for imfit/imstat)
CASA_AVAILABLE = pipeline_utils.CASA_AVAILABLE
CASA_IMPORTS = pipeline_utils.CASA_IMPORTS

# ==============================================================================
# === Helper Functions ===
# ==============================================================================

def run_wsclean(ms_path, image_name_base, channels_out, intervals_out=None):
    """Executes WSClean command, imaging the CORRECTED_DATA column."""
    
    cmd = [config.WSCLEAN_PATH]
    
    # --- MODIFICATION: Update image size ---
    # Replace the default size with a larger one for better RMS estimation
    wsclean_params = config.WSCLEAN_COMMON_PARAMS.copy()
    size_index = -1
    for i, param in enumerate(wsclean_params):
        if param == '-size':
            size_index = i
            break
            
    if size_index != -1 and size_index + 2 < len(wsclean_params):
        wsclean_params[size_index + 1] = '512'
        wsclean_params[size_index + 2] = '512'
        logger.info("Updated WSClean image size to 512x512.")
        
    cmd.extend(wsclean_params)
    # --- END MODIFICATION ---

    cmd.extend(["-data-column", "CORRECTED_DATA", "-join-channels"])
    cmd.extend(["-channels-out", str(channels_out)])
    
    if intervals_out:
        cmd.extend(["-intervals-out", str(intervals_out)])
        desc = f"WSClean (Scintillation: {intervals_out} intervals)"
    else:
        desc = f"WSClean (Spectrum)"

    cmd.extend(["-name", image_name_base])
    cmd.append(ms_path)
    
    original_openblas_threads = os.environ.get("OPENBLAS_NUM_THREADS")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    try:
        pipeline_utils.run_external_command(
            cmd,
            description=desc,
            logger_name=logger.name
        )
        return True
    except RuntimeError:
        return False
    finally:
        if original_openblas_threads is None:
            os.environ.pop("OPENBLAS_NUM_THREADS", None)
        else:
            os.environ["OPENBLAS_NUM_THREADS"] = original_openblas_threads


def measure_flux_imfit(fits_image_path):
    """Uses CASA imfit to measure the flux density of a source at the phase center."""
    if not CASA_AVAILABLE:
        logger.error("CASA not available. Cannot use imfit for flux measurement.")
        return None

    logger.debug(f"Measuring flux using imfit on: {os.path.basename(fits_image_path)}")
    
    try:
        # --- MODIFICATION START: Robustly get the imfit tool ---
        imfit_tool = CASA_IMPORTS.get('imfit')
        if not imfit_tool:
            from casatools import imfit as imfit_tool
        # --- MODIFICATION END ---

        # Fit a point source near the center of the image.
        results = imfit_tool(imagename=fits_image_path, box="")

        if results and results.get('converged', False):
            flux_jy = results['results']['component0']['flux']['value'][0]
            logger.debug(f"imfit result: {flux_jy:.4f} Jy")
            return flux_jy
        else:
            logger.warning(f"imfit failed to converge for {os.path.basename(fits_image_path)}. Falling back to imstat.")
            return measure_flux_imstat(fits_image_path)

    except Exception as e:
        logger.error(f"Error during CASA imfit execution on {fits_image_path}: {e}")
        # Add a fallback to imstat if imfit fails for any reason
        logger.info("Falling back to imstat due to imfit error.")
        return measure_flux_imstat(fits_image_path)


def measure_flux_imstat(fits_image_path):
    """Uses CASA imstat to measure the peak pixel value as a fallback."""
    if not CASA_AVAILABLE: return None
    
    try:
        # --- MODIFICATION START: Robustly get the imstat tool ---
        imstat_tool = CASA_IMPORTS.get('imstat')
        if not imstat_tool:
            from casatools import imstat as imstat_tool
        # --- MODIFICATION END ---
        
        stats = imstat_tool(imagename=fits_image_path, box="") # Use a small box around the center
        peak_flux = stats['max'][0]
        logger.debug(f"imstat peak: {peak_flux:.4f} Jy")
        return peak_flux
    except Exception as e:
        logger.error(f"Error during CASA imstat execution on {fits_image_path}: {e}")
        return None

# ==============================================================================
# === Analysis Functions ===
# ==============================================================================

def analyze_spectrum(image_name_base, source_name, qa_dir):
    """Analyzes spectrum FITS files and generates Flux vs Frequency plot."""
    logger.info(f"Analyzing Spectrum for {source_name}")
    
    search_pattern = f"{image_name_base}-????-image.fits"
    fits_files = sorted(glob.glob(search_pattern))
    
    if not fits_files:
        logger.warning("No spectrum FITS files found. Cannot analyze spectrum.")
        return

    fluxes = []
    
    for fits_file in tqdm(fits_files, desc=f"Measuring Spectrum Flux ({source_name})", unit="file"):
        flux = measure_flux_imfit(fits_file)
        fluxes.append(flux if flux is not None else np.nan)

    plt.figure(figsize=(10, 6))
    plt.plot(fluxes, marker='o', linestyle='-', label=f'Apparent Flux ({source_name})')
    plt.xlabel('Channel Index (Placeholder for Frequency)')
    plt.ylabel('Flux Density (Jy)')
    plt.title(f'Flux vs. Frequency - {source_name}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(qa_dir, f"QA_plot_spectrum_{source_name}.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved spectrum plot to {os.path.basename(plot_path)}")


def analyze_scintillation(image_name_base, source_name, qa_dir, channels_out):
    """Analyzes scintillation FITS files and generates Flux vs Time plots."""
    logger.info(f"Analyzing Scintillation for {source_name}")

    for channel_idx in range(channels_out):
        channel_str = f"{channel_idx:04d}"
        search_pattern = f"{image_name_base}-????-{channel_str}-image.fits"
        fits_files = sorted(glob.glob(search_pattern))
        
        if not fits_files:
            continue

        fluxes = []
        
        for fits_file in tqdm(fits_files, desc=f"Measuring Scintillation (Chan {channel_idx})", unit="file", leave=False):
            flux = measure_flux_imfit(fits_file)
            fluxes.append(flux if flux is not None else np.nan)

        plt.figure(figsize=(10, 4))
        plt.plot(fluxes, marker='.', linestyle='-', label=f'Channel {channel_idx}')
        plt.xlabel('Time Interval Index')
        plt.ylabel('Flux Density (Jy)')
        plt.title(f'Scintillation Light Curve - {source_name} (Chan {channel_idx})')
        plt.grid(True)
        plot_path = os.path.join(qa_dir, f"QA_plot_scintillation_{source_name}_chan{channel_idx}.png")
        plt.savefig(plot_path)
        plt.close()

    logger.info(f"Saved scintillation plots for {source_name}.")


# ==============================================================================
# === Main Logic ===
# ==============================================================================

def run_imaging_qa(context):
    """
    Orchestrates the Imaging QA workflow on pre-calibrated data.
    """
    logger.info("Starting Imaging QA.")

    ms_path = context.get('concatenated_ms_path')
    qa_dir = context['qa_dir']
    
    channels_out = config.WSCLEAN_DEFAULT_CHANNELS_OUT
    
    # --- MODIFICATION START ---
    # Dynamically get the number of integrations for scintillation imaging
    num_integrations = pipeline_utils.get_num_integrations(ms_path)
    if num_integrations:
        intervals_out = num_integrations
        logger.info(f"Dynamically determined Intervals Out: {intervals_out}")
    else:
        intervals_out = config.WSCLEAN_DEFAULT_INTERVALS_OUT
        logger.warning(f"Could not determine number of integrations. Falling back to default: {intervals_out}")
    # --- MODIFICATION END ---
    
    modeled_sources = context.get('model_sources', [])
    
    if not modeled_sources:
        logger.warning("No sources were modeled in Step 2. Skipping Imaging QA.")
        return True

    for source_name in modeled_sources:
        coords = config.PRIMARY_SOURCES.get(source_name)
        if not coords: continue
        
        logger.info(f"\n--- Analyzing Source: {source_name} ---")
            
        logger.info(f"Rotating phase center to {source_name} ({coords['ra']}, {coords['dec']})...")
        cmd_rotate = [config.CHGCENTRE_PATH, ms_path, coords["ra"], coords["dec"]]
        try:
            pipeline_utils.run_external_command(cmd_rotate, f"chgcentre ({source_name})", logger.name)
        except RuntimeError:
            logger.error(f"chgcentre failed for {source_name}. Skipping imaging QA for this source.")
            continue

        image_base_spec = os.path.join(qa_dir, f"{source_name}_spectrum")
        image_base_scint = os.path.join(qa_dir, f"{source_name}_scintillation")

        if run_wsclean(ms_path, image_base_spec, channels_out, None):
            analyze_spectrum(image_base_spec, source_name, qa_dir)
        else:
            logger.error(f"Spectral imaging failed for {source_name}.")

        if run_wsclean(ms_path, image_base_scint, channels_out, intervals_out):
            analyze_scintillation(image_base_scint, source_name, qa_dir, channels_out)
        else:
            logger.error(f"Scintillation imaging failed for {source_name}.")
        
    logger.info("Imaging QA step finished.")
    return True

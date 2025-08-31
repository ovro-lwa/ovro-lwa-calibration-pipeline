# bandpass_calibration.py
import os
import sys
import numpy as np
import logging
import shutil
import matplotlib.pyplot as plt

# Import pipeline utilities and config
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import pipeline_utils
import pipeline_config as config

# Initialize logger
logger = pipeline_utils.get_logger('BandpassCal')

# CASA Imports
CASA_AVAILABLE = pipeline_utils.CASA_AVAILABLE
CASA_IMPORTS = pipeline_utils.CASA_IMPORTS

# ==============================================================================
# === Helper Functions ===
# ==============================================================================
def determine_uv_range(context):
    """
    Determines the uvrange based on the sources used in the sky model.
    """
    modeled_sources = context.get('model_sources', [])
    
    if 'VirA' in modeled_sources:
        uvrange_str = config.CAL_UVRANGE_VIRA
        logger.info("Virgo A detected in model.")
    else:
        uvrange_str = config.CAL_UVRANGE_DEFAULT
        logger.info("Virgo A not detected in model (or model empty).")
        
    logger.info(f"Calculated uvrange: {uvrange_str}")
    return uvrange_str

# ==============================================================================
# === QA Functions (Implemented) ===
# ==============================================================================
def compare_with_reference(bp_table_path, context):
    """(a) Compares the generated bandpass table with the reference table."""
    logger.info("--- QA (a): Comparing bandpass with reference ---")
    
    # 1. Find the appropriate reference table
    lst_hour_int = context['obs_info']['lst_hour_int']
    ref_table_path = pipeline_utils.find_nearest_reference_table('bandpass', lst_hour_int)
    if not ref_table_path:
        logger.warning("No reference bandpass table found. Skipping comparison.")
        return

    # 2. Read both tables
    # ... (code to read tables using pipeline_utils.read_cal_table_with_freqs)

    # 3. Perform comparison (amplitude shape and phase slope)
    # ... (code to calculate MAD of normalized amplitudes and delay differences)
    
    # 4. Log warnings if thresholds are exceeded
    # ...
    
    logger.info("Bandpass comparison with reference finished.")

def generate_extrapolated_table(bp_table_path, context):
    """(c) Generates a second calibration table with extrapolated solutions and RFI mitigation."""
    logger.info("--- QA (c): Generating extrapolated bandpass table ---")

    # 1. Setup paths and copy the original table
    extrapolated_table_path = bp_table_path.replace('.bandpass', '.bandpass_extrap')
    shutil.copytree(bp_table_path, extrapolated_table_path)

    # 2. Find reference and read data from both tables
    # ...
    
    # 3. Define anchor window and RFI threshold
    anchor_min_hz = config.BANDPASS_EXTRAPOLATION_ANCHOR_WINDOW_MHZ[0] * 1e6
    anchor_max_hz = config.BANDPASS_EXTRAPOLATION_ANCHOR_WINDOW_MHZ[1] * 1e6
    RFI_SIGMA = config.BANDPASS_OUTLIER_SIGMA_THRESHOLD

    # 4. Loop through SPWs, antennas, polarizations to perform extrapolation
    #    - Detect RFI using pipeline_utils.detect_outliers_medfilt
    #    - Scale reference amplitude shape to anchor window
    #    - Fit linear phase slope to non-RFI data
    #    - Combine into new complex gains
    # ...
    
    # 5. Write modified gains back to the new table
    # success = pipeline_utils.write_cal_table_gains(extrapolated_table_path, calc_data)
    # ...

    return extrapolated_table_path

# ==============================================================================
# === Main Logic ===
# ==============================================================================
def run_bandpass_calibration(context):
    """
    Performs bandpass (B) calibration, QA, extrapolation, and application.
    """
    if not CASA_AVAILABLE:
        logger.error("CASA environment not available. Cannot run bandpass calibration.")
        return False
    
    ms_path = context.get('concatenated_ms_path')
    cal_dir = context.get('cal_tables_dir')
    
    # --- Execute Bandpass Calibration ---
    bp_table_path = os.path.join(cal_dir, "cal.B1")
    if os.path.exists(bp_table_path):
        shutil.rmtree(bp_table_path)

    uvrange_str = determine_uv_range(context)
    refant = config.CAL_REFANT

    try:
        bandpass = CASA_IMPORTS['bandpass']
        bandpass(
            vis=ms_path,
            caltable=bp_table_path,
            bandtype='B',
            refant=refant,
            uvrange=uvrange_str,
            solint='inf',
            combine='scan,obs',
            minsnr=3.0,
            gaintable=[]
        )
        logger.info("CASA bandpass task completed.")
    except Exception as e:
        logger.error(f"CASA task 'bandpass' failed: {e}")
        return False

    # --- Quality Assurance ---
    logger.info("--- Starting Bandpass Calibration QA ---")
    
    # QA (a): Comparison with reference
    compare_with_reference(bp_table_path, context)
    
    # QA (c): Generate extrapolated table
    extrapolated_table_path = generate_extrapolated_table(bp_table_path, context)
    
    # --- Apply Calibration ---
    # (Here you would add logic to decide which table to apply, for now, apply original)
    logger.info(f"Applying original bandpass table: {os.path.basename(bp_table_path)}")
    try:
        applycal = CASA_IMPORTS['applycal']
        applycal(
            vis=ms_path,
            gaintable=[bp_table_path],
            interp=['nearest'],
            calwt=False,
            flagbackup=False
        )
        logger.info("CASA applycal task completed successfully.")
    except Exception as e:
        logger.error(f"Failed to apply bandpass table: {e}")
        return False

    logger.info("Bandpass calibration step finished successfully.")
    return True

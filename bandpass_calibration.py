# bandpass_calibration.py
import os
import sys
import numpy as np
import logging
import shutil

# Import pipeline utilities and config
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import pipeline_utils
import pipeline_config as config

# Initialize logger
logger = pipeline_utils.get_logger('BandpassCal')

# CASA Imports handled by pipeline_utils
CASA_AVAILABLE = pipeline_utils.CASA_AVAILABLE
CASA_IMPORTS = pipeline_utils.CASA_IMPORTS

# ==============================================================================
# === Helper Functions ===
# ==============================================================================

def determine_uv_range(context):
    """
    Determines the uvrange based on the sources used in the sky model.
    (This is identical logic to delay_calibration.py)
    """
    # (ii) UV Range Selection Logic
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
# === Main Logic ===
# ==============================================================================

def run_bandpass_calibration(context):
    """
    Performs bandpass (B) calibration and applies the solution to the MS.
    """
    logger.info("Starting Bandpass Calibration (gaintype=B).")

    if not CASA_AVAILABLE:
        logger.error("CASA environment not available. Cannot run calibration tasks.")
        return False

    ms_path = context.get('concatenated_ms_path')
    if not ms_path or not os.path.exists(ms_path):
        logger.error(f"MS not found: {ms_path}")
        return False

    tables_dir = context['tables_dir']
    obs_date_str = context['obs_info']['obs_date']
    lst_str = context['obs_info']['lst_hour']

    bp_table_name = f"calibration_{obs_date_str}_{lst_str}.B"
    bp_table_path = os.path.join(tables_dir, bp_table_name)

    if os.path.exists(bp_table_path):
        logger.info(f"Removing existing bandpass table: {bp_table_path}")
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

    # --- Quality Assurance (Placeholder) ---
    logger.info("--- Starting Bandpass Calibration QA (Placeholder) ---")
    bandpass_qa_success = True # Assume success for now

    # --- NEW: Apply the Bandpass Table ---
    if bandpass_qa_success:
        logger.info("Applying bandpass calibration table to the MS...")
        try:
            applycal = CASA_IMPORTS['applycal']
            applycal(
                vis=ms_path,
                gaintable=[bp_table_path], # Apply ONLY the bandpass table
                interp=['nearest'],
                calwt=False,
                flagbackup=False
            )
            logger.info("CASA applycal task completed successfully.")
        except Exception as e:
            logger.error(f"Failed to apply bandpass table: {e}")
            return False

    logger.info("Bandpass Calibration step finished successfully.")
    return True

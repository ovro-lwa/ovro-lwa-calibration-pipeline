# delay_calibration.py
import os
import sys
import numpy as np
import logging
import shutil
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
import astropy.units as u

# Import pipeline utilities and config
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import pipeline_utils
import pipeline_config as config

# Initialize logger
logger = pipeline_utils.get_logger('DelayCal')

# CASA Imports
CASA_AVAILABLE = pipeline_utils.CASA_AVAILABLE
CASA_IMPORTS = pipeline_utils.CASA_IMPORTS

# ==============================================================================
# === Main Logic ===
# ==============================================================================
def run_delay_calibration(context):
    """
    Performs delay (K) calibration for diagnostic purposes.
    """
    if not CASA_AVAILABLE:
        logger.error("CASA environment not available. Cannot run delay calibration.")
        return False

    ms_path = context.get('concatenated_ms_path')
    cal_dir = context.get('cal_tables_dir')
    qa_dir = context.get('qa_dir')
    
    # --- Execute Delay Calibration ---
    delay_table_path = os.path.join(cal_dir, "cal.K1")
    if os.path.exists(delay_table_path):
        shutil.rmtree(delay_table_path)
    
    # ... (code to determine uv_range and spw_selection)

    try:
        gaincal = CASA_IMPORTS['gaincal']
        gaincal(
            vis=ms_path,
            caltable=delay_table_path,
            gaintype='K',
            # ... other gaincal parameters
        )
        logger.info("CASA gaincal task completed.")
    except Exception as e:
        logger.error(f"CASA task 'gaincal' failed: {e}")
        return False

    # --- Quality Assurance and Diagnostics ---
    logger.info("--- Starting Delay Calibration QA and Diagnostics ---")
    cal_data = pipeline_utils.read_cal_table(delay_table_path)
    if not cal_data: return False

    # Find the appropriate reference table based on LST (NEW)
    lst_hour_int = context['obs_info']['lst_hour_int']
    ref_table_path = pipeline_utils.find_nearest_reference_table('delay', lst_hour_int)
    
    # Run diagnostics
    # problematic_ants, ref_cal_data = pipeline_utils.compare_and_diagnose_delays(...)
    
    # Generate the plot
    delay_plot_path = os.path.join(qa_dir, "delay_calibration.png")
    # pipeline_utils.plot_delays(...)

    logger.info("Delay calibration and diagnostics completed.")
    return True

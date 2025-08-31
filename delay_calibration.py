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

# CASA Imports handled by pipeline_utils
CASA_AVAILABLE = pipeline_utils.CASA_AVAILABLE
CASA_IMPORTS = pipeline_utils.CASA_IMPORTS

# ==============================================================================
# === Helper Functions ===
# ==============================================================================

def determine_uv_range(context):
    """
    Determines the uvrange based on the sources used in the sky model.
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

def determine_spw_selection(ms_path, min_freq_mhz):
    """
    Determines the SPW selection string for CASA tasks based on a minimum frequency.
    It assumes a single SPW (0) after mstransform.
    """
    logger.info(f"Determining SPW selection for frequencies >= {min_freq_mhz} MHz.")
    spw_info = pipeline_utils.get_ms_frequency_metadata(ms_path)
    if not spw_info or 0 not in spw_info:
        logger.error("Could not retrieve metadata for SPW 0 from the MS.")
        return None

    freqs_hz = spw_info[0]['freqs_hz']
    min_freq_hz = min_freq_mhz * 1e6

    # Find the first channel index that is at or above the minimum frequency
    valid_channels = np.where(freqs_hz >= min_freq_hz)[0]
    if len(valid_channels) == 0:
        logger.error(f"No channels found above the minimum frequency of {min_freq_mhz} MHz.")
        return None

    start_chan = valid_channels[0]
    num_chans = len(freqs_hz)
    
    # CASA's SPW selection format is 'spw_id:start_chan~end_chan'
    spw_selection_str = f"0:{start_chan}~{num_chans - 1}"
    logger.info(f"Generated SPW selection string: '{spw_selection_str}'")
    return spw_selection_str

def check_calibrator_elevation(context):
    """Checks if major calibrators (CygA, CasA) are at sufficient elevation."""
    # (v) Calibrator Elevation Warning
    
    # Coordinates for CygA and CasA
    cyga_info = config.PRIMARY_SOURCES["CygA"]
    casa_info = config.PRIMARY_SOURCES["CasA"]
    cyga_coord = SkyCoord(cyga_info['ra'], cyga_info['dec'], unit=(u.hourangle, u.deg))
    casa_coord = SkyCoord(casa_info['ra'], casa_info['dec'], unit=(u.hourangle, u.deg))
    
    time_dt = context['obs_info']['mid_time_dt']
    
    # Convert datetime back to Astropy Time object for calculation
    time_obj = Time(time_dt, scale='utc', location=config.OVRO_LWA_LOCATION)

    # Calculate AltAz
    altaz_frame = AltAz(obstime=time_obj, location=config.OVRO_LWA_LOCATION)
    
    cyga_aa = cyga_coord.transform_to(altaz_frame)
    casa_aa = casa_coord.transform_to(altaz_frame)

    cyga_el = cyga_aa.alt.deg
    casa_el = casa_aa.alt.deg
    
    logger.info(f"Elevation at midpoint: CygA={cyga_el:.1f} deg, CasA={casa_el:.1f} deg.")
    
    if cyga_el < config.DELAY_MIN_ELEVATION_DEG and casa_el < config.DELAY_MIN_ELEVATION_DEG:
        logger.warning(f"WARNING: Neither Cygnus A nor Cassiopeia A are above {config.DELAY_MIN_ELEVATION_DEG} degrees elevation.")
        logger.warning("Delay calibration results might be suboptimal.")

# ==============================================================================
# === Main Logic ===
# ==============================================================================

def run_delay_calibration(context):
    """
    Performs delay (K) calibration for diagnostic purposes.
    """
    logger.info("Starting Delay Calibration (gaintype=K) - Diagnostic Only.")

    if not CASA_AVAILABLE:
        logger.error("CASA environment not available. Cannot run calibration tasks.")
        return False

    # (v) Check environmental conditions
    check_calibrator_elevation(context)

    ms_path = context.get('concatenated_ms_path')
    if not ms_path or not os.path.exists(ms_path):
        logger.error(f"MS not found: {ms_path}")
        return False

    # Define output paths
    tables_dir = context['tables_dir']
    qa_dir = context['qa_dir']
    obs_date_str = context['obs_info']['obs_date']
    lst_str = context['obs_info']['lst_hour']
    
    # Define standardized output names
    delay_table_name = f"calibration_{obs_date_str}_{lst_str}.K"
    delay_table_path = os.path.join(tables_dir, delay_table_name)
    delay_plot_path = os.path.join(qa_dir, f"QA_plot_delays_{obs_date_str}_{lst_str}.png")

    # (Cleanup remains the same)

    # 1. Determine Parameters
    uvrange_str = determine_uv_range(context)
    spw_selection_str = determine_spw_selection(ms_path, config.DELAY_CAL_MIN_FREQ_MHZ)
    
    if not spw_selection_str:
        logger.error("Delay calibration cannot proceed due to frequency constraints.")
        return False

    refant = config.CAL_REFANT

    # 2. Execute gaincal
    logger.info("Running CASA task 'gaincal' (gaintype=K)...")
    logger.info(f"  Ref Ant: {refant}")
    
    try:
        # Get the CASA task reference
        gaincal = CASA_IMPORTS['gaincal']
        
        # Execute the task
        gaincal(
            vis=ms_path,
            caltable=delay_table_path,
            gaintype='K',
            refant=refant,
            uvrange=uvrange_str,
            spw=spw_selection_str,
            solint='inf',
            minsnr=3.0,
            # Crucially, we do NOT apply any prior tables (like Bandpass)
            gaintable=[] 
        )
        
        logger.info("CASA gaincal task completed.")

    except Exception as e:
        logger.error(f"CASA task 'gaincal' failed: {e}")
        return False


    # 3. Quality Assurance and Diagnostics
    logger.info("--- Starting Delay Calibration QA and Diagnostics ---")

    cal_data = pipeline_utils.read_cal_table(delay_table_path)
    if not cal_data:
        logger.error("Failed to read back the generated delay table. Cannot perform QA.")
        return False

    # Run the comparison and diagnostics first to identify problematic antennas
    problematic_ants, ref_cal_data = pipeline_utils.compare_and_diagnose_delays(
        context,
        cal_data,
        config.REFERENCE_DELAY_TABLE_K,
        qa_dir
    )

    # Now generate the plot, passing in the reference data and problematic antennas
    plot_success = pipeline_utils.plot_delays(
        cal_data,
        delay_plot_path,
        ref_cal_data=ref_cal_data,
        problematic_ants=problematic_ants
    )

    if not plot_success:
        logger.warning("QA Plot generation failed, but calibration may have succeeded.")

    logger.info("Delay Calibration step finished successfully.")
    return True

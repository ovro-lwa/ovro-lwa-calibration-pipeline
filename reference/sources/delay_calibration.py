# delay_calibration.py
import os
import sys
import numpy as np
import logging
import shutil
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
import astropy.units as u
import warnings
import re # Imported for regex parsing in SPW selection

# Import pipeline utilities, config, and the QA module
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import pipeline_utils
import pipeline_config as config
import qa # Import the centralized QA module

# Initialize logger
logger = pipeline_utils.get_logger('DelayCal')

# CASA Imports (handled via pipeline_utils)
CASA_AVAILABLE = pipeline_utils.CASA_AVAILABLE
CASA_IMPORTS = pipeline_utils.CASA_IMPORTS

# Check for specific required task
if CASA_AVAILABLE and 'gaincal' in CASA_IMPORTS:
    gaincal = CASA_IMPORTS['gaincal']
else:
    # If gaincal isn't available, this module cannot function.
    # We log the error but allow the pipeline to continue as this step is diagnostic.
    logger.error("CASA environment or 'gaincal' task not available.")
    gaincal = None

# ==============================================================================
# === Helper Functions ===
# ==============================================================================

def calculate_source_elevations(context):
    """Calculates the elevation of modeled sources at the observation midpoint."""
    # Access obs_mid_time correctly through obs_info
    obs_midpoint = context.get('obs_info', {}).get('obs_mid_time')
    modeled_sources = context.get('model_sources', [])
    elevations = {}

    if not obs_midpoint: return elevations

    altaz_frame = AltAz(obstime=obs_midpoint, location=config.OVRO_LOCATION)

    for source in modeled_sources:
        # FIX: Use the PRIMARY_SOURCES dictionary which contains the SkyCoord objects
        details = config.PRIMARY_SOURCES.get(source)
        if details and 'skycoord' in details:
            coords = details['skycoord']
            # Suppress ERFA warnings during transformation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="astropy._erfa")
                source_altaz = coords.transform_to(altaz_frame)
            elevations[source] = source_altaz.alt.deg
        else:
            logger.warning(f"Coordinates not found in config.PRIMARY_SOURCES for modeled source: {source}")

    return elevations

def determine_spw_selection_metadata_fallback(ms_path, min_freq_mhz):
    """Fallback method using msmetadata if context analysis is unavailable."""
    # (This fallback remains in case the primary method fails unexpectedly or --input_ms is used)
    logger.warning("Attempting fallback to msmetadata for SPW selection.")
    metadata = pipeline_utils.get_ms_metadata(ms_path)

    if metadata:
        selected_spws = []
        for spw_id, info in metadata['spw_info'].items():
            # Check if the maximum frequency of the SPW is above the threshold
            if info['max_freq_mhz'] >= min_freq_mhz:
                selected_spws.append(str(spw_id))

        if selected_spws:
             # Sort for consistency
            spw_selection_str = ",".join(sorted(selected_spws, key=int))
            logger.info(f"Generated SPW selection string (from metadata fallback): '{spw_selection_str}'")
            return spw_selection_str

    # Final fallback if metadata also fails
    logger.error("Could not retrieve MS metadata. Cannot apply frequency cut reliably.")
    logger.warning("Defaulting to all SPWs (Calibration might be poor due to low-frequency RFI).")
    return ''

def determine_spw_selection(ms_path, context, min_freq_mhz=config.DELAY_CAL_MIN_FREQ_MHZ):
    """
    Determines the SPW selection string based on the initial file analysis structure.
    This method avoids relying on msmetadata, addressing the observed errors.
    """
    logger.info(f"Determining SPW selection for frequencies >= {min_freq_mhz} MHz using file structure.")

    # 1. Access the initial file analysis from the context
    obs_info = context.get('obs_info', {})
    # unique_freqs_detected stores the frequency strings (e.g., '13MHz') identified during initialization
    unique_freqs_detected = obs_info.get('unique_freqs_detected')

    if not unique_freqs_detected:
        # Fallback if the context information is missing
        logger.warning("File structure information (unique frequencies) missing in context.")
        return determine_spw_selection_metadata_fallback(ms_path, min_freq_mhz)

    # 2. Parse the frequency strings and sort them
    freq_mapping = []
    for freq_str in unique_freqs_detected:
        match = re.match(r'(\d+)MHz', freq_str)
        if match:
            # Use the integer value (e.g., 41) for comparison
            freq_val = int(match.group(1))
            freq_mapping.append((freq_val, freq_str))
        else:
            logger.warning(f"Could not parse frequency string: {freq_str}")

    # Sort by frequency value. Since input files are sorted before concat, this defines the SPW order.
    freq_mapping.sort(key=lambda x: x[0])

    # 3. Determine SPW IDs based on the sorted order
    selected_spw_ids = []
    for spw_id, (freq_val, freq_str) in enumerate(freq_mapping):
        # Check if the approximate frequency is above the threshold
        if freq_val >= min_freq_mhz:
            selected_spw_ids.append(spw_id)
            logger.debug(f"  SPW {spw_id} ({freq_str}) selected.")
        else:
            logger.debug(f"  SPW {spw_id} ({freq_str}) excluded.")

    # 4. Generate the selection string
    if selected_spw_ids:
         # Use CASA range syntax for compactness if the selection is contiguous (e.g., '6~15')
        start_id = selected_spw_ids[0]
        end_id = selected_spw_ids[-1]
        
        # Check if the list is contiguous
        if len(selected_spw_ids) == (end_id - start_id + 1):
             spw_selection_str = f"{start_id}~{end_id}"
        else:
             # Use comma-separated list if non-contiguous (e.g., if 50MHz was missing)
             spw_selection_str = ",".join(map(str, selected_spw_ids))

        logger.info(f"Generated SPW selection string (from file structure): '{spw_selection_str}'")
        return spw_selection_str
    else:
        logger.warning(f"No SPWs found above {min_freq_mhz} MHz (from file structure). Defaulting to all SPWs.")
        return ''


# ==============================================================================
# === Main Calibration Function ===
# ==============================================================================

def run_delay_calibration(context):
    """
    Performs delay calibration (gaintype=K) as a diagnostic step.
    This step is non-critical for the final calibration but crucial for QA.
    """
    logger.info("Starting Delay Calibration (gaintype=K) - Diagnostic Only.")

    if not gaincal:
        # Return True as it's non-critical for the pipeline flow
        return True

    ms_path = context.get('concat_ms')
    if not ms_path or not os.path.exists(ms_path):
        logger.error(f"Input MS not found: {ms_path}")
        return True # Non-critical

    # Retrieve observation identifiers from context (using standardized keys)
    try:
        tables_dir = context['tables_dir']
        obs_date_str = context['obs_info']['obs_date']
        lst_hour_str = context['obs_info']['lst_hour']
    except KeyError as e:
        logger.error(f"Missing essential context info: {e}. Cannot generate table names.")
        return True # Non-critical

    # Define output calibration table path
    delay_table_name = f"diagnostic_{obs_date_str}_{lst_hour_str}.K"
    delay_table_path = os.path.join(tables_dir, delay_table_name)

    # Ensure a clean start
    if os.path.exists(delay_table_path):
        logger.info(f"Removing existing delay table: {delay_table_path}")
        shutil.rmtree(delay_table_path, ignore_errors=True)

    # --- Calibration Parameters Setup ---
    refant = config.CAL_REFANT

    # Log elevations (Diagnostic info)
    elevations = calculate_source_elevations(context)
    if elevations:
        elevation_logs = [f"{src}={elev:.1f} deg" for src, elev in elevations.items()]
        logger.info(f"Elevation at midpoint: {', '.join(elevation_logs)}.")
    else:
        if not context.get('model_sources'):
             logger.warning("Sky model is empty. Delay calibration will likely fail or be meaningless.")
        else:
             logger.info("Elevations could not be calculated for modeled sources.")


    # Determine UV Range (using the centralized utility function)
    uvrange_str = pipeline_utils.determine_calibration_uv_range(context)


    # Determine SPW selection
    try:
        spw_selection = determine_spw_selection(ms_path, context)
    except Exception:
        logger.error("Failed during SPW selection determination. Defaulting to all SPWs.", exc_info=True)
        spw_selection = ''

    # --- Run CASA task 'gaincal' ---
    logger.info("Running CASA task 'gaincal' (gaintype=K)...")
    # Added Scan: 1 to the log message
    logger.info(f"  Ref Ant: {refant}, UVRange: {uvrange_str}, SPW Selection: {spw_selection or 'All'}, Scan: 1")

    try:
        # The bandpass calibration has already been applied to the CORRECTED_DATA column 
        # in the previous step (if successful). Applying it again here is redundant and slow.
        gaintables = []
        logger.info("Solving delays using CORRECTED_DATA (Bandpass assumed applied if available).")

        # FIX: Use only the first scan (integration) for delay calibration to speed up the process.
        logger.info("Using only the first scan (scan='1') for fast delay solution.")

        gaincal(
            vis=ms_path,
            caltable=delay_table_path,
            gaintype='K',
            scan='1',                 # Select only the first scan
            solint='inf',             # Average over the duration of the selected scan
            refant=refant,
            uvrange=uvrange_str,
            spw=spw_selection,
            minsnr=config.DELAY_CAL_MIN_SNR,
            # combine='scan' is generally not needed when scan selection is specific
            gaintable=gaintables
        )
        logger.info("CASA gaincal task completed.")

        # Store the path in the context
        context['calibration_tables']['delay'] = delay_table_path

        # --- Run QA and Diagnostics ---
        logger.info("--- Starting Delay Calibration QA and Diagnostics ---")
        
        # Determine the dynamic reference path (using improved logic in config)
        reference_delay_table = config.get_reference_table_path('delay', lst_hour_str)
        
        # Call the centralized QA function from the qa module
        qa_success = qa.analyze_and_plot_delays(context, delay_table_path, reference_delay_table)

        if not qa_success:
             logger.warning("Delay QA analysis reported issues or failed to complete.")

        # The calibration step succeeded if gaincal finished and QA ran (even if QA failed)
        return True

    except Exception as e:
        logger.error(f"An error occurred during delay calibration: {e}", exc_info=True)

        # Clean up failed calibration table
        if os.path.exists(delay_table_path):
            try:
                shutil.rmtree(delay_table_path)
            except Exception as cleanup_e:
                logger.warning(f"Failed to clean up failed delay table: {cleanup_e}")

        # Return True as it's non-critical for the pipeline flow
        logger.warning("Delay calibration step failed but is non-critical. Continuing pipeline.")
        return True

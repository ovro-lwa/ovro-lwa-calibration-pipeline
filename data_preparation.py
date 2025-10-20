# data_preparation.py

import os
import shutil
import sys
import numpy as np
import json
import logging
from tqdm import tqdm

# Astropy (not strictly required here but kept for consistency)
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord

# Import pipeline utilities and config
# Ensure the script directory is in the path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    
import pipeline_utils
import pipeline_config as config

# Initialize logger for this module
logger = pipeline_utils.get_logger('DataPrep')

# CASA Imports handled by pipeline_utils
CASA_AVAILABLE = pipeline_utils.CASA_AVAILABLE
CASA_IMPORTS = pipeline_utils.CASA_IMPORTS

# Get table tools reference if available
pt = CASA_IMPORTS.get('table_tools_module')


# ==============================================================================
# === Helper Functions ===
# ==============================================================================

def fix_ms_field_id(msfile):
    """Fixes FIELD_ID in an MS by setting all entries to 0."""
    # Use the centralized check from pipeline_utils
    table_tools_available = CASA_IMPORTS.get('table_tools_available', False)
    pt = CASA_IMPORTS.get('table_tools_module')

    if not table_tools_available or pt is None:
        logger.error("casatables/pyrap.tables not available. Cannot fix FIELD_ID.")
        # Raise an error as this is a critical step
        raise ImportError("MS table manipulation library missing.")

    logger.info(f"Fixing FIELD_ID for MS: {msfile}")
    try:
        # Open the table in read-write mode
        t = pt.table(msfile, readonly=False)
        # Get the FIELD_ID column
        fid = t.getcol('FIELD_ID')
        # Create a new array of zeros with the same shape
        fid_new = np.zeros(fid.shape, dtype=int)
        # Put the new column back into the table
        t.putcol('FIELD_ID', fid_new)
        t.close()
        logger.info("FIELD_ID column successfully set to 0.")
    except Exception as e:
        logger.exception(f"Failed to fix FIELD_ID for {msfile}")
        raise # Propagate the error

def apply_flags_with_casa(msfile, antenna_ids):
    """Applies flags using standard CASA flagdata task."""
    
    # Check centralized CASA availability
    if not CASA_AVAILABLE:
         logger.error("CASA not available for flagging (flagdata).")
         raise EnvironmentError("CASA environment missing for required flagging step.")

    # Ensure the input is a list or array
    if not isinstance(antenna_ids, (list, np.ndarray)):
        logger.error("Antenna IDs must be provided as a list or numpy array.")
        raise TypeError("Invalid type for antenna_ids.")

    if not antenna_ids:
        logger.info("No antennas selected for flagging.")
        return

    try:
        # Convert antenna IDs (assumed to be 0-indexed correlator numbers) to a CASA selection string
        # Ensure they are strings for the join operation
        antenna_selection = ",".join(map(str, antenna_ids))
        
        logger.info(f"Applying flags using CASA flagdata.")
        logger.info(f"Total antennas being flagged: {len(antenna_ids)}")
        logger.debug(f"Flagging selection string: {antenna_selection}")

        # Use the imported reference for flagdata
        flagdata = CASA_IMPORTS.get('flagdata')
        if flagdata:
            flagdata(vis=msfile, mode='manual', antenna=antenna_selection)
            logger.info("CASA flagdata task completed.")
        else:
            logger.error("CASA task 'flagdata' not available.")
            raise EnvironmentError("CASA 'flagdata' task missing.")
        
    except Exception as e:
        logger.exception(f"Failed to apply flags using CASA flagdata.")
        raise # Propagate error

# ==============================================================================
# === Flagging Sub-module ===
# ==============================================================================

def identify_bad_antennas_mnc(context):
    """
    Identifies bad antennas using the external MNC helper script via a conda environment.
    Returns a list of antenna IDs on success (even if empty), or None if the script failed or was not found.
    """
    logger.info("Identifying bad antennas using MNC helper script...")
    
    script_dir = context['script_dir']
    obs_info = context['obs_info']
    
    # Get the midpoint time from the context (Astropy Time object)
    midpoint_time = obs_info['obs_mid_time']
    mjd_time_str = str(midpoint_time.mjd)

    mnc_helper_script_path = config.get_mnc_helper_path(script_dir)

    if not os.path.exists(mnc_helper_script_path):
        logger.warning(f"MNC Helper script not found at {mnc_helper_script_path}. Skipping MNC identification.")
        return None # Indicate that the identification did not run successfully

    # Construct the command to run the helper script within the specified conda environment
    helper_cmd = [
        'conda', 'run', '--no-capture-output', '-n', config.CONDA_ENV_MNC,
        'python', mnc_helper_script_path, mjd_time_str
    ]
    
    # Execute the command and capture its output
    # run_command returns the stdout string on success (exit code 0), or None on failure (non-zero exit code)
    stdout_output = pipeline_utils.run_command(
        helper_cmd,
        task_name="Bad Antenna Identification (MNC)",
        return_output=True
    )

    # Parse the output
    if stdout_output is not None:
        # The command executed successfully (exit code 0). Now parse the JSON.
        json_output = None
        # Look for the JSON output line in the script's stdout
        # Handle potential differences in line endings (\n vs \r\n)
        for line in stdout_output.strip().splitlines():
            if line.strip().startswith('{') and line.strip().endswith('}'):
                try:
                    json_output = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        
        if json_output:
            # Extract the list of bad correlator numbers (antenna IDs)
            bad_correlator_numbers = json_output.get("bad_correlator_numbers", [])
            logger.info(f"MNC Helper script identified {len(bad_correlator_numbers)} bad antennas.")
            return bad_correlator_numbers
        else:
            # This case means the script ran but didn't output valid JSON. Treat as 0 antennas found.
            logger.warning("MNC script ran successfully but could not find or decode valid JSON output. Assuming 0 bad antennas.")
            return []
    else:
         # The command failed (non-zero exit code).
         logger.error("MNC bad antenna helper script failed during execution.")
         return None


def run_flagging_steps(msfile_path, context, skip_mnc_flagging=False):
    """Runs the flagging steps (Antenna health and AOFlagger) on the provided MS."""
    logger.info("--- Starting Flagging Steps (Antenna Health and AOFlagger) ---")

    try:
        # --- 1. Antenna Health Flagging (MNC + Config Logic) ---
        
        if not skip_mnc_flagging:
            # Run the MNC identification
            # Returns list (potentially empty) on success, None on failure.
            mnc_bad_antennas = identify_bad_antennas_mnc(context)
            
            antennas_to_flag = set()

            # Apply the conditional logic for determining the final flag list
            if mnc_bad_antennas is None:
                # MNC script failed or was not found. Use the standard additional list as a safety measure.
                logger.warning("MNC identification failed or skipped. Falling back to ADDITIONAL_BAD_ANTENNAS.")
                antennas_to_flag.update(config.ADDITIONAL_BAD_ANTENNAS)
            
            elif len(mnc_bad_antennas) == 0:
                # MNC script succeeded but found 0 bad antennas. Use the expanded fallback list exclusively.
                logger.info("MNC script returned 0 bad antennas. Applying FALLBACK_BAD_ANTENNAS.")
                antennas_to_flag.update(config.FALLBACK_BAD_ANTENNAS)

            else:
                # MNC script succeeded and found bad antennas. Combine with the standard additional list.
                logger.info("Combining MNC results with ADDITIONAL_BAD_ANTENNAS.")
                antennas_to_flag.update(mnc_bad_antennas)
                antennas_to_flag.update(config.ADDITIONAL_BAD_ANTENNAS)

            # Apply the determined flags
            if antennas_to_flag:
                # Convert the set to a sorted list for consistent application
                apply_flags_with_casa(msfile_path, sorted(list(antennas_to_flag)))
            else:
                logger.info("No antennas met the criteria for health flagging.")
        
        else:
            logger.warning("Skipping MNC-based antenna health flagging as requested by user.")


        # --- 2. Baseline-based flagging (AOFlagger) ---
        logger.info("Performing baseline-based flagging with AOFlagger...")
        
        # Construct the AOFlagger command (as a list for robust execution)
        aoflagger_cmd = [
            config.AOFLAGGER_EXECUTABLE,
            '-strategy',
            config.AOFLAGGER_STRATEGY_PATH,
            msfile_path
        ]
        
        # Execute AOFlagger
        success = pipeline_utils.run_command(aoflagger_cmd, task_name="AOFlagger")
        
        if not success:
            logger.error("AOFlagger execution failed.")
            return False

        logger.info("--- Flagging Steps Completed ---")
        return True

    except Exception as e:
        logger.error(f"An unexpected error occurred during the flagging steps: {e}", exc_info=True)
        return False

# ==============================================================================
# === Main Data Preparation Logic ===
# ==============================================================================


def run_data_preparation(context, force_flagging_only=False, skip_mnc_flagging=False):
    """
    Runs the full data preparation sequence: concatenation, field ID fixing, phase centering, and flagging.
    If force_flagging_only is True, it assumes concatenation and field ID fixing are done.
    """
    logger.info("Data Preparation: Starting.")
    if not CASA_AVAILABLE:
        logger.error("CASA environment not available. Cannot perform required Data Preparation steps.")
        return False

    final_ms_path = context['concat_ms']
    ms_files = context['obs_info']['ms_files']

    if not force_flagging_only:
        # Step 1: Concatenate
        logger.info(f"Concatenating {len(ms_files)} MS files into {os.path.basename(final_ms_path)}...")
        try:
            concat = CASA_IMPORTS.get('concat')
            if not concat:
                raise EnvironmentError("CASA task 'concat' not available.")
            # Execute the CASA concat task
            concat(vis=ms_files, concatvis=final_ms_path, dirtol='', copypointing=False)
            logger.info("CASA concat task finished.")
        except Exception:
            logger.exception("Failed to concatenate MS files.")
            return False

        # Step 2: Fix Field ID
        try:
            fix_ms_field_id(final_ms_path)
        except Exception:
            logger.exception("Failed during FIELD_ID fixing step.")
            return False
            
        # Step 2.5: Set Phase Center
        logger.info("Setting phase center based on CAL_PHASE_CENTER config...")
        try:
            if not pipeline_utils.set_phase_center(final_ms_path, context):
                logger.error("Failed to set phase center. Halting data preparation.")
                return False
            logger.info("Phase center set successfully.")
        except Exception as e:
            logger.exception("An unexpected error occurred during phase center correction.")
            return False
            
    else:
        logger.info("Data Preparation: Running in flagging-only mode (skipping concatenation/fixing/re-phasing).")
        # Ensure the MS path provided actually exists if we skip concatenation
        if not os.path.exists(final_ms_path):
            logger.error(f"MS file not found for flagging-only mode: {final_ms_path}")
            return False

    # Step 3: Flagging
    flagging_success = run_flagging_steps(final_ms_path, context, skip_mnc_flagging=skip_mnc_flagging)
    
    # CRITICAL: Check if flagging succeeded before continuing
    if not flagging_success:
        logger.error("Flagging sub-step failed. Data preparation cannot continue.")
        return False

    logger.info("Data Preparation: All steps completed.")
    return True

if __name__ == "__main__":
    # This block allows the script to be run standalone for testing if needed, 
    # but primarily it's intended to be imported.
    print("This script is designed to be called by the master pipeline.")

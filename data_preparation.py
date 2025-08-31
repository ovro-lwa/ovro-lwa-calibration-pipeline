# data_preparation.py

import os
import shutil
import sys
import numpy as np
import json
import logging
from tqdm import tqdm

# Astropy
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord

# Import pipeline utilities and config
import pipeline_utils
import pipeline_config as config

# Initialize logger for this module
logger = pipeline_utils.get_logger('DataPrep')

# CASA Imports handled by pipeline_utils
CASA_AVAILABLE = pipeline_utils.CASA_AVAILABLE
TABLE_TOOLS_AVAILABLE = pipeline_utils.TABLE_TOOLS_AVAILABLE
CASA_IMPORTS = pipeline_utils.CASA_IMPORTS

# Get table tools reference if available
pt = CASA_IMPORTS.get('table_tools')


# --- ORCA / Anthealth Imports (Attempt direct import) ---
# We check for availability but rely mostly on the helper script for identification.
_orca_available = False
try:
    # We only need the flagging wrappers here if available
    from orca.transform.flagging import flag_ants, flag_with_aoflagger
    _orca_available = True
except ImportError as e:
    _orca_available = False
    # Use print here as the logger might not be fully initialized yet during import time
    # Check if stderr is available before writing
    if sys.stderr:
        print(f"\nINFO: ORCA/MNC modules not available in the current environment. Fallback methods will be used.", file=sys.stderr)


# ==============================================================================
# === Helper Functions ===
# ==============================================================================

def fix_ms_field_id(msfile):
    """Fixes FIELD_ID in an MS."""
    # Check the specific availability flag for table tools
    if not TABLE_TOOLS_AVAILABLE or pt is None:
        logger.error("casatables/pyrap.tables not available. Cannot fix FIELD_ID.")
        # Log the specific error if available
        if 'table_tools_error' in CASA_IMPORTS:
            logger.debug(f"Table tools import error: {CASA_IMPORTS['table_tools_error']}")
        raise ImportError("MS table manipulation library missing.")
        
    logger.info(f"Fixing FIELD_ID for MS: {msfile}")
    try:
        # ack=False suppresses verbose output if using pyrap.tables
        # Use a context manager to ensure the table is closed properly
        with pt.table(msfile, readonly=False, ack=False) as t:
            fid = t.getcol('FIELD_ID')
            if np.all(fid == 0):
                logger.info("FIELD_ID already set to 0. No changes needed.")
                return

            t.putcol('FIELD_ID', np.zeros(fid.shape, dtype=int))
            t.flush()
            logger.info(f"Set all FIELD_ID to 0.")
    except Exception as e:
        logger.exception(f"Failed to fix FIELD_ID for {msfile}")
        raise

def apply_flags_with_casa(msfile, bad_corr_nums):
    """Fallback function to apply flags using standard CASA flagdata."""
    logger.info(f"Applying flags using CASA flagdata (Fallback).")
    
    # Check centralized CASA availability
    if not CASA_AVAILABLE:
         logger.error("CASA not available for flagging (flagdata).")
         raise EnvironmentError("CASA environment missing for required flagging step.")

    try:
        # Assuming correlator numbers map directly to CASA antenna indices (0-indexed).
        antenna_selection = ",".join(map(str, bad_corr_nums))
        
        if not antenna_selection:
            logger.info("No antennas selected for flagging.")
            return

        logger.info(f"Flagging antennas: {antenna_selection}")

        # Use the imported reference for flagdata (CASA tasks are imported directly)
        flagdata = CASA_IMPORTS['flagdata']
        flagdata(vis=msfile, mode='manual', antenna=antenna_selection)
        
    except Exception as e:
        logger.exception(f"Failed to apply flags using CASA flagdata.")
        raise # Propagate error

# ==============================================================================
# === Flagging Sub-module (Refactored) ===
# ==============================================================================

# (run_flagging_steps remains the same as previous versions)
def run_flagging_steps(msfile_path, context):
    """Runs only the flagging steps (Antenna health and AOFlagger) on the provided MS."""
    logger.info("--- Starting Flagging Steps (Antenna Health and AOFlagger) ---")
    
    script_dir = context['script_dir']
    obs_info = context['obs_info']

    # --- 6. Identify and flag bad antennas (ORCA/MNC Helper Script) ---
    logger.info("Identifying bad antennas using MNC helper script...")
    
    # Calculate MJD time
    mjd_time_astropy = Time(obs_info['mid_time_dt'], format='datetime', scale='utc')
    mjd_time_float = mjd_time_astropy.mjd
    # Ensure it is passed as a string representation of the float
    mjd_time_str = str(mjd_time_float)

    bad_correlator_numbers = []
    identification_success = False

    # We rely on the helper script now.
    MNC_HELPER_SCRIPT = config.get_mnc_helper_path(script_dir)

    if os.path.exists(MNC_HELPER_SCRIPT):
        helper_cmd = [
            'conda', 'run', '--no-capture-output', '-n', config.CONDA_ENV_MNC, 
            'python', MNC_HELPER_SCRIPT, mjd_time_str # Pass MJD as string
        ]
        
        try:
            # Use the robust runner and capture the output
            stdout_output = pipeline_utils.run_external_command(
                helper_cmd,
                description=f"MNC Helper Script (Env: {config.CONDA_ENV_MNC})",
                logger_name=logger.name,
                return_output=True
            )
            
            # Parse the JSON output
            json_output = None
            for line in stdout_output.strip().split('\n'):
                try:
                    if line.startswith('{') and line.endswith('}'):
                         json_data = json.loads(line)
                         if isinstance(json_data, dict) and "bad_correlator_numbers" in json_data:
                            json_output = json_data
                            break
                except json.JSONDecodeError:
                    continue

            if json_output:
                bad_correlator_numbers = json_output.get("bad_correlator_numbers", [])
                identification_success = True
                data_ts = json_output.get("data_timestamp_mjd")
                logger.info(f"Helper script used data timestamp MJD: {data_ts}")
            else:
                logger.error("No valid JSON output found from helper script despite successful execution (check detailed logs).")

        except RuntimeError:
            pass # Error logged by runner
        except Exception as e:
            logger.exception(f"Helper script failed during execution setup or output parsing.")
    else:
        logger.warning(f"MNC Helper script not found at {MNC_HELPER_SCRIPT}. Skipping antenna health flagging.")

    # Apply the flags
    if identification_success:
        logger.info(f"Identified {len(bad_correlator_numbers)} bad antennas.")

        if len(bad_correlator_numbers) > 0:
            flag_fraction = len(bad_correlator_numbers) / config.TOTAL_ANTENNAS
            logger.info(f"Flagging fraction: {flag_fraction*100:.1f}%")
            if flag_fraction > config.MAX_BAD_ANTENNA_FRACTION:
                logger.warning(f"WARNING: More than {config.MAX_BAD_ANTENNA_FRACTION*100}% of antennas are flagged.")
            
            # Apply flags: Prefer ORCA wrapper, otherwise use CASA flagdata
            # This might raise EnvironmentError if both ORCA and CASA are missing.
            try:
                if _orca_available:
                    logger.info(f"Flagging using orca.transform.flagging.flag_ants")
                    flag_ants(ms=msfile_path, antennas=bad_correlator_numbers)
                else:
                    apply_flags_with_casa(msfile_path, bad_correlator_numbers)
            except Exception as e:
                if _orca_available:
                     logger.warning(f"ORCA flag_ants failed: {e}. Attempting CASA fallback.")
                     try:
                         apply_flags_with_casa(msfile_path, bad_correlator_numbers)
                     except Exception as e_fallback:
                          # Both failed, raise the error to halt the pipeline if critical
                         logger.error(f"CASA fallback for flagging also failed: {e_fallback}")
                         raise
                else:
                    # If ORCA wasn't available and CASA failed (or wasn't available), raise the error.
                    logger.error(f"Flagging failed: {e}.")
                    raise
                
    elif MNC_HELPER_SCRIPT and os.path.exists(MNC_HELPER_SCRIPT):
        # Only log error if the script was supposed to run but failed
        logger.error("Identification of bad antennas failed. No antennas flagged.")


    # --- 7. Baseline-based flagging (AOFlagger) ---
    logger.info("Performing baseline-based flagging with AOFlagger...")
    
    aoflagger_success = False
    
    # Method 1: Prioritize ORCA wrapper (if available)
    if _orca_available:
        try:
            logger.info(f"Calling orca.transform.flagging.flag_with_aoflagger...")
            # Note: The ORCA wrapper handles its own execution internally
            flag_with_aoflagger(
                ms=msfile_path,
                strategy=config.AOFALAGGER_STRATEGY_PATH,
                n_threads=1 # Using 1 thread for stability
            )
            aoflagger_success = True
            logger.info("ORCA AOFlagger wrapper finished.")
        except Exception as e:
            logger.warning(f"ORCA wrapper for AOFlagger failed: {e}. Attempting fallback.")

    # Method 2: Fallback (Direct execution using the robust runner)
    if not aoflagger_success:
        logger.info("Attempting direct AOFlagger execution.")
        aoflagger_cmd = [
            config.AOFLAGGER_PATH,
            '-strategy',
            config.AOFALAGGER_STRATEGY_PATH,
            msfile_path
        ]
        try:
            pipeline_utils.run_external_command(
                aoflagger_cmd,
                description="AOFlagger (Direct Execution)",
                logger_name=logger.name
            )
            aoflagger_success = True
        except RuntimeError:
            # Error already logged by the runner
            pass

    if not aoflagger_success:
        logger.error("All AOFlagger methods failed. Proceeding without baseline flagging.")

    logger.info("--- Flagging Steps Completed ---")
    return True

# ==============================================================================
# === Main Data Preparation Logic ===
# ==============================================================================

def run_data_preparation(context, force_flagging_only=False):
    """
    Runs the data preparation steps.
    If force_flagging_only=True, it assumes concatenation/transform is done 
    and only runs flagging on the context['concatenated_ms_path'].
    """
    logger.info(f"Data Preparation: Starting.")
    
    # Check CASA availability early if we are not just running flagging (which might rely on ORCA)
    if not CASA_AVAILABLE and not force_flagging_only:
        logger.error("CASA environment not available. Cannot perform required Data Preparation steps (concat, mstransform).")
        return False

    if not _orca_available:
        logger.info("ORCA/MNC modules not loaded. Fallback methods (CASA) will be used for flagging application.")

    # Handle Flagging Only Mode (NEW for efficient rerunning)
    if force_flagging_only:
        ms_path = context.get('concatenated_ms_path')
        if not ms_path or not os.path.exists(ms_path):
            logger.error("force_flagging_only=True, but no valid MS path found in context.")
            return False
        logger.info(f"Running in FLAGGING ONLY mode on {os.path.basename(ms_path)}")
        
        # Important: Clear previous flags if re-running flagging on the same MS
        # This requires CASA
        if CASA_AVAILABLE:
            try:
                logger.info("Clearing existing flags before re-running flagging steps.")
                # Use imported reference (CASA tasks are imported directly)
                flagdata = CASA_IMPORTS['flagdata']
                flagdata(vis=ms_path, mode='unflag')
            except Exception as e:
                logger.warning(f"Could not clear existing flags: {e}. Proceeding, but results may be inconsistent.")
        else:
            # If CASA isn't available, we can't unflag.
            logger.warning("CASA not available. Cannot clear existing flags before re-running flagging. Results may be inconsistent.")

        # In this mode, we just run flagging and return.
        # Note: run_flagging_steps might fail if it relies on CASA for applying antenna health flags (if ORCA is not available).
        try:
            run_flagging_steps(ms_path, context)
            return True
        except (EnvironmentError, ImportError, Exception) as e:
            logger.error(f"Flagging steps failed, potentially due to missing environment components (ORCA or CASA fallback): {e}")
            return False

    # --- Standard Data Preparation Flow (Steps 1-5) ---

    obs_info = context['obs_info']
    output_ms_dir = context['ms_dir']
    ms_files = obs_info['ms_files']
    script_dir = context['script_dir']

    # Define output filenames
    obs_date_str = obs_info['obs_date']
    lst_str = obs_info['lst_hour']
    
    intermediate_concat_ms_path = os.path.join(output_ms_dir, f"intermediate_concat.ms")
    final_fullband_ms_path = os.path.join(output_ms_dir, f"fullband_calibration_{obs_date_str}_{lst_str}.ms")

    # --- 1. Time Span Check ---
    duration_minutes = obs_info['duration_minutes']
    if duration_minutes > config.MAX_OBSERVATION_DURATION_MINUTES:
        logger.warning(f"WARNING: Data spans longer than recommended maximum ({config.MAX_OBSERVATION_DURATION_MINUTES} min).")

    # --- 2. Concatenate ---
    logger.info(f"Concatenating {len(ms_files)} MS files...")
        
    try:
        # Use imported references (CASA tasks)
        concat = CASA_IMPORTS['concat']
        # CASA tasks handle their own logging and progress updates.
        # Wrapping in tqdm here just shows task start/finish.
        with tqdm(total=1, desc="CASA Concatenation", unit="task") as pbar:
             # Ensure ms_files is a list of strings (paths)
             concat(vis=ms_files, concatvis=intermediate_concat_ms_path, dirtol='', copypointing=False)
             pbar.update(1)
        logger.info("CASA concat task finished.")
    except Exception as e:
        logger.exception(f"Failed to concatenate MS files.")
        return False

    # --- 3. Remove sub-band structure (mstransform) ---
    logger.info(f"Removing sub-band structure (combining SPWs) using mstransform...")
    try:
        # Use imported references (CASA tasks)
        mstransform = CASA_IMPORTS['mstransform']
        with tqdm(total=1, desc="CASA mstransform (Combine SPWs)", unit="task") as pbar:
            mstransform(
                vis=intermediate_concat_ms_path,
                outputvis=final_fullband_ms_path,
                combinespws=True,
                spw='',
                reindex=True,
                datacolumn='all',
                createmms=False,
            )
            pbar.update(1)

        logger.info("CASA mstransform task finished.")
        if os.path.exists(intermediate_concat_ms_path):
            logger.info("Cleaning up intermediate concatenated MS.")
            try:
                shutil.rmtree(intermediate_concat_ms_path)
            except OSError as e:
                logger.warning(f"Could not remove intermediate MS: {e}")

    except Exception as e:
        logger.exception(f"Failed to mstransform (combinespws) MS.")
        return False

    # --- 4. Fix FIELD_ID ---
    try:
        fix_ms_field_id(final_fullband_ms_path)
    except (ImportError, Exception):
        # Error already logged in the function (includes check for TABLE_TOOLS_AVAILABLE)
        return False

    # --- 5. Change phase center (chgcentre) ---
    logger.info(f"Setting phase center to zenith at midpoint using chgcentre...")
    try:
        mid_dt = obs_info['mid_time_dt']
        mid_time_astropy = Time(mid_dt, format='datetime', scale='utc', location=config.OVRO_LWA_LOCATION)
        
        # Calculate zenith coordinates
        zenith_altaz_frame = AltAz(obstime=mid_time_astropy, location=config.OVRO_LWA_LOCATION)
        zenith_icrs = SkyCoord(alt=90*u.deg, az=0*u.deg, frame=zenith_altaz_frame).icrs
        
        # CRITICAL FIX: Ensure RA and Dec are separate strings
        ra_str = zenith_icrs.ra.to_string(unit=u.hourangle, sep='hms', precision=3)
        dec_str = zenith_icrs.dec.to_string(unit=u.deg, sep='dms', precision=3)
        
        # CRITICAL FIX: Pass RA and Dec as separate arguments in the list
        chgcentre_cmd = [
            config.CHGCENTRE_PATH,
            final_fullband_ms_path,
            ra_str,
            dec_str
        ]
        
        # Use the robust runner
        pipeline_utils.run_external_command(
            chgcentre_cmd, 
            description=f"chgcentre (RA={ra_str}, Dec={dec_str})",
            logger_name=logger.name
        )
        # Success is logged by the runner
        
    except RuntimeError:
        # Error is already logged by the runner
        return False
    except Exception as e:
        logger.exception(f"Error during phase center calculation or execution setup.")
        return False

    # Update the context before calling flagging steps
    context['concatenated_ms_path'] = final_fullband_ms_path

    # --- Run Flagging (Steps 6-7) ---
    try:
        run_flagging_steps(final_fullband_ms_path, context)
    except (EnvironmentError, ImportError, Exception) as e:
        # This might happen if ORCA is missing AND CASA is missing/failed for the fallback flag application
        logger.error(f"Flagging steps failed due to environment issues: {e}. Halting preparation.")
        return False

    logger.info("Data Preparation: All steps completed.")
    
    # Context is already updated
    return True

# Standalone execution block (if needed for testing)
if __name__ == "__main__":
    print("This script is designed to be called by the master pipeline.")

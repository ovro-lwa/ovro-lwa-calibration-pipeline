# pipeline_utils.py
import os
import glob
import re
import sys
import logging
import subprocess
import shutil
from datetime import datetime, timedelta
from astropy.time import Time
# Import ICRS explicitly for use in imaging transformations
from astropy.coordinates import AltAz, SkyCoord, ICRS
import astropy.units as u
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
import h5py
import warnings
import math

# Import configuration first
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import pipeline_config as config

# Matplotlib configuration for non-interactive plotting
try:
    import matplotlib
    matplotlib.use('Agg') # Use a non-interactive backend
    import matplotlib.pyplot as plt
    _plotting_available = True
except ImportError:
    _plotting_available = False
    print("WARNING: Matplotlib not found. QA plotting will be disabled.", file=sys.stderr)


# ==============================================================================
# === CASA Availability Check (Centralized and Comprehensive) ===
# ==============================================================================

CASA_AVAILABLE = False
TABLE_TOOLS_AVAILABLE = False
CASA_IMPORTS = {}
CASA_TASKS_AVAILABLE = False

try:
    # 1. Check core casatools
    import casatools
    CASA_IMPORTS['casatools'] = casatools

    # 2. Define tool names, including the quantity/quanta fallback
    tool_names = ['measures', 'msmetadata', 'componentlist', 'table']

    # Determine the name for the quantity/quanta tool
    quantity_tool_name = None
    if hasattr(casatools, 'quantity'):
        quantity_tool_name = 'quantity'
    elif hasattr(casatools, 'quanta'):
        quantity_tool_name = 'quanta'

    if quantity_tool_name:
         tool_names.append(quantity_tool_name)
    else:
        # Raise an import error if critical tools are missing
        raise ImportError("Neither 'quantity' nor 'quanta' found in casatools.")


    # 3. Check specific tools by attempting instantiation
    for factory_name in tool_names:
        if hasattr(casatools, factory_name):
            factory_func = getattr(casatools, factory_name)
            # Instantiate tools for later use (optional, but verifies functionality)
            # We store the factory function itself in CASA_IMPORTS
            CASA_IMPORTS[factory_name] = factory_func

            if factory_name == 'table':
                 TABLE_TOOLS_AVAILABLE = True
        else:
            raise ImportError(f"CASA tool factory '{factory_name}' not found.")

    CASA_AVAILABLE = True

    # 4. Check casatasks (required for calibration/imaging QA)
    try:
        # We import essential tasks used across the pipeline
        from casatasks import concat, flagdata, bandpass, applycal, gaincal, ft, imfit, imstat
        CASA_IMPORTS['concat'] = concat
        CASA_IMPORTS['flagdata'] = flagdata
        CASA_IMPORTS['bandpass'] = bandpass
        CASA_IMPORTS['applycal'] = applycal
        CASA_IMPORTS['gaincal'] = gaincal
        CASA_IMPORTS['ft'] = ft
        CASA_IMPORTS['imfit'] = imfit
        CASA_IMPORTS['imstat'] = imstat
        CASA_TASKS_AVAILABLE = True
    except ImportError as e:
        print(f"WARNING: Could not import all required casatasks. Calibration/QA might fail. Error: {e}", file=sys.stderr)
        CASA_TASKS_AVAILABLE = False


except ImportError as e:
    CASA_AVAILABLE = False
    TABLE_TOOLS_AVAILABLE = False
    print(f"WARNING: CASA environment not fully initialized or found. Pipeline steps requiring CASA will fail. Error: {e}", file=sys.stderr)


# ==============================================================================
# === Logging Utilities ===
# ==============================================================================

def setup_logging(log_directory, log_filename='pipeline.log'):
    """
    Sets up the logging configuration for the pipeline.
    Creates a root logger ('OVRO_Pipeline') and configures handlers.
    """
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_filepath = os.path.join(log_directory, log_filename)

    # Get the root pipeline logger
    logger = logging.getLogger('OVRO_Pipeline')
    logger.setLevel(logging.DEBUG) # Set root logger to capture all levels

    # Avoid adding duplicate handlers if logging is re-initialized
    if logger.handlers:
        # If handlers exist, we assume logging is already set up (e.g., module reload)
        pass
    else:
        # Create console handler (for stdout)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO) # Console output level
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)

        # Create file handler (for detailed log file)
        fh = logging.FileHandler(log_filepath, mode='w') # 'w' to overwrite previous logs
        fh.setLevel(logging.DEBUG) # File output level
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    # Suppress noisy loggers from other libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('HDF5').setLevel(logging.WARNING)


    logger.info(f"Logging initialized. Detailed log saving to: {log_filepath}")
    return logger, log_filepath

def get_logger(name):
    """
    Retrieves a logger with the specified name, prefixed by the root logger name.
    Ensures that logs from different modules are captured by the root configuration.
    """
    return logging.getLogger(f'OVRO_Pipeline.{name}')

def update_log_filename(new_log_filepath):
    """
    Updates the file handler of the root logger to point to a new file path.
    Used when the working directory or final log name is determined after initialization.
    """
    logger = logging.getLogger('OVRO_Pipeline')
    new_fh = None

    # Find and update the existing FileHandler
    # We need to iterate over a copy of the list because we modify it
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            # Create a new handler with the new path
            new_fh = logging.FileHandler(new_log_filepath, mode='a') # 'a' to append
            new_fh.setLevel(handler.level)
            new_fh.setFormatter(handler.formatter)
            # Remove the old handler
            logger.removeHandler(handler)
            break

    if new_fh:
        logger.addHandler(new_fh)
    elif not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
         # If somehow no file handler exists, attempt to recover gracefully
        console_handler = next((h for h in logger.handlers if isinstance(h, logging.StreamHandler)), None)
        formatter = console_handler.formatter if console_handler else logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        new_fh = logging.FileHandler(new_log_filepath, mode='a')
        new_fh.setLevel(logging.DEBUG)
        new_fh.setFormatter(formatter)
        logger.addHandler(new_fh)


# Initialize a basic logger for utilities module itself
logger = get_logger('Utils')

# ==============================================================================
# === Casacore/Pyrap Tables Check (for direct table access) ===
# ==============================================================================

CASA_IMPORTS['table_tools_available'] = False
try:
    # Prefer the older pyrap.tables if available, as per existing scripts
    import pyrap.tables as pt
    CASA_IMPORTS['table_tools_available'] = True
    CASA_IMPORTS['table_tools_module'] = pt
    logger.debug("Found table tools library: pyrap.tables")
except ImportError:
    try:
        # Fallback to the newer casacore.tables
        import casacore.tables as pt
        CASA_IMPORTS['table_tools_available'] = True
        CASA_IMPORTS['table_tools_module'] = pt
        logger.debug("Found table tools library: casacore.tables")
    except ImportError:
        # If neither is found, store that information
        CASA_IMPORTS['table_tools_module'] = None
        if CASA_AVAILABLE:
            print("WARNING: Neither pyrap.tables nor casacore.tables found. Direct table editing will be disabled.", file=sys.stderr)

# For backwards compatibility with QA script
CASACORE_TABLES_AVAILABLE = CASA_IMPORTS['table_tools_available']

# ==============================================================================
# === Context and Metadata Management ===
# ==============================================================================

def initialize_context(input_dir, output_base_dir, script_dir):
    """
    Initializes the pipeline context in a two-phase process.
    """
    # --- Phase 1: Initial Setup and Temporary Directory ---
    context = {
        'input_dir': os.path.abspath(input_dir),
        'output_base_dir': os.path.abspath(output_base_dir),
        'start_time': datetime.now(),
        'status': 'initialized',
        'errors': [],
        'warnings': [],
        'modeled_sources': [], # Sources included in the sky model
        'calibration_tables': {}, # Calibration tables generated
    }
    
    context['script_dir'] = script_dir

    # Create a temporary working directory based on the start timestamp
    timestamp = context['start_time'].strftime('%Y%m%d_%H%M%S')
    temp_parent_dir = os.path.join(context['output_base_dir'], 'temp_processing')
    
    os.makedirs(temp_parent_dir, exist_ok=True)

    temp_working_dir = os.path.join(temp_parent_dir, timestamp)
    os.makedirs(temp_working_dir)

    context['working_dir'] = temp_working_dir
    context['ms_dir'] = os.path.join(temp_working_dir, 'ms')
    context['qa_dir'] = os.path.join(temp_working_dir, 'QA')
    
    # Ensure tables_dir exists (used by calibration steps)
    context['tables_dir'] = os.path.join(temp_working_dir, 'tables')
    os.makedirs(context['tables_dir'], exist_ok=True)

    # Create subdirectories
    os.makedirs(context['ms_dir'])
    # QA dir is created by setup_logging

    # Initialize logging in the temporary directory
    root_logger, temp_log_filepath = setup_logging(context['qa_dir'])
    context['log_filepath'] = temp_log_filepath

    # --- Phase 2: Context Finalization (Analyze Data and Move Directory) ---
    root_logger.info("--- Starting Context Finalization (Phase 2) ---")
    try:
        # 1. Analyze Input Files
        # CRITICAL: Sorting ensures deterministic SPW assignment during concatenation
        ms_files = sorted(glob.glob(os.path.join(context['input_dir'], '*.ms')))
        if not ms_files:
            raise FileNotFoundError("No MS files found in the input directory.")
        context['input_files'] = ms_files
        context['num_files'] = len(ms_files)

        # This now includes the dynamic detection of integrations and sub-bands
        obs_metadata = analyze_observation_metadata(ms_files)
        
        # Add the MS file list to the metadata
        obs_metadata['ms_files'] = ms_files

        # Assign all observation-specific metadata to the 'obs_info' key
        context['obs_info'] = obs_metadata

        # 2. Determine Final Directory Structure and Identifiers
        obs_date = context['obs_info']['obs_mid_time'].datetime
        date_str = obs_date.strftime('%Y-%m-%d')
        # Calculate LST hour robustly
        lst_hour_float = context['obs_info']['obs_mid_lst_hour']
        # Handle the case where LST might wrap around midnight (though unlikely for midpoint)
        lst_hour_int = int(math.floor(lst_hour_float)) % 24
        lst_hour_str = f"{lst_hour_int:02d}h"
        
        # Store LST hour string in obs_info for use in other modules (e.g., add_sky_model)
        context['obs_info']['lst_hour'] = lst_hour_str
        context['obs_info']['obs_date'] = date_str


        # Define the time identifier used for file naming
        time_identifier = f"{date_str}_{lst_hour_str}"
        context['time_identifier'] = time_identifier

        # Define the final parent directory structure (e.g., /base/YYYY-MM-DD/LST/)
        # This is used by the master script to determine the final location (success/failure)
        final_parent_structure = os.path.join(context['output_base_dir'], date_str, lst_hour_str)
        context['final_parent_structure'] = final_parent_structure
        
        # Define the temporary working directory path within the final structure (for the move operation)
        # e.g., /base/YYYY-MM-DD/LST/working/timestamp/
        temp_working_in_final_structure = os.path.join(final_parent_structure, 'working', timestamp)

        # 3. Move the Working Directory
        root_logger.info(f"Moving working directory from {context['working_dir']} to {temp_working_in_final_structure}")
        # Ensure the destination parent exists before moving
        os.makedirs(os.path.dirname(temp_working_in_final_structure), exist_ok=True)
            
        shutil.move(context['working_dir'], temp_working_in_final_structure)

        # 4. Update Context Paths to reflect the new location
        context['working_dir'] = temp_working_in_final_structure
        context['ms_dir'] = os.path.join(temp_working_in_final_structure, 'ms')
        context['qa_dir'] = os.path.join(temp_working_in_final_structure, 'QA')
        context['tables_dir'] = os.path.join(temp_working_in_final_structure, 'tables')

        # 5. Rename and Update Log File
        final_log_filename = f"pipeline_{time_identifier}.log"
        final_log_filepath = os.path.join(context['qa_dir'], final_log_filename)

        # Handle potential OS-specific renaming issues with open files
        # The log file was moved along with the directory, find it there.
        temp_log_in_new_dir = os.path.join(context['qa_dir'], os.path.basename(temp_log_filepath))

        if os.path.exists(temp_log_in_new_dir) and temp_log_in_new_dir != final_log_filepath:
             # Rename the file on disk before updating the handler
             try:
                 os.rename(temp_log_in_new_dir, final_log_filepath)
             except OSError as e:
                 root_logger.warning(f"Could not rename log file immediately (file might be busy): {e}")
                 # If rename fails, the update_log_filename will create the new file and start writing there.

        # Update the logging handlers to point to the new file
        update_log_filename(final_log_filepath)
        context['log_filepath'] = final_log_filepath

        # Define the primary output MS name
        context['concat_ms'] = os.path.join(context['ms_dir'], f'fullband_calibration_{time_identifier}.ms')

        root_logger.info("Working directory moved and log file updated.")

    except Exception as e:
        root_logger.error(f"Failed to finalize context: {e}", exc_info=True)
        context['status'] = 'error_context_finalization'
        raise

    root_logger.info("--- Context Finalization Complete ---")
    return context

def analyze_observation_metadata(ms_files):
    """
    Analyzes a list of MS files by parsing filenames to extract timestamps, 
    integrations, and sub-bands. (Updated for dynamic detection and duration fix)
    """
    logger.info(f"Analyzing {len(ms_files)} input files...")
    
    # Data structures to store parsed information
    timestamps_dt = []
    unique_times = set()
    # unique_freqs will store the frequency strings (e.g., '13MHz')
    unique_freqs = set()

    # Regex to match the expected filename format: YYYYMMDD_HHMMSS_FreqMHz.ms
    filename_pattern = re.compile(r'(\d{8})_(\d{6})_(\d+MHz)(?:_averaged)?\.ms')

    # We rely on ms_files being sorted (done in initialize_context)
    for ms_path in ms_files:
        filename = os.path.basename(ms_path)
        match = filename_pattern.match(filename)
        if match:
            date_str, time_str, freq_str = match.groups()
            timestamp_str = f'{date_str}_{time_str}'
            
            # Track unique time and frequency identifiers
            unique_times.add(timestamp_str)
            unique_freqs.add(freq_str)

            try:
                dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                timestamps_dt.append(dt)
            except ValueError:
                logger.warning(f"Could not parse timestamp from filename: {filename}")
        else:
            logger.warning(f"Filename does not match expected pattern: {filename}")

    if not timestamps_dt:
        raise ValueError("Could not extract any valid timestamps from filenames.")

    # Calculate dynamic parameters
    num_integrations_detected = len(unique_times)
    num_subbands_detected = len(unique_freqs)
    
    # --- Duration Calculation Fix ---
    # The timestamps in the filenames represent the MIDPOINT of the integration.
    # To calculate the total duration accurately, we need the integration time.
    
    # 1. Dynamically determine the integration time from the time differences
    sorted_unique_times_dt = sorted([datetime.strptime(ts, '%Y%m%d_%H%M%S') for ts in unique_times])
    if len(sorted_unique_times_dt) > 1:
        # Calculate differences between consecutive unique timestamps
        time_diffs = [(sorted_unique_times_dt[i+1] - sorted_unique_times_dt[i]).total_seconds() for i in range(len(sorted_unique_times_dt)-1)]
        # Use the median difference as the robust estimate of integration time
        integration_time_sec = np.median(time_diffs)
        logger.info(f"Dynamically determined integration time: {integration_time_sec:.3f} seconds.")
    else:
        # Fallback if only one integration is present (e.g., the standard LWA integration time)
        integration_time_sec = 10.031 
        logger.warning(f"Only one integration detected. Falling back to standard integration time: {integration_time_sec:.3f}s.")

    # 2. Calculate time boundaries and duration
    # The timestamps used here are the midpoints.
    start_midpoint_dt = min(timestamps_dt)
    end_midpoint_dt = max(timestamps_dt)
    
    # Duration based on midpoints (this is short by one integration time)
    duration_midpoints = end_midpoint_dt - start_midpoint_dt
    
    # Corrected total duration (from start of first integration to end of last integration)
    total_duration_sec = duration_midpoints.total_seconds() + integration_time_sec
    
    # Calculate the true start and end times of the observation
    start_time_dt = start_midpoint_dt - timedelta(seconds=integration_time_sec / 2)
    end_time_dt = end_midpoint_dt + timedelta(seconds=integration_time_sec / 2)

    # Calculate the midpoint of the entire observation
    midpoint_time_dt = start_time_dt + timedelta(seconds=total_duration_sec / 2)

    # --- End of Duration Calculation Fix ---

    # Convert to Astropy Time objects for LST calculation
    midpoint_time = Time(midpoint_time_dt, scale='utc', location=config.OVRO_LOCATION)

    # Calculate LST
    # Suppress ERFA warnings that sometimes occur with future dates (e.g., dubious year)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="astropy._erfa")
        # Specifically suppress the "dubious year" warning seen in the logs
        warnings.filterwarnings("ignore", message="ERFA function \"dtf2d\" yielded 1 of \"dubious year (Note 6)\"")
        try:
             lst = midpoint_time.sidereal_time('mean')
        except Exception as e:
             logger.error(f"Error calculating LST: {e}")
             raise

    logger.info(f"Observation Midpoint: {midpoint_time.isot} UTC. Calculated LST: {lst.hour:.2f}h.")
    logger.info(f"Total observation duration: {total_duration_sec/60:.2f} minutes ({total_duration_sec:.2f} seconds).")
    logger.info(f"Detected Structure: {num_integrations_detected} integrations x {num_subbands_detected} sub-bands.")

    # Optional consistency check
    if len(ms_files) != num_integrations_detected * num_subbands_detected:
        logger.warning(f"Inconsistent data detected! Expected {num_integrations_detected * num_subbands_detected} files based on structure, but found {len(ms_files)}. Some data may be missing.")


    return {
        'obs_start_time': Time(start_time_dt, scale='utc'),
        'obs_end_time': Time(end_time_dt, scale='utc'),
        'obs_mid_time': midpoint_time,
        'obs_duration_sec': total_duration_sec,
        'integration_time_sec': integration_time_sec,
        'obs_mid_lst': lst,
        'obs_mid_lst_hour': lst.hour,
        'num_integrations_detected': num_integrations_detected,
        'num_subbands_detected': num_subbands_detected,
        # ADDED: Store the detected frequency strings for SPW selection logic
        'unique_freqs_detected': unique_freqs, 
    }

# ==============================================================================
# === Execution Utilities ===
# ==============================================================================

# ... (run_command remains the same) ...
def run_command(command, task_name="External Command", return_output=False):
    """
    Runs an external command, logs output, and handles errors.
    If return_output is True, it captures and returns stdout instead of logging it.
    Command should be provided as a list of strings (e.g., ['ls', '-l']).
    """
    logger.info(f"Executing: {task_name}")
    
    # Ensure command is a list for subprocess calls
    if isinstance(command, str):
        logger.error("run_command received a string command. It must be a list (e.g., ['cmd', 'arg1']).")
        # Attempting to split might be unsafe, better to enforce list input.
        return None if return_output else False

    # Ensure all elements are strings for the logging/subprocess call
    cmd_list = [str(c) for c in command]
    logger.debug(f"Command: {' '.join(cmd_list)}")

    try:
        if return_output:
            # Capture output and return it as a string
            # check=True raises CalledProcessError on non-zero exit code
            result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
            logger.info(f"Task '{task_name}' completed successfully.")
            
            # Log stderr if it exists, as it may contain useful warnings (like Astropy ERFA warnings)
            if result.stderr:
                logger.debug(f"[{task_name} STDERR]\n{result.stderr.strip()}")
                
            return result.stdout
        else:
            # Stream output to the logger in real-time
            process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            for line in iter(process.stdout.readline, ''):
                logger.debug(f'[{task_name}] {line.strip()}')
            process.stdout.close()
            return_code = process.wait()
            
            if return_code:
                # Raise error if the process failed
                raise subprocess.CalledProcessError(return_code, cmd_list)
            
            logger.info(f"Task '{task_name}' completed successfully.")
            return True # Indicates success

    except subprocess.CalledProcessError as e:
        logger.error(f"Task '{task_name}' failed with return code {e.returncode}.")
        # If return_output was used, the error object contains stdout/stderr
        if return_output:
            if e.stdout:
                logger.error(f"STDOUT:\n{e.stdout}")
            if e.stderr:
                logger.error(f"STDERR:\n{e.stderr}")
        # If streaming output was used, it's already logged above.
            
        if return_output:
            return None # Indicates failure
        else:
            return False # Indicates failure
    except FileNotFoundError:
        # This usually happens if the executable itself is not found
        logger.error(f"Command not found for task '{task_name}': {cmd_list[0]}")
        if return_output:
            return None
        else:
            return False
    except Exception as e:
        # Catch other unexpected errors during execution
        logger.exception(f"An unexpected error occurred during task '{task_name}'.")
        if return_output:
            return None
        else:
            return False

# ==============================================================================
# === CASA-Specific Utilities ===
# ==============================================================================

def get_ms_metadata(ms_path):
    """
    Retrieves essential metadata from a Measurement Set using CASA tools (msmetadata).
    (Improved robustness against msmetadata tool issues)
    """
    meta_logger = get_logger('Utils.Metadata')
    if not CASA_AVAILABLE or 'msmetadata' not in CASA_IMPORTS:
        meta_logger.error("CASA or msmetadata tool not available.")
        return None
    
    if not os.path.exists(ms_path):
        meta_logger.error(f"MS path does not exist: {ms_path}")
        return None

    # Instantiate the tool
    msmd_factory = CASA_IMPORTS['msmetadata']
    msmd = msmd_factory()
    
    try:
        msmd.open(ms_path)

        # 1. Frequencies and SPWs (Robust retrieval)
        spw_ids = []
        # Check if spwids method exists and works (Issue i)
        if hasattr(msmd, 'spwids'):
            try:
                # Convert potential numpy array to list
                spw_ids = list(msmd.spwids())
            except Exception as e:
                meta_logger.warning(f"msmd.spwids() failed: {e}. Trying datadescids().")
        
        # If spwids() failed or didn't exist (as seen in the log), try datadescids()
        if not spw_ids:
            meta_logger.warning("msmd.spwids() failed or returned empty. Trying msmd.datadescids().")
            if hasattr(msmd, 'datadescids'):
                try:
                    # Often maps 1:1 with SPWs in LWA data
                    spw_ids = list(msmd.datadescids())
                    if spw_ids:
                        meta_logger.info("Using Data Description IDs as SPW IDs.")
                except Exception as e:
                    meta_logger.error(f"msmd.datadescids() also failed: {e}.")

        if not spw_ids:
            raise ValueError("Could not retrieve SPW IDs or Data Description IDs.")

        
        freq_info = {}
        all_freqs = []
        for spw_id in spw_ids:
            # Use try-except block inside loop in case one SPW is corrupted
            try:
                # Ensure ID is integer if required by the tool version
                spw_id_int = int(spw_id)
                # We use spw_info which should work for both SPW IDs and Data Desc IDs in this context
                spw_info = msmd.spw_info(spw_id_int)
                freqs = spw_info['freqs'] / 1e6 # Convert Hz to MHz
                freq_info[spw_id_int] = {
                    'chan_width_mhz': spw_info['chan_widths'][0] / 1e6,
                    'num_chans': len(freqs),
                    'min_freq_mhz': freqs.min(),
                    'max_freq_mhz': freqs.max(),
                    'freqs_mhz': freqs
                }
                all_freqs.extend(freqs)
            except Exception as e:
                meta_logger.warning(f"Could not read info for ID {spw_id}: {e}")
                continue

        if not freq_info:
             raise ValueError("No valid SPW/DataDesc information found.")

        # 2. Time and Integrations
        times = msmd.times()
        if times.size == 0:
             raise ValueError("No time data found in MS.")
             
        int_time_info = msmd.exposuretime()
        int_time_sec = int_time_info['value'] if int_time_info else np.nan
         
        # Number of unique time stamps across the entire dataset
        num_integrations = len(np.unique(times))

        meta_logger.info(f"Found {num_integrations} integrations in {os.path.basename(ms_path)} (MS Metadata).")

        # 3. Antennas
        antenna_names = msmd.antenna_names()
        num_antennas = len(antenna_names)

        # Use .done() instead of .close() for msmd robustness
        msmd.done()

        return {
            'spw_info': freq_info,
            'spw_ids': list(freq_info.keys()), # Use only successfully read SPWs
            'all_freqs_mhz': np.array(sorted(all_freqs)),
            'num_integrations': num_integrations,
            'integration_time_sec': int_time_sec,
            'num_antennas': num_antennas,
            'antenna_names': antenna_names
        }

    except Exception as e:
        meta_logger.error(f"Error retrieving metadata from {ms_path}: {e}", exc_info=True)
        # Ensure tool is closed
        try:
            msmd.done()
        except Exception:
            try:
                msmd.close()
            except Exception:
                pass
        return None

def get_casa_task(task_name):
    """Helper to retrieve a CASA task if available."""
    if CASA_TASKS_AVAILABLE and task_name in CASA_IMPORTS:
        return CASA_IMPORTS[task_name]
    else:
        logger.error(f"CASA task '{task_name}' is not available.")
        return None

# ==============================================================================
# === NEW/MOVED: Phase Center Utilities ===
# ==============================================================================

def set_phase_center(ms_path, context):
    """
    Ensures the MS has a common phase center using the external chgcentre tool.
    The target phase center is determined by the CAL_PHASE_CENTER config variable.
    (Moved from bandpass_calibration.py)
    """
    util_logger = get_logger('Utils.PhaseCenter')
    target = config.CAL_PHASE_CENTER
    if not target:
        util_logger.info("CAL_PHASE_CENTER is not set. Skipping phase center correction.")
        return True

    util_logger.info(f"Setting common phase center. Target: '{target}'")
    
    try:
        if target.lower() == 'zenith':
            # Calculate zenith at the midpoint of the observation
            mid_time = context['obs_info']['obs_mid_time']
            zenith_altaz_frame = AltAz(obstime=mid_time, location=config.OVRO_LOCATION)
            target_icrs = SkyCoord(alt=90*u.deg, az=0*u.deg, frame=zenith_altaz_frame).icrs
            ra_str = target_icrs.ra.to_string(unit=u.hourangle, sep='hms', precision=3)
            dec_str = target_icrs.dec.to_string(unit=u.deg, sep='dms', precision=3)
        
        elif target in config.PRIMARY_SOURCES:
            # Get coordinates from the source dictionary
            source_coord = config.PRIMARY_SOURCES[target]['skycoord']
            ra_str = source_coord.ra.to_string(unit=u.hourangle, sep='hms', precision=3)
            dec_str = source_coord.dec.to_string(unit=u.deg, sep='dms', precision=3)
        
        else:
            util_logger.error(f"Invalid CAL_PHASE_CENTER value: '{target}'. Must be 'zenith' or a primary source name.")
            return False

        chgcentre_cmd = [config.CHGCENTRE_PATH, ms_path, ra_str, dec_str]
        
        success = run_command(
            chgcentre_cmd, 
            task_name=f"chgcentre (RA={ra_str}, Dec={dec_str})"
        )
        if not success:
            util_logger.error("chgcentre tool failed. See logs for details.")
            return False

    except Exception as e:
        util_logger.error(f"An unexpected error occurred during phase center correction: {e}", exc_info=True)
        return False
        
    util_logger.info("Phase center correction applied successfully.")
    return True

def check_phase_center(ms_path, context):
    """
    Verifies the phase center of the MS (FIELD_ID 0) against the CAL_PHASE_CENTER config.
    Returns True if the center matches (within tolerance), False otherwise.
    
    Uses chgcentre to read the current phase center, as msmetadata.phasedir failed.
    """
    util_logger = get_logger('Utils.PhaseCenter')
    target = config.CAL_PHASE_CENTER
    
    if not target:
        util_logger.info("CAL_PHASE_CENTER is not set. Skipping phase center verification.")
        return True
    
    # 1. Determine target coordinates
    try:
        if target.lower() == 'zenith':
            mid_time = context['obs_info']['obs_mid_time']
            zenith_altaz_frame = AltAz(obstime=mid_time, location=config.OVRO_LOCATION)
            target_icrs = SkyCoord(alt=90*u.deg, az=0*u.deg, frame=zenith_altaz_frame).icrs
        elif target in config.PRIMARY_SOURCES:
            target_icrs = config.PRIMARY_SOURCES[target]['skycoord']
        else:
            util_logger.error(f"Invalid CAL_PHASE_CENTER value: '{target}'. Cannot verify.")
            return False
    except Exception as e:
        util_logger.error(f"Failed to calculate target coordinates for '{target}': {e}")
        return False

    # 2. Get actual coordinates from MS using chgcentre
    actual_skycoord = None
    try:
        # Run chgcentre with just the MS path to get its output
        chgcentre_cmd = [config.CHGCENTRE_PATH, ms_path]
        stdout = run_command(chgcentre_cmd, task_name="Check Phase Center", return_output=True)

        if stdout is None:
            # run_command already logged the error
            util_logger.error("chgcentre command failed during verification.")
            return False

        # Parse the output. Based on logs[cite: 13, 16], the line looks like:
        # Processing field "...": (RA) (DEC) -> ...
        # Or if run without target coords, it might just print the "from" part.
        # We will look for the first valid coordinate string pair in the output.
        
        # Regex to find an RA (e.g., 17h55m47.162s) and Dec (e.g., 37d14m49.948s)
        # This assumes the *first* field reported is the one we care about (FIELD_ID 0)
        coord_pattern = re.compile(r'(\d{1,2}h\d{1,2}m\d{1,2}\.\d+s) (\S\d{1,2}d\d{1,2}m\d{1,2}\.\d+s)')
        match = coord_pattern.search(stdout)

        if not match:
            util_logger.error("Could not parse RA/Dec from chgcentre output.")
            util_logger.debug(f"chgcentre output was: {stdout}")
            return False
            
        ra_str = match.group(1)
        dec_str = match.group(2)
        
        util_logger.info(f"Found phase center from chgcentre: RA={ra_str} Dec={dec_str}")
        
        # Assume ICRS/J2000, which is what chgcentre uses
        actual_skycoord = SkyCoord(f'{ra_str} {dec_str}', frame='icrs', unit=(u.hourangle, u.deg))

    except Exception as e:
        util_logger.error(f"Failed to read/parse phase center using chgcentre: {e}", exc_info=True)
        return False

    # 3. Compare coordinates
    separation_arcsec = target_icrs.separation(actual_skycoord).to_value(u.arcsec)
    tolerance_arcsec = 1.0 # 1 arcsecond tolerance

    if separation_arcsec < tolerance_arcsec:
        util_logger.info(f"Phase center confirmed. Target: {target}, Separation: {separation_arcsec:.3f} arcsec")
        return True
    else:
        util_logger.error(f"Phase center mismatch!")
        util_logger.error(f"  -> Target ({target}): {target_icrs.to_string('hmsdms')}")
        util_logger.error(f"  -> Actual (FIELD 0): {actual_skycoord.to_string('hmsdms')}")
        util_logger.error(f"  -> Separation: {separation_arcsec:.3f} arcsec")
        return False


# ==============================================================================
# === Calibration Utilities ===
# ==============================================================================

def determine_calibration_uv_range(context):
    """
    Determines the uvrange string based on the sources used in the sky model.
    (Centralized logic for use by Bandpass and Delay calibration)
    """
    # Use the main Utils logger for this function
    
    modeled_sources = context.get('model_sources', [])
    
    # Check if Virgo A is present, which may require a specific UV range
    if 'VirA' in modeled_sources:
        uvrange_str = config.CAL_UVRANGE_VIRA
        logger.info("Virgo A detected in model. Using VirA-specific uvrange.")
    else:
        uvrange_str = config.CAL_UVRANGE_DEFAULT
        # Log if model is empty or just doesn't contain VirA
        if not modeled_sources:
            logger.info("Model is empty. Using default uvrange.")
        else:
            logger.info("Virgo A not detected in model. Using default uvrange.")
        
    logger.info(f"Selected calibration uvrange: {uvrange_str}")
    return uvrange_str
# ==============================================================================
# === Beam Model Utilities ===
# ==============================================================================

# In pipeline_utils.py

def get_beam_interpolator(beam_model_path):
    """
    Loads the HDF5 beam model and creates a 3D interpolator for the Stokes I power beam.
    This logic is harmonized with the add_sky_model.py implementation.
    The interpolator takes input coordinates as (ZA_rad, AZ_rad, Freq_MHz).
    """
    logger.info(f"Loading and interpolating beam model from {beam_model_path}...")
    if not os.path.exists(beam_model_path):
        logger.error(f"Beam model file not found: {beam_model_path}")
        return None

    try:
        with h5py.File(beam_model_path, 'r') as hf:
            # Check for essential keys from the known-good model structure
            required_keys = ['freq_Hz', 'theta_pts', 'phi_pts', 'X_pol_Efields/etheta']
            if not all(key in hf for key in required_keys):
                logger.error("Beam model HDF5 file is missing expected keys (e.g., 'freq_Hz', 'theta_pts').")
                raise KeyError("Invalid HDF5 beam file format. Structure does not match expected model.")

            # Load data based on the structure from add_sky_model.py
            fq_orig_hz = hf['freq_Hz'][:]
            th_orig_rad = hf['theta_pts'][:]  # Assumes theta is Zenith Angle in radians
            ph_orig_rad = hf['phi_pts'][:]    # Assumes phi is Azimuth in radians
            
            Exth = hf['X_pol_Efields/etheta'][:]
            Exph = hf['X_pol_Efields/ephi'][:]
            Eyth = hf['Y_pol_Efields/etheta'][:]
            Eyph = hf['Y_pol_Efields/ephi'][:]

        # Calculate Stokes I power beam (unnormalized)
        PbX = np.abs(Exth)**2 + np.abs(Exph)**2
        PbY = np.abs(Eyth)**2 + np.abs(Eyph)**2
        power_beam_unnormalized = PbX + PbY

        # Ensure coordinate axes are sorted for the interpolator
        fq_s_idx, th_s_idx, ph_s_idx = np.argsort(fq_orig_hz), np.argsort(th_orig_rad), np.argsort(ph_orig_rad)
        
        # CRITICAL FIX: Convert frequency to MHz for consistency with calculate_beam_attenuation
        fq_s_mhz = (fq_orig_hz[fq_s_idx]) / 1e6 
        th_s_rad = th_orig_rad[th_s_idx]
        ph_s_rad = ph_orig_rad[ph_s_idx]

        # Sort the beam data according to the sorted axes
        power_beam_sorted = power_beam_unnormalized[fq_s_idx, :, :][:, th_s_idx, :][:, :, ph_s_idx]

        # Normalize the beam by the gain at Zenith (theta=0)
        zenith_theta_idx = np.argmin(np.abs(th_s_rad))
        zenith_gain_vs_freq = power_beam_sorted[:, zenith_theta_idx, 0]
        # Avoid division by zero
        zenith_gain_vs_freq[zenith_gain_vs_freq == 0] = 1e-9
        
        power_beam_normalized = power_beam_sorted / zenith_gain_vs_freq[:, np.newaxis, np.newaxis]
        
        # Create the 3D interpolator (Freq, Theta, Phi)
        interpolator = RegularGridInterpolator(
            (fq_s_mhz, th_s_rad, ph_s_rad),
            power_beam_normalized,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        logger.info("Beam interpolator created successfully.")
        return interpolator

    except Exception as e:
        logger.error(f"An error occurred while loading beam model {beam_model_path}: {e}", exc_info=True)
        return None


def calculate_beam_attenuation(context, source_name, freqs_mhz, beam_interpolator):
    """
    Calculates the beam attenuation (power beam gain) for a source at specific frequencies.
    Attenuation = Power_Gain.
    """
    if beam_interpolator is None:
        return np.ones_like(freqs_mhz, dtype=float)

    source_details = config.PRIMARY_SOURCES.get(source_name)
    if not (source_details and 'skycoord' in source_details):
        logger.error(f"Source coordinates not found in config.PRIMARY_SOURCES for {source_name}.")
        return np.ones_like(freqs_mhz, dtype=float)
    source_coords = source_details['skycoord']

    obs_time = context['obs_info'].get('obs_mid_time')
    if not obs_time:
        logger.error(f"Observation time missing from context for beam calculation.")
        return np.ones_like(freqs_mhz, dtype=float)

    altaz_frame = AltAz(obstime=obs_time, location=config.OVRO_LOCATION)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="astropy._erfa")
        source_altaz = source_coords.transform_to(altaz_frame)
    
    za_rad = np.pi/2 - source_altaz.alt.rad
    az_rad = source_altaz.az.rad
    if za_rad < 0:
        za_rad = 0.0

    freqs_mhz = np.atleast_1d(freqs_mhz)
    # Ensure points are ordered as (Freq, ZA, AZ) to match the interpolator
    points = np.vstack((freqs_mhz, 
                        np.full_like(freqs_mhz, za_rad), 
                        np.full_like(freqs_mhz, az_rad))).T

    try:
        # MODIFIED: The interpolator now directly returns power gain.
        # The squaring operation has been removed.
        power_gain = beam_interpolator(points)
        
        # Set very small gain values to zero to avoid numerical issues
        power_gain[power_gain < 1e-6] = 0.0
        return power_gain
    except Exception as e:
        logger.error(f"Error during beam interpolation for {source_name}: {e}")
        return np.ones_like(freqs_mhz, dtype=float)

# ==============================================================================
# === Data Mapping and Diagnostics ===
# ==============================================================================

# ... (Data Mapping and Diagnostics remain the same) ...

def load_antenna_mapping():
    """
    Loads the antenna mapping from the configuration file.
    Returns a dictionary mapping correlator number (antenna ID) to SNAP2 location.
    """
    mapping = config.ANTENNA_MAPPING
    if not mapping:
        logger.error("Antenna mapping (ANTENNA_MAPPING) is empty or missing in pipeline_config.py.")
        return None

    # Use get_logger specific to mapping for clearer logs
    map_logger = get_logger('Utils.Mapping')
    map_logger.info(f"Loaded {len(mapping)} antenna mappings directly from config.")
    return mapping

def map_delays_to_snap2(delay_data, antenna_mapping):
    """
    Maps delay data (indexed by antenna ID) to SNAP2 board groups.

    Args:
        delay_data (dict): Dictionary where keys are antenna IDs.
        antenna_mapping (dict): Mapping from antenna ID to SNAP2 location string.

    Returns:
        dict: Dictionary grouped by SNAP2 board (e.g., 'R3'), containing lists of
              (antenna_id, delay_value) tuples.
    """
    snap2_groups = {}

    for ant_id, delay_value in delay_data.items():
        location_str = antenna_mapping.get(ant_id)
        if not location_str:
            logger.warning(f"Antenna ID {ant_id} not found in mapping.")
            continue

        # Extract the Rack identifier (e.g., 'R3' from 'R3C1A')
        # Assumes standard format R<rack>C<chassis>A/B
        match = re.match(r'(R\d+)', location_str)
        if match:
            rack_id = match.group(1)
            if rack_id not in snap2_groups:
                snap2_groups[rack_id] = []
            snap2_groups[rack_id].append((ant_id, delay_value))
        else:
            logger.warning(f"Could not parse Rack ID from location string: {location_str} for Ant {ant_id}")

    return snap2_groups

def run_snap2_diagnostics(problematic_delays, antenna_mapping, threshold_percent=config.SNAP2_FAILURE_THRESHOLD_PERCENT):
    """
    Runs diagnostics on SNAP2 boards based on problematic delay values.
    """
    diag_logger = get_logger('QA.Diagnostics')
    diag_logger.info("Running SNAP2 board diagnostics based on problematic delay values.")

    # 1. Map problematic delays to SNAP2 groups
    problematic_groups = map_delays_to_snap2(problematic_delays, antenna_mapping)

    # 2. Determine the total number of antennas associated with each SNAP2 group
    # We need to iterate over the full mapping to get the totals per rack
    total_counts_per_group = {}
    for location_str in antenna_mapping.values():
        match = re.match(r'(R\d+)', location_str)
        if match:
            rack_id = match.group(1)
            total_counts_per_group[rack_id] = total_counts_per_group.get(rack_id, 0) + 1

    # 3. Analyze failure rates
    failed_boards = []
    # Sort groups for consistent logging output
    if total_counts_per_group:
        # Sort Racks numerically (R3, R4, ..., R13)
        all_groups = sorted(list(total_counts_per_group.keys()), key=lambda x: int(x[1:]))
    else:
        all_groups = []
        diag_logger.warning("No groups found in antenna mapping. Cannot run SNAP2 diagnostics.")


    for rack_id in all_groups:
        total_antennas = total_counts_per_group[rack_id]
        # Get the list of problematic antennas for this rack, default to empty list if none
        failed_antennas = problematic_groups.get(rack_id, [])
        num_failed = len(failed_antennas)

        if total_antennas > 0:
            failure_rate = (num_failed / total_antennas) * 100
        else:
            failure_rate = 0

        diag_logger.info(f"SNAP2 Board Group {rack_id}: {num_failed}/{total_antennas} antennas with problematic delays ({failure_rate:.0f}%).")

        if failure_rate >= threshold_percent:
            diag_logger.error(f"CRITICAL: SNAP2 Board Group {rack_id} exceeds failure threshold ({threshold_percent}%). Potential board issue detected.")
            failed_boards.append(rack_id)

    if not failed_boards and all_groups:
        diag_logger.info("SNAP2 diagnostics completed successfully. No boards exceeded the failure threshold.")
    elif failed_boards:
        diag_logger.warning(f"SNAP2 diagnostics completed. {len(failed_boards)} boards exceeded the failure threshold.")

    return failed_boards

# ==============================================================================
# === Helper Functions ===
# ==============================================================================

def calculate_uvrange(ms_path, min_baseline_lambda=5, max_baseline_lambda=350):
    """uuu
    (DEPRECATED - use determine_calibration_uv_range(context) instead)
    Calculates a static uvrange string for CASA tasks.
    """
    logger.warning("pipeline_utils.calculate_uvrange is deprecated. Use determine_calibration_uv_range(context).")
    uvrange_str = f">{min_baseline_lambda}lambda,<{max_baseline_lambda}lambda"
    return uvrange_str

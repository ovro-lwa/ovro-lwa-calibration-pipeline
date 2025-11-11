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

# We can only import config if script_dir is correct
try:
    import pipeline_config as config
except ImportError:
    # This will be handled by the main script's early sys.path insert
    pass

# Matplotlib configuration (unchanged)
try:
    import matplotlib
    matplotlib.use('Agg') # Use a non-interactive backend
    import matplotlib.pyplot as plt
    _plotting_available = True
except ImportError:
    _plotting_available = False
    print("WARNING: Matplotlib not found. QA plotting will be disabled.", file=sys.stderr)


# ==============================================================================
# === CASA Availability Check (Unchanged) ===
# ==============================================================================
CASA_AVAILABLE = False
TABLE_TOOLS_AVAILABLE = False
CASA_IMPORTS = {}
CASA_TASKS_AVAILABLE = False

try:
    import casatools
    CASA_IMPORTS['casatools'] = casatools
    tool_names = ['measures', 'msmetadata', 'componentlist', 'table']
    quantity_tool_name = None
    if hasattr(casatools, 'quantity'): quantity_tool_name = 'quantity'
    elif hasattr(casatools, 'quanta'): quantity_tool_name = 'quanta'
    if quantity_tool_name: tool_names.append(quantity_tool_name)
    else: raise ImportError("Neither 'quantity' nor 'quanta' found.")
    for factory_name in tool_names:
        if hasattr(casatools, factory_name):
            factory_func = getattr(casatools, factory_name)
            CASA_IMPORTS[factory_name] = factory_func
            if factory_name == 'table': TABLE_TOOLS_AVAILABLE = True
        else: raise ImportError(f"CASA tool factory '{factory_name}' not found.")
    CASA_AVAILABLE = True
    try:
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
        print(f"WARNING: Could not import all required casatasks. Error: {e}", file=sys.stderr)
        CASA_TASKS_AVAILABLE = False
except ImportError as e:
    CASA_AVAILABLE = False
    TABLE_TOOLS_AVAILABLE = False
    print(f"WARNING: CASA environment not fully initialized. Error: {e}", file=sys.stderr)

# ==============================================================================
# === Logging Utilities (Unchanged) ===
# ==============================================================================
def setup_logging(log_directory, log_filename='pipeline.log'):
    """Sets up the logging configuration."""
    if not os.path.exists(log_directory): os.makedirs(log_directory)
    log_filepath = os.path.join(log_directory, log_filename)
    logger = logging.getLogger('OVRO_Pipeline')
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)
        fh = logging.FileHandler(log_filepath, mode='w')
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('HDF5').setLevel(logging.WARNING)
    logger.info(f"Logging initialized. Log saving to: {log_filepath}")
    return logger, log_filepath

def get_logger(name):
    """Retrieves a logger prefixed by the root logger name."""
    root_logger_name = 'OVRO_Pipeline'
    if not logging.getLogger(root_logger_name).hasHandlers():
        # Fallback if root logger isn't set up, just get a basic logger
        # This can happen if utils is imported before main setup
        basic_logger = logging.getLogger(name)
        if not basic_logger.hasHandlers():
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            basic_logger.addHandler(ch)
        return basic_logger

    root_logger_level = logging.getLogger(root_logger_name).getEffectiveLevel()
    sub_logger = logging.getLogger(f'{root_logger_name}.{name}')
    if sub_logger.level > root_logger_level:
         sub_logger.setLevel(root_logger_level)
    return sub_logger

def update_log_filename(new_log_filepath):
    """Updates the file handler of the root logger."""
    logger = logging.getLogger('OVRO_Pipeline')
    new_fh = None
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            new_fh = logging.FileHandler(new_log_filepath, mode='a')
            new_fh.setLevel(handler.level)
            new_fh.setFormatter(handler.formatter)
            logger.removeHandler(handler)
            break
    if new_fh:
        logger.addHandler(new_fh)
    elif not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        console_handler = next((h for h in logger.handlers if isinstance(h, logging.StreamHandler)), None)
        formatter = console_handler.formatter if console_handler else logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        new_fh = logging.FileHandler(new_log_filepath, mode='a')
        new_fh.setLevel(logging.DEBUG)
        new_fh.setFormatter(formatter)
        logger.addHandler(new_fh)

# Get a logger instance for this module, but don't configure it
# It will inherit from the root logger set up by the main script
# OR it will be overridden by the passed-in logger in run_command
logger = get_logger('Utils')

# ==============================================================================
# === Casacore/Pyrap Tables Check (Unchanged) ===
# ==============================================================================
CASA_IMPORTS['table_tools_available'] = False
try:
    import pyrap.tables as pt
    CASA_IMPORTS['table_tools_available'] = True
    CASA_IMPORTS['table_tools_module'] = pt
    logger.debug("Found table tools library: pyrap.tables")
except ImportError:
    try:
        import casacore.tables as pt
        CASA_IMPORTS['table_tools_available'] = True
        CASA_IMPORTS['table_tools_module'] = pt
        logger.debug("Found table tools library: casacore.tables")
    except ImportError:
        CASA_IMPORTS['table_tools_module'] = None
        if CASA_AVAILABLE:
            print("WARNING: Neither pyrap.tables nor casacore.tables found.", file=sys.stderr)

CASACORE_TABLES_AVAILABLE = CASA_IMPORTS['table_tools_available']

# ==============================================================================
# === Context and Metadata Management (Unchanged) ===
# ==============================================================================
# ... (initialize_context, analyze_observation_metadata unchanged) ...
def initialize_context(input_dir, output_base_dir, script_dir):
    """Initializes the pipeline context in a two-phase process."""
    context = {
        'input_dir': os.path.abspath(input_dir),
        'output_base_dir': os.path.abspath(output_base_dir),
        'start_time': datetime.now(),
        'status': 'initialized',
        'errors': [], 'warnings': [],
        'modeled_sources': [], 'calibration_tables': {},
        'script_dir': script_dir
    }
    timestamp = context['start_time'].strftime('%Y%m%d_%H%M%S')
    temp_parent_dir = os.path.join(context['output_base_dir'], 'temp_processing')
    os.makedirs(temp_parent_dir, exist_ok=True)
    temp_working_dir = os.path.join(temp_parent_dir, timestamp)
    os.makedirs(temp_working_dir)
    context['working_dir'] = temp_working_dir
    
    # --- Create main directory structure ---
    context['ms_dir'] = os.path.join(temp_working_dir, 'ms')
    context['qa_dir'] = os.path.join(temp_working_dir, 'QA')
    context['tables_dir'] = os.path.join(temp_working_dir, 'tables')
    os.makedirs(context['tables_dir'])
    os.makedirs(context['ms_dir'])
    os.makedirs(context['qa_dir'])

    # --- Create NEW QA subdirectory structure ---
    context['ood_dir'] = os.path.join(context['qa_dir'], 'OOD')
    context['website_dir'] = os.path.join(context['qa_dir'], 'QA_website')
    context['web_bp_dir'] = os.path.join(context['website_dir'], 'bandpass_plots')
    context['web_scint_dir'] = os.path.join(context['website_dir'], 'scintillation_images')
    context['web_spec_dir'] = os.path.join(context['website_dir'], 'spectrum_images')
    context['web_allsky_dir'] = os.path.join(context['website_dir'], 'all_sky_images')
    
    # Create all directories
    for dir_path in [context['ood_dir'], context['website_dir'], context['web_bp_dir'],
                     context['web_scint_dir'], context['web_spec_dir'], context['web_allsky_dir']]:
        os.makedirs(dir_path, exist_ok=True)
    
    # --- Setup logging ---
    root_logger, temp_log_filepath = setup_logging(context['qa_dir'])
    context['log_filepath'] = temp_log_filepath

    root_logger.info("--- Starting Context Finalization (Phase 2) ---")
    try:
        ms_files = sorted(glob.glob(os.path.join(context['input_dir'], '*.ms')))
        if not ms_files: raise FileNotFoundError("No MS files found.")
        context['input_files'] = ms_files
        context['num_files'] = len(ms_files)
        obs_metadata = analyze_observation_metadata(ms_files)
        obs_metadata['ms_files'] = ms_files
        context['obs_info'] = obs_metadata
        obs_date = context['obs_info']['obs_mid_time'].datetime
        date_str = obs_date.strftime('%Y-%m-%d')
        lst_hour_float = context['obs_info']['obs_mid_lst_hour']
        lst_hour_int = int(math.floor(lst_hour_float)) % 24
        lst_hour_str = f"{lst_hour_int:02d}h"
        context['obs_info']['lst_hour'] = lst_hour_str
        context['obs_info']['obs_date'] = date_str
        time_identifier = f"{date_str}_{lst_hour_str}"
        context['time_identifier'] = time_identifier
        final_parent_structure = os.path.join(context['output_base_dir'], date_str, lst_hour_str)
        context['final_parent_structure'] = final_parent_structure
        temp_working_in_final_structure = os.path.join(final_parent_structure, 'working', timestamp)
        root_logger.info(f"Moving working directory to {temp_working_in_final_structure}")
        os.makedirs(os.path.dirname(temp_working_in_final_structure), exist_ok=True)
        shutil.move(context['working_dir'], temp_working_in_final_structure)
        
        # --- Update all context paths after the move ---
        context['working_dir'] = temp_working_in_final_structure
        context['ms_dir'] = os.path.join(temp_working_in_final_structure, 'ms')
        context['qa_dir'] = os.path.join(temp_working_in_final_structure, 'QA')
        context['tables_dir'] = os.path.join(temp_working_in_final_structure, 'tables')
        context['ood_dir'] = os.path.join(context['qa_dir'], 'OOD')
        context['website_dir'] = os.path.join(context['qa_dir'], 'QA_website')
        context['web_bp_dir'] = os.path.join(context['website_dir'], 'bandpass_plots')
        context['web_scint_dir'] = os.path.join(context['website_dir'], 'scintillation_images')
        context['web_spec_dir'] = os.path.join(context['website_dir'], 'spectrum_images')
        context['web_allsky_dir'] = os.path.join(context['website_dir'], 'all_sky_images')
        
        # --- Update log file path ---
        final_log_filename = f"pipeline_{time_identifier}.log"
        final_log_filepath = os.path.join(context['qa_dir'], final_log_filename)
        temp_log_in_new_dir = os.path.join(context['qa_dir'], os.path.basename(temp_log_filepath))
        if os.path.exists(temp_log_in_new_dir) and temp_log_in_new_dir != final_log_filepath:
             try: os.rename(temp_log_in_new_dir, final_log_filepath)
             except OSError as e: root_logger.warning(f"Could not rename log file: {e}")
        update_log_filename(final_log_filepath)
        context['log_filepath'] = final_log_filepath
        context['concat_ms'] = os.path.join(context['ms_dir'], f'fullband_calibration_{time_identifier}.ms')
        root_logger.info("Working directory moved and log file updated.")
    except Exception as e:
        root_logger.error(f"Failed to finalize context: {e}", exc_info=True)
        context['status'] = 'error_context_finalization'
        raise
    root_logger.info("--- Context Finalization Complete ---")
    return context

def analyze_observation_metadata(ms_files):
    """Analyzes MS filenames to extract metadata."""
    logger.info(f"Analyzing {len(ms_files)} input files...")
    timestamps_dt = []
    unique_times, unique_freqs = set(), set()
    filename_pattern = re.compile(r'(\d{8})_(\d{6})_(\d+MHz)(?:|_averaged)\.ms')
    for ms_path in ms_files:
        filename = os.path.basename(ms_path)
        match = filename_pattern.search(filename)
        if match:
            date_str, time_str, freq_str = match.groups()
            timestamp_str = f'{date_str}_{time_str}'
            unique_times.add(timestamp_str); unique_freqs.add(freq_str)
            try: dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S'); timestamps_dt.append(dt)
            except ValueError: logger.warning(f"Could not parse timestamp: {filename}")
        elif filename.endswith(".ms"): logger.warning(f"Filename does not match pattern: {filename}")
    if not timestamps_dt: raise ValueError("Could not extract valid timestamps.")
    num_integrations_detected, num_subbands_detected = len(unique_times), len(unique_freqs)
    sorted_unique_times_dt = sorted([datetime.strptime(ts, '%Y%m%d_%H%M%S') for ts in unique_times])
    if len(sorted_unique_times_dt) > 1:
        time_diffs = [(sorted_unique_times_dt[i+1] - t).total_seconds() for i, t in enumerate(sorted_unique_times_dt[:-1])]
        integration_time_sec = np.median(time_diffs)
        logger.info(f"Dynamically determined integration time: {integration_time_sec:.3f} s.")
    else: integration_time_sec = config.INTEGRATION_DURATION_SEC; logger.warning(f"Only one integration. Using default time: {integration_time_sec:.3f}s.")
    start_midpoint_dt, end_midpoint_dt = min(timestamps_dt), max(timestamps_dt)
    duration_midpoints = end_midpoint_dt - start_midpoint_dt
    total_duration_sec = duration_midpoints.total_seconds() + integration_time_sec
    start_time_dt = start_midpoint_dt - timedelta(seconds=integration_time_sec / 2)
    end_time_dt = end_midpoint_dt + timedelta(seconds=integration_time_sec / 2)
    midpoint_time_dt = start_time_dt + timedelta(seconds=total_duration_sec / 2)
    midpoint_time = Time(midpoint_time_dt, scale='utc', location=config.OVRO_LOCATION)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="astropy._erfa")
        warnings.filterwarnings("ignore", message="ERFA function \"dtf2d\" yielded")
        try: lst = midpoint_time.sidereal_time('mean')
        except Exception as e: logger.error(f"Error calculating LST: {e}"); raise
    logger.info(f"Obs Midpoint: {midpoint_time.isot} UTC. LST: {lst.hour:.2f}h.")
    logger.info(f"Total duration: {total_duration_sec/60:.2f} min ({total_duration_sec:.2f} s).")
    logger.info(f"Detected Structure: {num_integrations_detected} integrations x {num_subbands_detected} sub-bands.")
    if len(ms_files) != num_integrations_detected * num_subbands_detected:
        logger.warning(f"Inconsistent data! Expected {num_integrations_detected * num_subbands_detected} files, found {len(ms_files)}.")
    return {
        'obs_start_time': Time(start_time_dt, scale='utc'), 'obs_end_time': Time(end_time_dt, scale='utc'),
        'obs_mid_time': midpoint_time, 'obs_duration_sec': total_duration_sec,
        'integration_time_sec': integration_time_sec, 'obs_mid_lst': lst, 'obs_mid_lst_hour': lst.hour,
        'num_integrations_detected': num_integrations_detected, 'num_subbands_detected': num_subbands_detected,
        'unique_freqs_detected': unique_freqs,
    }

# ==============================================================================
# === Execution Utilities (MODIFIED) ===
# ==============================================================================
def run_command(command, task_name="External Command", return_output=False, logger=None):
    """
    Runs an external command, logs output, handles errors.
    Accepts a logger instance to write DEBUG output to.
    """
    if logger is None:
        # Fallback to the module logger if none is provided
        logger = get_logger('Utils.RunCommand')
        logger.warning(f"run_command task '{task_name}' received no logger, using fallback.")

    logger.info(f"Executing: {task_name}")
    if isinstance(command, str):
        logger.error("run_command received a string. Must be a list.")
        return None if return_output else False
    cmd_list = [str(c) for c in command]
    logger.debug(f"Command: {' '.join(cmd_list)}")
    try:
        if return_output:
            result = subprocess.run(cmd_list, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
            logger.info(f"Task '{task_name}' completed.")
            if result.stderr: logger.debug(f"[{task_name} STDERR]\n{result.stderr.strip()}")
            return result.stdout
        else:
            process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
            # Log stdout/stderr at INFO level
            for line in iter(process.stdout.readline, ''): 
                # --- MODIFICATION: Log at INFO level to restore console output ---
                logger.info(f'[{task_name}] {line.strip()}')
                # --- END MODIFICATION ---
            process.stdout.close()
            return_code = process.wait()
            if return_code: raise subprocess.CalledProcessError(return_code, cmd_list)
            logger.info(f"Task '{task_name}' completed successfully.")
            return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Task '{task_name}' failed (Code {e.returncode}).")
        # Note: The debug output from the process has already been logged.
        if return_output:
            if e.stdout: logger.error(f"STDOUT:\n{e.stdout}")
            if e.stderr: logger.error(f"STDERR:\n{e.stderr}")
            return None
        else: return False
    except FileNotFoundError:
        logger.error(f"Command not found for task '{task_name}': {cmd_list[0]}")
        return None if return_output else False
    except Exception as e:
        logger.exception(f"Unexpected error during task '{task_name}'.")
        return None if return_output else False


# ==============================================================================
# === CASA-Specific Utilities (Unchanged) ===
# ==============================================================================
# ... (get_ms_metadata, get_casa_task unchanged) ...
def get_ms_metadata(ms_path):
    """Retrieves essential metadata from MS using CASA msmetadata."""
    meta_logger = get_logger('Utils.Metadata')
    if not CASA_AVAILABLE or 'msmetadata' not in CASA_IMPORTS:
        meta_logger.error("CASA or msmetadata tool not available.")
        return None
    if not os.path.exists(ms_path):
        meta_logger.error(f"MS path does not exist: {ms_path}")
        return None
    msmd_factory = CASA_IMPORTS['msmetadata']
    msmd = msmd_factory()
    try:
        msmd.open(ms_path)
        spw_ids = []
        if hasattr(msmd, 'spwids'):
            try: spw_ids = list(msmd.spwids())
            except Exception as e: meta_logger.warning(f"msmd.spwids() failed: {e}. Trying datadescids().")
        if not spw_ids:
            meta_logger.warning("msmd.spwids() failed/empty. Trying msmd.datadescids().")
            if hasattr(msmd, 'datadescids'):
                try:
                    spw_ids = list(msmd.datadescids())
                    if spw_ids: meta_logger.info("Using Data Description IDs as SPW IDs.")
                except Exception as e: meta_logger.error(f"msmd.datadescids() also failed: {e}.")
        if not spw_ids: raise ValueError("Could not retrieve SPW/DataDesc IDs.")

        freq_info, all_freqs = {}, []
        for spw_id in spw_ids:
            try:
                spw_id_int = int(spw_id)
                spw_info_dict = msmd.spw_info(spw_id_int)
                freqs = spw_info_dict['freqs'] / 1e6 # MHz
                freq_info[spw_id_int] = {
                    'chan_width_mhz': spw_info_dict['chan_widths'][0] / 1e6, 'num_chans': len(freqs),
                    'min_freq_mhz': freqs.min(), 'max_freq_mhz': freqs.max(), 'freqs_mhz': freqs
                }
                all_freqs.extend(freqs)
            except Exception as e: meta_logger.warning(f"Could not read info for ID {spw_id}: {e}"); continue
        if not freq_info: raise ValueError("No valid SPW/DataDesc info found.")
        times = msmd.times()
        if times.size == 0: raise ValueError("No time data found.")
        int_time_info = msmd.exposuretime(); int_time_sec = int_time_info['value'] if int_time_info else np.nan
        num_integrations = len(np.unique(times))
        meta_logger.info(f"Found {num_integrations} integrations in {os.path.basename(ms_path)}.")
        antenna_names = msmd.antenna_names(); num_antennas = len(antenna_names)
        msmd.done()
        return {
            'spw_info': freq_info, 'spw_ids': list(freq_info.keys()), 'all_freqs_mhz': np.array(sorted(all_freqs)),
            'num_integrations': num_integrations, 'integration_time_sec': int_time_sec,
            'num_antennas': num_antennas, 'antenna_names': antenna_names
        }
    except Exception as e:
        meta_logger.error(f"Error retrieving metadata from {ms_path}: {e}", exc_info=True)
        try: msmd.done()
        except Exception: pass
        return None

def get_casa_task(task_name):
    """Helper to retrieve a CASA task if available."""
    if CASA_TASKS_AVAILABLE and task_name in CASA_IMPORTS:
        return CASA_IMPORTS[task_name]
    else:
        logger.error(f"CASA task '{task_name}' is not available.")
        return None


# --- get_caltable_spw_map FUNCTION (Unchanged) ---
def get_caltable_spw_map(cal_table_path):
    """
    Reads the SPECTRAL_WINDOW sub-table, handling potential transposed frequency arrays.
    """
    map_logger = get_logger('Utils.SPWMap')
    spw_map = {}

    tb_tool_factory = None
    # Check for casatools first
    if CASA_AVAILABLE and 'table' in CASA_IMPORTS:
         tb_tool_factory = CASA_IMPORTS.get('table')
         map_logger.debug("Using casatools.table tool.")
    # Fallback to pyrap/casacore
    elif 'table_tools_module' in CASA_IMPORTS and CASA_IMPORTS['table_tools_module'] is not None:
         pt_module = CASA_IMPORTS['table_tools_module']
         tb_tool_factory = lambda: pt_module.table # Create a factory function
         map_logger.debug("Using pyrap/casacore table function.")
    else:
         map_logger.error("CASA table tools not available. Cannot read cal table SPW map.")
         return spw_map

    if not tb_tool_factory:
         map_logger.error("CASA table tool factory function not found.")
         return spw_map

    spw_table_path = os.path.join(cal_table_path, 'SPECTRAL_WINDOW')
    if not os.path.isdir(spw_table_path):
        map_logger.error(f"SPECTRAL_WINDOW sub-table not found: {spw_table_path}")
        return spw_map

    tb = None
    table_accessor = None # For pyrap/casacore

    try:
        map_logger.info(f"Reading SPW map from cal table: {cal_table_path}")

        # Instantiate differently depending on the library
        if CASA_AVAILABLE and 'table' in CASA_IMPORTS:
             tb = tb_tool_factory()
             tb.open(spw_table_path)
             accessor = tb # Use the tb object directly
        else:
             # pyrap/casacore returns the table object directly
             table_accessor = tb_tool_factory()(spw_table_path) # pt.table(path)
             accessor = table_accessor # Use the returned object

        # Read columns
        all_chan_freqs_hz = accessor.getcol('CHAN_FREQ')
        num_chans_per_spw = accessor.getcol('NUM_CHAN')
        num_spw_in_table = accessor.nrows()

        map_logger.info(f"Found {num_spw_in_table} SPWs defined in the table.")
        map_logger.debug(f"Shape of CHAN_FREQ column read: {all_chan_freqs_hz.shape}")
        map_logger.debug(f"NUM_CHAN per SPW: {num_chans_per_spw}")

        # --- Handle Transposed Array ---
        if all_chan_freqs_hz.ndim == 2 and all_chan_freqs_hz.shape[0] != num_spw_in_table:
            if all_chan_freqs_hz.shape[1] == num_spw_in_table:
                 map_logger.warning("CHAN_FREQ array appears transposed (Shape: NChan x NSPW). Transposing.")
                 all_chan_freqs_hz = all_chan_freqs_hz.T # Transpose to (NSPW x NChan)
                 map_logger.debug(f"Shape after transpose: {all_chan_freqs_hz.shape}")
            else:
                 map_logger.error(f"Unexpected 2D shape for CHAN_FREQ: {all_chan_freqs_hz.shape}. Expected ({num_spw_in_table}, NChan) or (NChan, {num_spw_in_table}). Cannot proceed.")
                 return {}
        elif all_chan_freqs_hz.ndim == 1 and num_spw_in_table == 1:
             map_logger.debug("Reshaping 1D CHAN_FREQ array for single SPW.")
             all_chan_freqs_hz = all_chan_freqs_hz.reshape(1, -1)
        elif all_chan_freqs_hz.ndim != 2 or all_chan_freqs_hz.shape[0] != num_spw_in_table:
             map_logger.error(f"Unexpected shape or dimensions for CHAN_FREQ: {all_chan_freqs_hz.shape}. Expected ({num_spw_in_table}, NChan).")
             return {}
        # --- End Transpose Handling ---

        # Get constants from the config module
        global config
        if 'config' not in globals():
            import pipeline_config as config
            
        all_bands = config.ALL_POSSIBLE_SUB_BANDS
        subband_bw_mhz = config.SUBBAND_BW_MHZ

        for spw_id in range(num_spw_in_table):
            nchan_this_spw = num_chans_per_spw[spw_id]
            if nchan_this_spw > 0:
                if spw_id >= all_chan_freqs_hz.shape[0] or nchan_this_spw > all_chan_freqs_hz.shape[1]:
                     map_logger.error(f"Index out of bounds for SPW {spw_id}. Freq shape: {all_chan_freqs_hz.shape}, NChan: {nchan_this_spw}")
                     continue
                spw_freqs_hz = all_chan_freqs_hz[spw_id, :nchan_this_spw]
                if spw_freqs_hz.size == 0:
                    map_logger.warning(f"Extracted 0 frequencies for SPW {spw_id}. Skipping.")
                    continue

                spw_center_freq_mhz = np.mean(spw_freqs_hz) / 1e6
                spw_min_freq_mhz = spw_freqs_hz.min() / 1e6
                spw_max_freq_mhz = spw_freqs_hz.max() / 1e6
                map_logger.debug(f"Processing Cal Table SPW ID {spw_id} (NChan={nchan_this_spw}, Freq Range: {spw_min_freq_mhz:.2f}-{spw_max_freq_mhz:.2f} MHz, Center: {spw_center_freq_mhz:.2f} MHz)")

                match_found = False
                for band_name in all_bands:
                    try:
                        band_lower_mhz = float(band_name.replace('MHz', ''))
                        band_upper_mhz = band_lower_mhz + subband_bw_mhz
                        tolerance = config.CHAN_BW_KHZ * 1e-3 # 1 channel tolerance
                        if (band_lower_mhz - tolerance) <= spw_center_freq_mhz < (band_upper_mhz + tolerance):
                            map_logger.debug(f"  --> Matched {band_name} (Range: {band_lower_mhz:.1f}-{band_upper_mhz:.1f} MHz)")
                            if band_name in spw_map:
                                 map_logger.warning(f"  --> Sub-band {band_name} maps to multiple SPWs! Overwriting SPW {spw_map[band_name]} with SPW {spw_id}.")
                            spw_map[band_name] = spw_id
                            match_found = True
                            break
                    except ValueError: continue

                if not match_found:
                    map_logger.warning(f"Could not find matching sub-band range for Cal Table SPW ID {spw_id} (Center: {spw_center_freq_mhz:.2f} MHz).")
            else:
                map_logger.warning(f"Skipping Cal Table SPW ID {spw_id} as NUM_CHAN = 0.")

    except Exception as e:
        map_logger.error(f"Failed to read SPECTRAL_WINDOW table: {e}", exc_info=True)
        spw_map = {}
    finally:
        if tb is not None:
            try: tb.close()
            except Exception: pass
        if table_accessor is not None:
             try: table_accessor.close()
             except Exception: pass

    if not spw_map:
        map_logger.error("Failed to generate SPW map from calibration table.")
    else:
        map_logger.info(f"Successfully generated SPW map from cal table: {spw_map}")

    return spw_map


# ==============================================================================
# === Phase Center Utilities (Unchanged) ===
# ==============================================================================
# ... (set_phase_center, check_phase_center unchanged) ...
def set_phase_center(ms_path, context):
    """Sets common phase center using chgcentre tool."""
    util_logger = get_logger('Utils.PhaseCenter')
    target = config.CAL_PHASE_CENTER
    if not target: util_logger.info("CAL_PHASE_CENTER not set. Skipping."); return True
    util_logger.info(f"Setting common phase center. Target: '{target}'")
    try:
        if target.lower() == 'zenith':
            mid_time = context['obs_info']['obs_mid_time']
            zenith_altaz_frame = AltAz(obstime=mid_time, location=config.OVRO_LOCATION)
            target_icrs = SkyCoord(alt=90*u.deg, az=0*u.deg, frame=zenith_altaz_frame).icrs
            ra_str, dec_str = target_icrs.ra.to_string(unit=u.hourangle, sep='hms', precision=3), target_icrs.dec.to_string(unit=u.deg, sep='dms', precision=3)
        elif target in config.PRIMARY_SOURCES:
            source_coord = config.PRIMARY_SOURCES[target]['skycoord']
            ra_str, dec_str = source_coord.ra.to_string(unit=u.hourangle, sep='hms', precision=3), source_coord.dec.to_string(unit=u.deg, sep='dms', precision=3)
        else: util_logger.error(f"Invalid CAL_PHASE_CENTER: '{target}'."); return False
        chgcentre_cmd = [config.CHGCENTRE_PATH, ms_path, ra_str, dec_str]
        # NOTE: This call will use the fallback logger, not a passed one.
        # This is OK for this tool as it's not the one failing.
        if not run_command(chgcentre_cmd, task_name=f"chgcentre (RA={ra_str}, Dec={dec_str})", logger=util_logger): 
            util_logger.error("chgcentre tool failed."); return False
    except Exception as e: util_logger.error(f"Unexpected error: {e}", exc_info=True); return False
    util_logger.info("Phase center correction applied successfully."); return True

def check_phase_center(ms_path, context):
    """Verifies MS phase center against CAL_PHASE_CENTER config."""
    util_logger = get_logger('Utils.PhaseCenter'); target = config.CAL_PHASE_CENTER
    if not target: util_logger.info("CAL_PHASE_CENTER not set. Skipping."); return True
    try:
        if target.lower() == 'zenith':
            mid_time = context['obs_info']['obs_mid_time']
            zenith_altaz_frame = AltAz(obstime=mid_time, location=config.OVRO_LOCATION)
            target_icrs = SkyCoord(alt=90*u.deg, az=0*u.deg, frame=zenith_altaz_frame).icrs
        elif target in config.PRIMARY_SOURCES: target_icrs = config.PRIMARY_SOURCES[target]['skycoord']
        else: util_logger.error(f"Invalid CAL_PHASE_CENTER: '{target}'."); return False
    except Exception as e: util_logger.error(f"Failed to calculate target coords: {e}"); return False
    actual_skycoord = None
    try:
        chgcentre_cmd = [config.CHGCENTRE_PATH, ms_path]
        stdout = run_command(chgcentre_cmd, task_name="Check Phase Center", return_output=True, logger=util_logger)
        if stdout is None: util_logger.error("chgcentre command failed."); return False
        coord_pattern = re.compile(r'(\d{1,2}h\d{1,2}m\d{1,2}\.\d+s) (\S\d{1,2}d\d{1,2}m\d{1,2}\.\d+s)')
        match = coord_pattern.search(stdout)
        if not match: util_logger.error("Could not parse RA/Dec from chgcentre."); util_logger.debug(f"Output: {stdout}"); return False
        ra_str, dec_str = match.groups(); util_logger.info(f"Found phase center: RA={ra_str} Dec={dec_str}")
        actual_skycoord = SkyCoord(f'{ra_str} {dec_str}', frame='icrs', unit=(u.hourangle, u.deg))
    except Exception as e: util_logger.error(f"Failed to read/parse phase center: {e}", exc_info=True); return False
    separation_arcsec = target_icrs.separation(actual_skycoord).to_value(u.arcsec)
    if separation_arcsec < 1.0: util_logger.info(f"Phase center confirmed. Sep: {separation_arcsec:.3f} arcsec"); return True
    else:
        util_logger.error(f"Phase center mismatch! Target: {target_icrs.to_string('hmsdms')}, Actual: {actual_skycoord.to_string('hmsdms')}, Sep: {separation_arcsec:.3f} arcsec"); return False

# ==============================================================================
# === Calibration Utilities (Unchanged) ===
# ==============================================================================
# ... (determine_calibration_uv_range unchanged) ...
def determine_calibration_uv_range(context):
    """Determines uvrange string based on sources in sky model."""
    modeled_sources = context.get('model_sources', [])
    if 'VirA' in modeled_sources: uvrange_str = config.CAL_UVRANGE_VIRA; logger.info("Virgo A detected. Using VirA-specific uvrange.")
    else:
        uvrange_str = config.CAL_UVRANGE_DEFAULT
        if not modeled_sources: logger.info("Model empty. Using default uvrange.")
        else: logger.info("Virgo A not detected. Using default uvrange.")
    logger.info(f"Selected calibration uvrange: {uvrange_str}"); return uvrange_str

# ==============================================================================
# === Beam Model Utilities (Unchanged) ===
# ==============================================================================
# ... (get_beam_interpolator, calculate_beam_attenuation unchanged) ...
def get_beam_interpolator(beam_model_path):
    """Loads HDF5 beam model and creates 3D interpolator."""
    logger.info(f"Loading and interpolating beam model from {beam_model_path}...")
    if not os.path.exists(beam_model_path): logger.error(f"Beam file not found: {beam_model_path}"); return None
    try:
        with h5py.File(beam_model_path, 'r') as hf:
            if not all(key in hf for key in ['freq_Hz', 'theta_pts', 'phi_pts', 'X_pol_Efields/etheta']): raise KeyError("Invalid HDF5 beam file.")
            fq_orig_hz = hf['freq_Hz'][:]; th_orig_rad = hf['theta_pts'][:]; ph_orig_rad = hf['phi_pts'][:]
            Exth = hf['X_pol_Efields/etheta'][:]; Exph = hf['X_pol_Efields/ephi'][:]
            Eyth = hf['Y_pol_Efields/etheta'][:]; Eyph = hf['Y_pol_Efields/ephi'][:]
        PbX = np.abs(Exth)**2 + np.abs(Exph)**2; PbY = np.abs(Eyth)**2 + np.abs(Eyph)**2
        power_beam_unnormalized = PbX + PbY
        fq_s_idx, th_s_idx, ph_s_idx = np.argsort(fq_orig_hz), np.argsort(th_orig_rad), np.argsort(ph_orig_rad)
        fq_s_mhz, th_s_rad, ph_s_rad = (fq_orig_hz[fq_s_idx]) / 1e6, th_orig_rad[th_s_idx], ph_orig_rad[ph_s_idx]
        power_beam_sorted = power_beam_unnormalized[fq_s_idx, :, :][:, th_s_idx, :][:, :, ph_s_idx]
        zenith_theta_idx = np.argmin(np.abs(th_s_rad))
        zenith_gain_vs_freq = power_beam_sorted[:, zenith_theta_idx, 0]
        zenith_gain_vs_freq[zenith_gain_vs_freq == 0] = 1e-9
        power_beam_normalized = power_beam_sorted / zenith_gain_vs_freq[:, np.newaxis, np.newaxis]
        interpolator = RegularGridInterpolator((fq_s_mhz, th_s_rad, ph_s_rad), power_beam_normalized, method='linear', bounds_error=False, fill_value=0.0)
        logger.info("Beam interpolator created."); return interpolator
    except Exception as e: logger.error(f"Error loading beam model {beam_model_path}: {e}", exc_info=True); return None

def calculate_beam_attenuation(context, source_name, freqs_mhz, beam_interpolator):
    """Calculates beam attenuation (power gain) for a source."""
    if beam_interpolator is None: return np.ones_like(freqs_mhz, dtype=float)
    source_details = config.PRIMARY_SOURCES.get(source_name)
    if not (source_details and 'skycoord' in source_details): logger.error(f"Coords not found for {source_name}."); return np.ones_like(freqs_mhz, dtype=float)
    source_coords = source_details['skycoord']; obs_time = context['obs_info'].get('obs_mid_time')
    if not obs_time: logger.error(f"Obs time missing."); return np.ones_like(freqs_mhz, dtype=float)
    altaz_frame = AltAz(obstime=obs_time, location=config.OVRO_LOCATION)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="astropy._erfa"); source_altaz = source_coords.transform_to(altaz_frame)
    za_rad, az_rad = np.pi/2 - source_altaz.alt.rad, source_altaz.az.rad
    if za_rad < 0: za_rad = 0.0
    freqs_mhz = np.atleast_1d(freqs_mhz); points = np.vstack((freqs_mhz, np.full_like(freqs_mhz, za_rad), np.full_like(freqs_mhz, az_rad))).T
    try: power_gain = beam_interpolator(points); power_gain[power_gain < 1e-6] = 0.0; return power_gain
    except Exception as e: logger.error(f"Error during beam interpolation for {source_name}: {e}"); return np.ones_like(freqs_mhz, dtype=float)

# ==============================================================================
# === Data Mapping and Diagnostics (Unchanged) ===
# ==============================================================================
# ... (load_antenna_mapping, map_delays_to_snap2, run_snap2_diagnostics unchanged) ...
def load_antenna_mapping():
    """Loads antenna mapping from config file."""
    mapping = config.ANTENNA_MAPPING
    if not mapping: logger.error("Antenna mapping empty or missing in config."); return None
    get_logger('Utils.Mapping').info(f"Loaded {len(mapping)} antenna mappings from config."); return mapping

def map_delays_to_snap2(delay_data, antenna_mapping):
    """Maps delay data to SNAP2 board groups."""
    snap2_groups = {}
    for ant_id, delay_value in delay_data.items():
        location_str = antenna_mapping.get(ant_id)
        if not location_str: logger.warning(f"Ant ID {ant_id} not in mapping."); continue
        match = re.match(r'(R\d+)', location_str)
        if match:
            rack_id = match.group(1)
            if rack_id not in snap2_groups: snap2_groups[rack_id] = []
            snap2_groups[rack_id].append((ant_id, delay_value))
        else: logger.warning(f"Could not parse Rack ID from {location_str} for Ant {ant_id}")
    return snap2_groups

def run_snap2_diagnostics(problematic_delays, antenna_mapping, threshold_percent=config.SNAP2_FAILURE_THRESHOLD_PERCENT):
    """Runs diagnostics on SNAP2 boards based on problematic delays."""
    diag_logger = get_logger('QA.Diagnostics'); diag_logger.info("Running SNAP2 board diagnostics.")
    problematic_groups = map_delays_to_snap2(problematic_delays, antenna_mapping)
    total_counts_per_group = {}
    for location_str in antenna_mapping.values():
        match = re.match(r'(R\d+)', location_str)
        if match: rack_id = match.group(1); total_counts_per_group[rack_id] = total_counts_per_group.get(rack_id, 0) + 1
    failed_boards = []
    if total_counts_per_group: all_groups = sorted(list(total_counts_per_group.keys()), key=lambda x: int(x[1:]))
    else: all_groups = []; diag_logger.warning("No groups found in mapping.")
    for rack_id in all_groups:
        total_antennas = total_counts_per_group[rack_id]; failed_antennas = problematic_groups.get(rack_id, [])
        num_failed = len(failed_antennas); failure_rate = (num_failed / total_antennas) * 100 if total_antennas > 0 else 0
        diag_logger.info(f"SNAP2 Group {rack_id}: {num_failed}/{total_antennas} problematic ({failure_rate:.0f}%).")
        if failure_rate >= threshold_percent: diag_logger.error(f"CRITICAL: SNAP2 Group {rack_id} exceeds threshold ({threshold_percent}%)."); failed_boards.append(rack_id)
    if not failed_boards and all_groups: diag_logger.info("SNAP2 diagnostics completed. No boards exceeded threshold.")
    elif failed_boards: diag_logger.warning(f"SNAP2 diagnostics completed. {len(failed_boards)} boards exceeded threshold.")
    return failed_boards

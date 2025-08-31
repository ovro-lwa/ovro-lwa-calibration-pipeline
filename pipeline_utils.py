# pipeline_utils.py
import os
import sys
import logging
import subprocess
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shutil
import glob
import datetime
import re

# Astropy imports
from astropy.time import Time
import astropy.units as u
try:
    from astropy.io import fits
    FITS_AVAILABLE = True
except ImportError:
    FITS_AVAILABLE = False
    print("WARNING: astropy.io.fits not available. FITS operations (Imaging QA) will fail.", file=sys.stderr)

# Scipy imports
try:
    from scipy.signal import medfilt
    SCIPY_AVAILABLE = True
except ImportError:
    medfilt = None
    SCIPY_AVAILABLE = False
    print("WARNING: scipy.signal.medfilt not available. RFI detection in extrapolation will fail.", file=sys.stderr)

# Import configuration
try:
    import pipeline_config as config
except ImportError:
    print("ERROR: pipeline_config.py not found.", file=sys.stderr)
    sys.exit(1)

# ==============================================================================
# === CASA Environment Initialization (Robust Abstraction) ===
# ==============================================================================
CASA_AVAILABLE = False
TABLE_TOOLS_AVAILABLE = False
CASA_IMPORTS = {}
_ACTIVE_TABLE_LIB = None # Internal state: 'casatools', 'pyrap', 'casatables'

try:
    # --- Import CASA Tools (Factories) and Tasks ---
    from casatools import table, msmetadata, componentlist, measures, quantity
    from casatasks import (imfit, imstat, applycal, bandpass, gaincal,
                           concat, mstransform, flagdata, ft)
    
    CASA_IMPORTS['table_factory'] = table
    CASA_IMPORTS['msmetadata_factory'] = msmetadata
    CASA_IMPORTS['componentlist_factory'] = componentlist
    CASA_IMPORTS['measures_factory'] = measures
    CASA_IMPORTS['quantity_factory'] = quantity
    CASA_IMPORTS.update({
        'imfit': imfit, 'imstat': imstat, 'applycal': applycal, 'bandpass': bandpass, 
        'gaincal': gaincal, 'concat': concat, 'mstransform': mstransform, 
        'flagdata': flagdata, 'ft': ft
    })
    CASA_AVAILABLE = True
    TABLE_TOOLS_AVAILABLE = True
    _ACTIVE_TABLE_LIB = 'casatools'

    # Suppress verbose CASA logging
    try:
        from casatasks.private import casa_transition as ct
        ct.logger_wrapper(origin="casatools").setLevel('ERROR')
        ct.logger_wrapper(origin="casatasks").setLevel('WARN')
    except ImportError:
        pass 
except ImportError as e:
    CASA_IMPORTS['import_error'] = str(e)
    print(f"WARNING: Full CASA environment import failed: {e}.", file=sys.stderr)

    # --- Fallback for Table Tools ---
    if not TABLE_TOOLS_AVAILABLE:
        try:
            import pyrap.tables as pt
            CASA_IMPORTS['table_module'] = pt
            TABLE_TOOLS_AVAILABLE = True
            _ACTIVE_TABLE_LIB = 'pyrap'
            print("INFO: Using 'pyrap.tables' for table manipulation.", file=sys.stderr)
        except ImportError:
            try:
                import casatables as ct_tables
                CASA_IMPORTS['table_module'] = ct_tables
                TABLE_TOOLS_AVAILABLE = True
                _ACTIVE_TABLE_LIB = 'casatables'
                print("INFO: Using 'casatables' for table manipulation.", file=sys.stderr)
            except ImportError:
                print("WARNING: No table manipulation tools found. Table operations will fail.", file=sys.stderr)

# ==============================================================================
# === Logging Setup ===
# ==============================================================================
def get_logger(name):
    """Returns a logger instance within the pipeline hierarchy."""
    if not name.startswith('OVRO_Pipeline'):
        full_name = f'OVRO_Pipeline.{name}'
    else:
        full_name = name
    return logging.getLogger(full_name)

logger = get_logger('Utils')

# ==============================================================================
# === Table Abstraction Helper ===
# ==============================================================================
def _get_table_context(ms_path, readonly=True, ack=False):
    """
    Provides a context manager for table access, abstracting library differences.
    """
    if not TABLE_TOOLS_AVAILABLE:
        raise ImportError("No table manipulation tools available.")
        
    if _ACTIVE_TABLE_LIB == 'casatools':
        tb_factory = CASA_IMPORTS['table_factory']
        tb_instance = tb_factory()
        tb_instance.open(ms_path, nomodify=readonly, acknowledge=ack)
        return tb_instance
    else:
        pt_module = CASA_IMPORTS['table_module']
        return pt_module.table(ms_path, readonly=readonly, ack=ack)

# ==============================================================================
# === Data Discovery and Metadata ===
# ==============================================================================
def discover_ms_files(data_folder):
    """Recursively finds and sorts raw .ms files."""
    ms_files = glob.glob(os.path.join(data_folder, '**/*.ms'), recursive=True)
    raw_ms_files = [f for f in ms_files if not ('fullband_' in f or 'intermediate_concat' in f)]
    raw_ms_files.sort()
    return raw_ms_files

def calculate_observation_metadata(ms_files):
    """Calculates observation times and LST by reading MS metadata (Robust)."""
    if not ms_files: return None
    
    start_times, end_times = [], []
    for msfile in ms_files:
        try:
            with _get_table_context(msfile, readonly=True) as t:
                times = t.getcol('TIME')
                if times.size > 0:
                    start_times.append(np.min(times))
                    end_times.append(np.max(times))
        except Exception as e:
            logger.error(f"Failed to read TIME column from {msfile}: {e}")
            return None

    if not start_times:
        logger.error("No valid time data found in any MS files.")
        return None

    start_time_mjd_s = min(start_times)
    end_time_mjd_s = max(end_times)
    mid_time_mjd_s = (start_time_mjd_s + end_time_mjd_s) / 2.0
    
    start_time_obj = Time(start_time_mjd_s / 86400.0, format='mjd', scale='utc')
    mid_time_obj = Time(mid_time_mjd_s / 86400.0, format='mjd', scale='utc', location=config.OVRO_LWA_LOCATION)
    
    lst = mid_time_obj.sidereal_time('apparent')
    lst_hour_int = int(round(lst.h))
    lst_hour_str = f"{lst_hour_int:02d}h"

    metadata = {
        'ms_files': ms_files,
        'start_time_dt': start_time_obj.datetime,
        'mid_time_dt': mid_time_obj.datetime,
        'duration_minutes': (end_time_mjd_s - start_time_mjd_s) / 60.0,
        'obs_date': start_time_obj.strftime('%Y%m%d'),
        'lst_hour': lst_hour_str,
        'lst_hour_int': lst_hour_int,
    }
    return metadata

# ==============================================================================
# === Pipeline Context and Directory Setup ===
# ==============================================================================
def setup_pipeline_context_phase1(data_folder, script_dir):
    """
    Sets up the initial pipeline context dictionary and creates the main
    working directory based on the data folder name.
    
    Returns the context dictionary or None on failure.
    """
    context = {'script_dir': script_dir}
    
    try:
        data_basename = os.path.basename(os.path.normpath(data_folder))
        if not data_basename:
            raise ValueError("Could not determine a valid directory name from the input data folder path.")
            
        working_dir = os.path.join(config.PARENT_OUTPUT_DIR, data_basename)
        context['working_dir'] = working_dir
        
        # Create directories
        os.makedirs(working_dir, exist_ok=True)
        
        # Define other key directories
        context['qa_dir'] = os.path.join(working_dir, 'QA')
        context['cal_tables_dir'] = os.path.join(working_dir, 'cal_tables')
        context['ms_dir'] = os.path.join(working_dir, 'MS')
        
        os.makedirs(context['qa_dir'], exist_ok=True)
        os.makedirs(context['cal_tables_dir'], exist_ok=True)
        os.makedirs(context['ms_dir'], exist_ok=True)

        return context

    except Exception as e:
        # Use a temporary logger or print since the full logger isn't set up yet
        print(f"CRITICAL ERROR during Phase 1 setup: {e}", file=sys.stderr)
        return None

def setup_pipeline_context_phase2(context, verbose=False):
    """
    Sets up the logging for the pipeline using the directories created in Phase 1.
    """
    try:
        log_filename = f"pipeline_log_{os.path.basename(context['working_dir'])}.log"
        log_filepath = os.path.join(context['working_dir'], log_filename)
        context['log_filepath'] = log_filepath

        # Configure the root logger for the pipeline
        log_level = logging.DEBUG if verbose else logging.INFO
        
        # Prevent duplicate handlers if logger is already configured
        root_logger = logging.getLogger('OVRO_Pipeline')
        if not root_logger.handlers:
            root_logger.setLevel(log_level)
            
            # Console Handler (prints INFO and above)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
            ch.setFormatter(ch_formatter)
            root_logger.addHandler(ch)
            
            # File Handler (prints all messages at the specified level)
            fh = logging.FileHandler(log_filepath, mode='a')
            fh.setLevel(log_level)
            fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(fh_formatter)
            root_logger.addHandler(fh)
        
        main_logger = get_logger('Master')
        main_logger.info("="*60)
        main_logger.info(f"Pipeline working directory: {context['working_dir']}")
        main_logger.info(f"Log file: {log_filepath}")
        main_logger.info("="*60)

        return context
        
    except Exception as e:
        print(f"CRITICAL ERROR during Phase 2 setup (logging): {e}", file=sys.stderr)
        return None

# ==============================================================================
# === External Command Execution ===
# ==============================================================================
def run_external_command(cmd, description, logger_name, env_vars=None):
    """
    Executes an external command, streaming its output and logging details.
    Raises RuntimeError on failure.
    """
    local_logger = logging.getLogger(logger_name)
    local_logger.info(f"Starting: {description}")
    
    if isinstance(cmd, str):
        local_logger.error("Command must be a list of arguments, not a single string.")
        raise ValueError("Command must be a list of arguments.")
        
    local_logger.debug(f"Executing Command: {' '.join(cmd)}")

    process_env = os.environ.copy()
    if env_vars:
        process_env.update(env_vars)

    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              bufsize=1, universal_newlines=True, env=process_env) as p:
            
            with tqdm(p.stdout, desc=f"Running {os.path.basename(cmd[0])}", unit="line", 
                      bar_format="{l_bar}{bar}| {elapsed}") as bar:
                for line in bar:
                    line = line.strip()
                    if line:
                        local_logger.debug(f"[EXT] {line}")

        if p.returncode != 0:
            raise RuntimeError(f"{description} failed with return code {p.returncode}. Check detailed logs.")
        
        local_logger.info(f"{description} completed successfully.")
        return True

    except FileNotFoundError:
        local_logger.error(f"Command not found: {cmd[0]}. Please check the path in pipeline_config.py.")
        raise
    except Exception as e:
        local_logger.error(f"An unexpected error occurred while running '{' '.join(cmd)}': {e}")
        raise

# ==============================================================================
# === Calibration & QA Utilities (Core) ===
# ==============================================================================
def read_cal_table(table_path):
    """Reads a CASA calibration table and returns a dictionary of its contents."""
    if not TABLE_TOOLS_AVAILABLE:
        logger.error("CASA table tools not available. Cannot read calibration table.")
        return None
        
    try:
        with _get_table_context(table_path, readonly=True) as t:
            data = {
                'ant1': t.getcol('ANTENNA1'),
                'gain': t.getcol('CPARAM'),
                'flag': t.getcol('FLAG'),
                'snr': t.getcol('SNR')
            }
        return data
    except Exception as e:
        logger.error(f"Failed to read CASA table {table_path}: {e}")
        return None

def find_nearest_reference_table(cal_type, lst_hour_int):
    """
    Finds the nearest reference calibration table based on LST hour.
    cal_type should be 'delay' or 'bandpass'.
    """
    if cal_type not in ['delay', 'bandpass']:
        raise ValueError("cal_type must be 'delay' or 'bandpass'")

    base_dir = os.path.join(config.REFERENCE_CALIBRATION_DIR, cal_type)
    if not os.path.exists(base_dir):
        logger.warning(f"Reference directory for '{cal_type}' does not exist: {base_dir}")
        return None

    lst_dirs = glob.glob(os.path.join(base_dir, '[0-9]*h'))
    if not lst_dirs:
        logger.warning(f"No LST directories found in {base_dir}")
        return None

    available_lsts = []
    for d in lst_dirs:
        try:
            hour = int(os.path.basename(d).replace('h', ''))
            available_lsts.append(hour)
        except ValueError:
            continue

    if not available_lsts:
        logger.warning(f"No valid LST hour directories found in {base_dir}")
        return None

    # Find the closest LST (handles wraparound)
    diffs = [min(abs(lst - lst_hour_int), 24 - abs(lst - lst_hour_int)) for lst in available_lsts]
    closest_lst_hour = available_lsts[np.argmin(diffs)]
    
    closest_dir = os.path.join(base_dir, f"{closest_lst_hour:02d}h")
    
    # Find any table inside that directory
    tables = glob.glob(os.path.join(closest_dir, '*'))
    if not tables:
        logger.warning(f"No reference tables found in the nearest LST directory: {closest_dir}")
        return None
    
    # Return the first one found
    ref_table = tables[0]
    logger.info(f"Found nearest reference {cal_type} table for LST {lst_hour_int}h: {os.path.basename(ref_table)} (from {closest_lst_hour:02d}h LST bin)")
    return ref_table

def detect_outliers_medfilt(data, kernel_size=5, sigma_threshold=3.0):
    """Detects outliers in 1D data using a median filter."""
    if not SCIPY_AVAILABLE:
        logger.error("scipy.signal.medfilt not available for outlier detection.")
        return np.zeros_like(data, dtype=bool)
    if len(data) <= kernel_size:
        return np.zeros_like(data, dtype=bool)

    # Use median filter to estimate the smooth underlying trend
    smoothed = medfilt(data, kernel_size=kernel_size)
    # Calculate deviations from the trend
    deviations = np.abs(data - smoothed)
    # Calculate statistics for deviations
    median_dev = np.median(deviations)
    std_dev = np.std(deviations)
    
    threshold = median_dev + sigma_threshold * std_dev
    outliers = deviations > threshold
    return outliers

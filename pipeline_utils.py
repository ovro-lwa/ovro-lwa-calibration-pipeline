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
import numpy as np
from tqdm import tqdm

# Matplotlib configuration for non-interactive plotting
try:
    import matplotlib
    matplotlib.use('Agg') # Use a non-interactive backend
    import matplotlib.pyplot as plt
    _plotting_available = True
except ImportError:
    _plotting_available = False
    print("WARNING: Matplotlib not found. QA plotting will be disabled.", file=sys.stderr)


# Import configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import pipeline_config as config

# ==============================================================================
# === CASA Availability Check (Centralized and Comprehensive) ===
# ==============================================================================
# The "Factory" approach, robustly handling the quantity/quanta name change.

CASA_AVAILABLE = False
TABLE_TOOLS_AVAILABLE = False
CASA_IMPORTS = {}

try:
    # 1. Check core casatools
    import casatools
    CASA_IMPORTS['casatools'] = casatools

    # 2. Define tool names, including the quantity/quanta fallback
    tool_names = ['measures', 'msmetadata', 'componentlist']
    
    # Determine the name for the quantity/quanta tool
    quantity_tool_name = None
    if hasattr(casatools, 'quantity'):
        quantity_tool_name = 'quantity'
    elif hasattr(casatools, 'quanta'):
        quantity_tool_name = 'quanta'
    
    if quantity_tool_name:
         tool_names.append(quantity_tool_name)
    else:
        # This is critical as we rely on it in add_sky_model.py
        raise ImportError("Neither 'quantity' nor 'quanta' found in casatools.")

    
    # 3. Check specific tools by attempting instantiation via the factory method
    for factory_name in tool_names:
        if hasattr(casatools, factory_name):
            factory_func = getattr(casatools, factory_name)
            factory_func() # Attempt instantiation to ensure it's functional
            # Store the factory function for later use
            # We will always store the quantity tool under the key 'quantity_factory'
            key_name = 'quantity_factory' if factory_name == quantity_tool_name else f'{factory_name}_factory'
            CASA_IMPORTS[key_name] = factory_func
        else:
            raise AttributeError(f"casatools does not have attribute '{factory_name}'")

    # 4. Check for the logger function
    logger_func = None
    if hasattr(casatools, 'logger'):
        logger_func = casatools.logger
    elif hasattr(casatools, 'logsink'):
        logger_func = casatools.logsink
    else:
         raise ImportError("Cannot find casatools logger or logsink.")
    CASA_IMPORTS['logger_func'] = logger_func

    # 5. Check required casatasks
    # Added gaincal, bandpass, and applycal
    from casatasks import concat, flagdata, mstransform, ft, gaincal, bandpass, applycal
    
    CASA_IMPORTS['concat'] = concat
    CASA_IMPORTS['flagdata'] = flagdata
    CASA_IMPORTS['mstransform'] = mstransform
    CASA_IMPORTS['ft'] = ft
    CASA_IMPORTS['gaincal'] = gaincal
    CASA_IMPORTS['bandpass'] = bandpass
    CASA_IMPORTS['applycal'] = applycal

    CASA_AVAILABLE = True

except (ImportError, AttributeError, Exception) as e:
    CASA_AVAILABLE = False
    error_msg = f"{type(e).__name__}: {e}"
    CASA_IMPORTS['error'] = error_msg


# 6. Check for table manipulation tools (Crucial for Data Prep and QA)
try:
    import casatables as pt
    TABLE_TOOLS_AVAILABLE = True
    CASA_IMPORTS['table_tools'] = pt
except ImportError:
    try:
        import pyrap.tables as pt
        TABLE_TOOLS_AVAILABLE = True
        CASA_IMPORTS['table_tools'] = pt
    except ImportError:
        try:
            # Fallback often used in ORCA/older environments
            import casacore.tables as pt
            TABLE_TOOLS_AVAILABLE = True
            CASA_IMPORTS['table_tools'] = pt
        except ImportError as e:
            TABLE_TOOLS_AVAILABLE = False
            CASA_IMPORTS['table_tools_error'] = str(e)

# ==============================================================================
# === Logging and Execution Utilities ===
# ==============================================================================

# (setup_logging, get_logger, run_external_command, initialize_casalog remain the same as the previously working version)
def setup_logging(working_dir, log_filename="pipeline.log", verbose=False):
    """Configures logging to file and console."""
    qa_dir = os.path.join(working_dir, "QA")
    os.makedirs(qa_dir, exist_ok=True)
    log_filepath = os.path.join(qa_dir, log_filename)
    logger = logging.getLogger('OVRO_Pipeline')
    logger.propagate = False 
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger, log_filepath

    fh = logging.FileHandler(log_filepath)
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(file_formatter)

    ch = logging.StreamHandler(sys.stdout)
    console_level = logging.DEBUG if verbose else logging.INFO
    ch.setLevel(console_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(console_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logging initialized. Detailed log saving to: {log_filepath}")
    return logger, log_filepath

def get_logger(module_name):
    """Utility function for modules to get their specific logger."""
    return logging.getLogger(f'OVRO_Pipeline.{module_name}')

def run_external_command(command, description="External command", logger_name='OVRO_Pipeline.Subprocess', return_output=False):
    """Executes a command and logs STDOUT/STDERR in real-time."""
    logger = logging.getLogger(logger_name)
    command_str_list = [str(c) for c in command]
    logger.info(f"Starting: {description}")
    logger.debug(f"Executing Command: {' '.join(command_str_list)}")
    output_lines = []
    try:
        process = subprocess.Popen(
            command_str_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        if process.stdout:
            tool_name = os.path.basename(command[0])
            with tqdm(desc=f"Running {tool_name}", unit="line", dynamic_ncols=True) as pbar:
                for line in iter(process.stdout.readline, ''):
                    stripped_line = line.strip()
                    if stripped_line:
                        output_lines.append(stripped_line)
                        logger.debug(f"[EXT] {stripped_line}")
                        pbar.update(1)
        process.wait()
    except FileNotFoundError:
        logger.error(f"Command not found: {command[0]}. Check configuration paths.")
        raise RuntimeError(f"Executable not found for {description}.")
    except Exception as e:
        logger.exception(f"Error during execution of {description}.")
        raise RuntimeError(f"Execution error during {description}: {e}")

    if process.returncode != 0:
        # Improved error handling
        if process.returncode < 0:
            import signal
            try:
                # Use getattr for compatibility if signal.Signals is not fully available
                signal_name = getattr(signal.Signals(-process.returncode), 'name', f"Signal {-process.returncode}")
                error_msg = f"{description} terminated by signal: {signal_name}."
            except (ValueError, AttributeError):
                 error_msg = f"{description} failed with return code {process.returncode}."
        else:
            error_msg = f"{description} failed with return code {process.returncode}."
            
        logger.error(error_msg + " Check detailed logs.")
        raise RuntimeError(error_msg)
    else:
        logger.info(f"{description} completed successfully.")
        if return_output:
            return "\n".join(output_lines)

def initialize_casalog():
    """Initializes the CASA logger instance robustly."""
    if not CASA_AVAILABLE:
        # Log the specific error if available
        if 'error' in CASA_IMPORTS:
             try:
                 logging.getLogger('OVRO_Pipeline.CASA_Check').debug(f"CASA not initialized. Error: {CASA_IMPORTS['error']}")
             except:
                 pass
        return None
    
    casalog_instance = None
    try:
        casalog_instance = CASA_IMPORTS['logger_func']()
    except Exception as e:
        try:
            logging.getLogger('OVRO_Pipeline').warning(f"Could not initialize CASA logger: {e}")
        except:
            pass
    return casalog_instance

# ==============================================================================
# === MS Metadata and QA Utilities (NEW) ===
# ==============================================================================

def get_ms_frequency_metadata(ms_path):
    """
    Reads frequency metadata from the MS SPECTRAL_WINDOW table using table tools (pt).
    Returns a dictionary mapping SPW ID to frequency information.
    """
    logger = get_logger('Utils.Metadata')
    
    if not TABLE_TOOLS_AVAILABLE:
        logger.error("Table tools (casatables/pyrap.tables/casacore.tables) not available. Cannot read MS metadata.")
        return None

    pt = CASA_IMPORTS['table_tools']
    spw_info = {}

    try:
        spw_table_path = os.path.join(ms_path, 'SPECTRAL_WINDOW')
        if not os.path.exists(spw_table_path):
            logger.error(f"SPECTRAL_WINDOW table not found in MS: {ms_path}")
            return None

        # Open the SPECTRAL_WINDOW subtable
        # Use ack=False for compatibility/less verbosity
        with pt.table(spw_table_path, ack=False) as t:
            # Iterate over rows (each row is an SPW)
            for spw_id in range(t.nrows()):
                chan_freqs = t.getcell('CHAN_FREQ', spw_id)
                
                if chan_freqs is None or len(chan_freqs) == 0:
                    logger.warning(f"Skipping SPW {spw_id}: No channel frequencies found.")
                    continue
                
                spw_info[spw_id] = {
                    'num_chans': len(chan_freqs),
                    'freqs_hz': chan_freqs,
                    'min_freq_hz': np.min(chan_freqs),
                    'max_freq_hz': np.max(chan_freqs),
                }
        
        if not spw_info:
            logger.error("No valid spectral window information found in the MS.")
            return None
            
        return spw_info

    except Exception as e:
        logger.exception(f"Error reading SPECTRAL_WINDOW table from {ms_path}")
        return None

def read_cal_table(table_path):
    """
    Reads a CASA calibration table using table tools (pt) and returns data, flags, antennas, and metadata.
    """
    logger = get_logger('Utils.QA')
    if not TABLE_TOOLS_AVAILABLE:
        logger.error("Table tools not available. Cannot read calibration table.")
        return None
    
    if not os.path.exists(table_path):
        logger.error(f"Calibration table not found: {table_path}")
        return None

    pt = CASA_IMPORTS['table_tools']
    try:
        # 1. Read main table data
        with pt.table(table_path, ack=False) as t:
            # Determine the data column (CPARAM for complex gains, FPARAM for floats like Delays)
            colnames = t.colnames()
            if 'CPARAM' in colnames:
                data_col = 'CPARAM'
            elif 'FPARAM' in colnames:
                data_col = 'FPARAM'
            else:
                logger.error(f"Neither CPARAM nor FPARAM found in table {table_path}")
                return None

            # Data shape is typically (N_rows, N_chan, N_pol)
            data = t.getcol(data_col)
            flags = t.getcol('FLAG')
            antenna1 = t.getcol('ANTENNA1')
            # Time/SPW info might be needed later for complex QA
            # time = t.getcol('TIME')
            # spw = t.getcol('SPECTRAL_WINDOW_ID')

            # Get metadata from keywords
            keywords = t.getkeywords()
            vis_cal_type = keywords.get('VisCal', 'Unknown')

        # 2. Read the ANTENNA subtable for names
        ant_names = {}
        ant_table_path = os.path.join(table_path, 'ANTENNA')
        if os.path.exists(ant_table_path):
            with pt.table(ant_table_path, ack=False) as t_ant:
                names = t_ant.getcol('NAME')
                for i, name in enumerate(names):
                    ant_names[i] = name
        else:
            logger.warning(f"ANTENNA subtable not found in {table_path}. Using indices as names.")
            unique_ants = np.unique(antenna1)
            for ant_idx in unique_ants:
                ant_names[ant_idx] = str(ant_idx)


        return {
            'data': data,
            'flags': flags,
            'antenna1_idx': antenna1,
            'antenna_names': ant_names,
            'type': vis_cal_type,
            'data_column': data_col
        }
    except Exception as e:
        logger.exception(f"Failed to read calibration table {table_path}")
        return None

def plot_delays(cal_data, output_filepath, ref_cal_data=None, problematic_ants=None):
    """
    Generates a plot of Delay (K) solutions, optionally comparing against a reference
    and highlighting problematic antennas.
    """
    logger = get_logger('Utils.QA')
    if not _plotting_available:
        logger.warning("Matplotlib not available. Skipping plot generation.")
        return False
    
    if problematic_ants is None:
        problematic_ants = set()
    
    # --- Prepare Data ---
    # Extract unflagged delays from the current calibration table
    current_delays = {
        ant: data[0, 0] for ant, data, flag in zip(cal_data['antenna1_idx'], cal_data['data'], cal_data['flags']) if not flag[0, 0]
    }
    
    # Extract unflagged delays from the reference table, if provided
    ref_delays = {}
    if ref_cal_data:
        ref_delays = {
            ant: data[0, 0] for ant, data, flag in zip(ref_cal_data['antenna1_idx'], ref_cal_data['data'], ref_cal_data['flags']) if not flag[0, 0]
        }

    # Get a sorted, unique list of all antenna indices from both tables
    all_antennas = sorted(list(set(current_delays.keys()) | set(ref_delays.keys())))
    
    current_plot_vals = [current_delays.get(ant, np.nan) for ant in all_antennas]
    ref_plot_vals = [ref_delays.get(ant, np.nan) for ant in all_antennas]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(20, 9))
    
    table_name = os.path.basename(output_filepath).replace('QA_plot_delays_', '').replace('.png', '')

    # Plot reference data if available
    if ref_delays:
        ax.plot(all_antennas, ref_plot_vals, 'o', color='gray', alpha=0.7, label='Reference Delays')

    # Plot current data
    ax.plot(all_antennas, current_plot_vals, '.', color='blue', label=f'Current Delays ({table_name})')
    
    # Highlight problematic antennas if any were found
    if problematic_ants:
        problematic_delays = [current_delays.get(ant, np.nan) for ant in all_antennas if ant in problematic_ants]
        problematic_indices = [ant for ant in all_antennas if ant in problematic_ants]
        ax.scatter(problematic_indices, problematic_delays, c='red', s=100, zorder=5, label='Problematic (>100ns diff)')

    # --- Formatting ---
    title_str = "Delay Calibration Solutions"
    if problematic_ants:
        title_str += f"\n {len(problematic_ants)} Problematic Antennas Found"
    ax.set_title(title_str, fontsize=16)
    
    ax.set_xlabel("Antenna Index", fontsize=12)
    ax.set_ylabel("Delay (ns)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()

    # Save the figure
    try:
        fig.savefig(output_filepath)
        logger.info(f"Saved delay comparison plot to {os.path.basename(output_filepath)}")
        return True
    except Exception as e:
        logger.error(f"Failed to save delay plot: {e}")
        return False
    finally:
        plt.close(fig)

# ==============================================================================
# === Hardware Utilities
# ==============================================================================

def load_antenna_mapping():
    """
    Loads the antenna mapping dictionary directly from the pipeline configuration.
    """
    logger = get_logger('Utils.Mapping')
    
    # Check if the mapping dictionary exists in the config file
    if hasattr(config, 'ANTENNA_MAPPING') and isinstance(config.ANTENNA_MAPPING, dict) and config.ANTENNA_MAPPING:
        mapping_dict = config.ANTENNA_MAPPING
        logger.info(f"Loaded {len(mapping_dict)} antenna mappings directly from config.")
        return mapping_dict
    else:
        logger.error("ANTENNA_MAPPING dictionary not found or is empty in pipeline_config.py.")
        logger.warning("SNAP2 diagnostics will be unavailable.")
        return {}

# In pipeline_utils.py, replace the existing compare_and_diagnose_delays function.

def compare_and_diagnose_delays(context, cal_data, ref_table_path, qa_dir):
    """
    Compares delay solutions against a reference and runs SNAP2 diagnostics.
    Returns the list of problematic antennas and the loaded reference data.
    """
    logger = get_logger('Utils.QA.Diagnostics')
    problematic_ants = set()
    ref_cal_data = None
    
    if not os.path.exists(ref_table_path):
        logger.warning(f"Reference delay table not found: {ref_table_path}. Skipping comparison.")
    else:
        logger.info(f"Comparing delays against reference: {os.path.basename(ref_table_path)}")
        ref_cal_data = read_cal_table(ref_table_path)
        
        if ref_cal_data:
            current_delays_ns = {ant: data[0, 0] for ant, data, flag in zip(cal_data['antenna1_idx'], cal_data['data'], cal_data['flags']) if not flag[0, 0]}
            ref_delays_ns = {ant: data[0, 0] for ant, data, flag in zip(ref_cal_data['antenna1_idx'], ref_cal_data['data'], ref_cal_data['flags']) if not flag[0, 0]}

            for ant_idx, delay_ns in current_delays_ns.items():
                if ant_idx in ref_delays_ns:
                    diff_ns = abs(delay_ns - ref_delays_ns[ant_idx])
                    if diff_ns > config.DELAY_DIFF_WARN_NS:
                        logger.warning(f"Antenna {ant_idx}: Problematic delay difference of {diff_ns:.1f} ns detected.")
                        problematic_ants.add(ant_idx)
    
    logger.info("Running SNAP2 board diagnostics based on problematic delay values.")
    # (The rest of the SNAP2 diagnostic logic remains the same as the last version)
    antenna_mapping = context.get('antenna_mapping', {})
    if not antenna_mapping:
        logger.warning("Antenna mapping not loaded. Skipping SNAP2 diagnostics.")
        return problematic_ants, ref_cal_data

    snap2_stats = {}
    for ant_idx, snap2_loc in antenna_mapping.items():
        board = snap2_loc.split('C')[0]
        if board not in snap2_stats:
            snap2_stats[board] = {'total': 0, 'problematic': 0}
        
        snap2_stats[board]['total'] += 1
        if ant_idx in problematic_ants:
            snap2_stats[board]['problematic'] += 1
            
    critical_failure = False
    for board, stats in snap2_stats.items():
        if stats['total'] > 0:
            problem_fraction = stats['problematic'] / stats['total']
            logger.info(f"SNAP2 Board Group {board}: {stats['problematic']}/{stats['total']} antennas with problematic delays ({problem_fraction:.0%}).")
            
            if problem_fraction >= config.SNAP2_FAILURE_THRESHOLD:
                logger.error(f"CRITICAL: SNAP2 board group {board} has a problematic fraction ({problem_fraction:.0%}) exceeding the failure threshold!")
                critical_failure = True

    if critical_failure:
        raise RuntimeError("SNAP2 diagnostics detected a critical failure based on delay value differences.")
    else:
        logger.info("SNAP2 diagnostics completed successfully. No boards exceeded the failure threshold.")
        
    return problematic_ants, ref_cal_data

# ==============================================================================
# === Pipeline Context Functions ===
# ==============================================================================

# (get_observation_info_from_files, setup_pipeline_context_phase1, setup_pipeline_context_phase2 remain the same as the previously working version)
def get_observation_info_from_files(ms_files):
    """Analyzes MS filenames to determine observation parameters."""
    logger = logging.getLogger('OVRO_Pipeline.Utils')
    if not ms_files:
        return None
    timestamps = []
    ms_files.sort()
    logger.info(f"Analyzing {len(ms_files)} input files...")
    for ms_file in tqdm(ms_files, desc="Analyzing MS files", unit="file"):
        base_name = os.path.basename(ms_file)
        match = re.match(r'(\d{8})_(\d{6})_.*\.ms', base_name)
        if match:
            date_str, time_str = match.groups()
            try:
                dt_object = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                timestamps.append(dt_object)
            except ValueError:
                logger.warning(f"Skipping file with invalid timestamp format: {base_name}")
                continue
    if not timestamps:
        logger.error("No valid timestamps found in filenames.")
        return None

    min_midpoint_time = min(timestamps)
    max_midpoint_time = max(timestamps)
    half_int = timedelta(seconds=config.SINGLE_INTEGRATION_DURATION_SECONDS / 2)
    true_start_time = min_midpoint_time - half_int
    true_end_time = max_midpoint_time + half_int
    obs_date_str = true_start_time.strftime('%Y-%m-%d')
    mid_dt = true_start_time + (true_end_time - true_start_time) / 2
    try:
        mid_time_astropy = Time(mid_dt, format='datetime', scale='utc', location=config.OVRO_LWA_LOCATION)
        lst = mid_time_astropy.sidereal_time('mean')
        lst_hour_int = int(lst.hour) 
        lst_str = f"{lst_hour_int:02d}h"
        logger.info(f"Observation Midpoint: {mid_dt.isoformat()} UTC. Calculated LST: {lst.hour:.2f}h.")
    except Exception as e:
        logger.warning(f"LST calculation failed: {e}. Falling back to UTC hour.")
        lst_str = f"{mid_dt.hour:02d}h"
    duration_minutes = (true_end_time - true_start_time).total_seconds() / 60.0
    logger.info(f"Total observation duration: {duration_minutes:.2f} minutes.")
    return {
        'obs_date': obs_date_str,
        'lst_hour': lst_str,
        'start_time': true_start_time,
        'end_time': true_end_time,
        'mid_time_dt': mid_dt,
        'duration_minutes': duration_minutes,
        'ms_files': ms_files
    }

def setup_pipeline_context_phase1(input_data_folder, script_dir):
    """Phase 1: Sets up the temporary working directory."""
    if not input_data_folder or not os.path.isdir(input_data_folder):
        print(f"ERROR: Input data folder invalid or missing: {input_data_folder}")
        return None
    ms_files = [os.path.abspath(f) for f in glob.glob(os.path.join(input_data_folder, '*.ms')) if os.path.isdir(f)]
    if not ms_files:
        print(f"ERROR: No Measurement Set directories found in '{input_data_folder}'")
        return None

    processing_datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_base_dir = os.path.join(config.PARENT_OUTPUT_DIR, "temp_processing")
    working_dir = os.path.join(temp_base_dir, processing_datetime_str)

    output_ms_dir = os.path.join(working_dir, "ms")
    output_tables_dir = os.path.join(working_dir, "tables")
    output_qa_dir = os.path.join(working_dir, "QA")

    context = {
        'script_dir': script_dir,
        'input_folder': input_data_folder,
        'obs_info': None,
        'ms_files_list': ms_files,
        'working_dir': working_dir,
        'ms_dir': output_ms_dir,
        'tables_dir': output_tables_dir,
        'qa_dir': output_qa_dir,
        'successful_dir': None,
        'unsuccessful_dir': None,
        'concatenated_ms_path': None,
        'processing_datetime_str': processing_datetime_str,
        'model_sources': [] # NEW: To store sources included in the sky model
    }
    return context

def setup_pipeline_context_phase2(context, verbose=False):
    """Phase 2: Initializes logging, calculates ObsInfo/LST, and moves the working directory."""
    # 1. Initialize logging
    try:
        os.makedirs(context['working_dir'], exist_ok=True) 
        logger, log_filepath = setup_logging(context['working_dir'], verbose=verbose)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize logging system: {e}. Halting.")
        return None

    logger.info("--- Starting Context Finalization (Phase 2) ---")

    # 2. Calculate Obs Info
    obs_info = get_observation_info_from_files(context['ms_files_list'])
    if not obs_info:
        logger.error("Could not determine observation info from MS files. Halting.")
        return None
    
    context['obs_info'] = obs_info

    # 3. Define the final directory structure
    base_date_lst_dir = os.path.join(config.PARENT_OUTPUT_DIR, obs_info['obs_date'], obs_info['lst_hour'])
    final_working_dir = os.path.join(base_date_lst_dir, "working", context['processing_datetime_str'])
    
    context['successful_dir'] = os.path.join(base_date_lst_dir, "successful", context['processing_datetime_str'])
    context['unsuccessful_dir'] = os.path.join(base_date_lst_dir, "unsuccessful", context['processing_datetime_str'])

    # 4. Move the working directory
    temp_working_dir = context['working_dir']
    
    if os.path.abspath(temp_working_dir) != os.path.abspath(final_working_dir):
        logger.info(f"Moving working directory from {temp_working_dir} to {final_working_dir}")
        try:
            os.makedirs(os.path.dirname(final_working_dir), exist_ok=True)
            
            # Close log handler
            file_handlers = []
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    file_handlers.append(handler)
            
            shutil.move(temp_working_dir, final_working_dir)
            
            # Update context paths
            context['working_dir'] = final_working_dir
            context['ms_dir'] = os.path.join(final_working_dir, "ms")
            context['tables_dir'] = os.path.join(final_working_dir, "tables")
            context['qa_dir'] = os.path.join(final_working_dir, "QA")
            
            # Re-open log handler
            new_log_filepath = os.path.join(context['qa_dir'], os.path.basename(log_filepath))
            for handler in file_handlers:
                handler.baseFilename = new_log_filepath
                handler.stream = handler._open()

            logger.info(f"Working directory moved and log file updated.")
            
            # Cleanup temp directory
            try:
                os.rmdir(os.path.dirname(temp_working_dir))
            except OSError:
                pass

        except Exception as e:
            logger.exception(f"Failed to move working directory. Halting.")
            return None

    os.makedirs(context['ms_dir'], exist_ok=True)
    os.makedirs(context['tables_dir'], exist_ok=True)

    initialize_casalog()
    logger.info("--- Context Finalization Complete ---")
    return context

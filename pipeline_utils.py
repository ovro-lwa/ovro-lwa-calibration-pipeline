# pipeline_utils.py
import os
import sys
import logging
import subprocess
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shutil

# ==============================================================================
# === CASA Environment Initialization ===
# ==============================================================================
CASA_AVAILABLE = False
TABLE_TOOLS_AVAILABLE = False
CASA_IMPORTS = {}

try:
    # Attempt to import core CASA tools
    from casatools import table, msmetadata, componentlist, imfit, imstat, applycal, bandpass, gaincal
    
    CASA_IMPORTS['table_tools'] = table
    CASA_IMPORTS['msmetadata'] = msmetadata
    CASA_IMPORTS['componentlist'] = componentlist
    CASA_IMPORTS['imfit'] = imfit
    CASA_IMPORTS['imstat'] = imstat
    CASA_IMPORTS['applycal'] = applycal
    CASA_IMPORTS['bandpass'] = bandpass
    CASA_IMPORTS['gaincal'] = gaincal

    CASA_AVAILABLE = True
    TABLE_TOOLS_AVAILABLE = True
    
    # Suppress verbose CASA logging to the console unless it's an error
    from casatasks.private import casa_transition as ct
    ct.logger_wrapper(origin="casatools").setLevel('ERROR')

except ImportError:
    print("WARNING: 'casatools' not found. CASA-dependent operations will fail.", file=sys.stderr)
except Exception as e:
    print(f"WARNING: An unexpected error occurred during CASA import: {e}", file=sys.stderr)

# ==============================================================================
# === Logging Setup ===
# ==============================================================================
def get_logger(name, log_filepath=None, level=logging.DEBUG):
    """Initializes and returns a logger instance."""
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if logger is already configured
    if not logger.handlers:
        logger.setLevel(level)
        
        # Console Handler (prints INFO and above)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)
        
        # File Handler (prints all DEBUG messages)
        if log_filepath:
            fh = logging.FileHandler(log_filepath, mode='a')
            fh.setLevel(logging.DEBUG)
            fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(fh_formatter)
            logger.addHandler(fh)
            
    return logger

# Initialize a default logger instance for utility functions
logger = get_logger('PipelineUtils')

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
    
    # For security and clarity, ensure command is a list of strings
    if isinstance(cmd, str):
        local_logger.error("Command must be a list of arguments, not a single string.")
        raise ValueError("Command must be a list of arguments.")
        
    local_logger.debug(f"Executing Command: {' '.join(cmd)}")

    # Combine current environment with any custom variables
    process_env = os.environ.copy()
    if env_vars:
        process_env.update(env_vars)

    try:
        # Use Popen to stream output in real-time
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              bufsize=1, universal_newlines=True, env=process_env) as p:
            
            # Use tqdm for a simple progress bar based on lines of output
            with tqdm(p.stdout, desc=f"Running {cmd[0].split('/')[-1]}", unit="line", 
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
# === Measurement Set Utilities ===
# ==============================================================================
def get_num_integrations(ms_path):
    """
    Determines the number of unique integrations (time steps) in a Measurement Set.
    """
    if not TABLE_TOOLS_AVAILABLE:
        logger.warning("CASA table tools not available. Cannot determine the number of integrations.")
        return None
    
    pt = CASA_IMPORTS.get('table_tools')
    if not pt:
        logger.warning("Could not get 'table_tools' from CASA imports.")
        return None
        
    try:
        with pt.table(ms_path, readonly=True, ack=False) as t:
            unique_times = np.unique(t.getcol('TIME'))
            num_integrations = len(unique_times)
            logger.info(f"Found {num_integrations} unique integrations in {os.path.basename(ms_path)}.")
            return num_integrations
    except Exception as e:
        logger.error(f"Failed to read number of integrations from {ms_path}: {e}")
        return None

# ==============================================================================
# === Calibration & QA Utilities ===
# ==============================================================================
def read_cal_table(table_path):
    """Reads a CASA calibration table and returns a dictionary of its contents."""
    if not TABLE_TOOLS_AVAILABLE:
        logger.error("CASA table tools not available. Cannot read calibration table.")
        return None
        
    pt = CASA_IMPORTS.get('table_tools')
    try:
        with pt.table(table_path, readonly=True, ack=False) as t:
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

def compare_and_diagnose_delays(context, cal_data, ref_table_path, qa_dir):
    """Compares a delay table to a reference and identifies problematic antennas."""
    problematic_ants = []
    ref_cal_data = None
    
    if os.path.exists(ref_table_path):
        ref_cal_data = read_cal_table(ref_table_path)
        if ref_cal_data:
            # Simple comparison logic (can be expanded)
            # Match antennas between current and reference calibration
            ant_map = {ant: i for i, ant in enumerate(cal_data['ant1'])}
            ref_ant_map = {ant: i for i, ant in enumerate(ref_cal_data['ant1'])}
            
            common_ants = sorted(list(set(ant_map.keys()) & set(ref_ant_map.keys())))
            
            diffs = []
            for ant in common_ants:
                idx = ant_map[ant]
                ref_idx = ref_ant_map[ant]
                # Compare the real part of the gain (delay)
                diff = np.abs(cal_data['gain'][0, 0, idx].real - ref_cal_data['gain'][0, 0, ref_idx].real)
                diffs.append(diff)

            if diffs:
                median_diff = np.median(diffs)
                std_diff = np.std(diffs)
                threshold = median_diff + 3 * std_diff # 3-sigma outlier
                
                for i, ant in enumerate(common_ants):
                    if diffs[i] > threshold:
                        problematic_ants.append(ant)
                
                logger.info(f"Identified {len(problematic_ants)} problematic antennas by comparing to reference: {problematic_ants}")

    return problematic_ants, ref_cal_data

def plot_delays(cal_data, plot_path, ref_cal_data=None, problematic_ants=None):
    """Generates and saves a plot of delays vs. antenna number."""
    if problematic_ants is None:
        problematic_ants = []
        
    try:
        antennas = cal_data['ant1']
        delays = cal_data['gain'][0, 0, :].real # Delays are in the real part for 'K' type
        
        plt.figure(figsize=(14, 7))
        plt.plot(antennas, delays, 'o', label='Measured Delays', zorder=5)

        if ref_cal_data:
            ref_antennas = ref_cal_data['ant1']
            ref_delays = ref_cal_data['gain'][0, 0, :].real
            plt.plot(ref_antennas, ref_delays, 'x', color='gray', label='Reference Delays', alpha=0.6, zorder=1)

        if problematic_ants:
            p_indices = [list(antennas).index(ant) for ant in problematic_ants]
            plt.plot(antennas[p_indices], delays[p_indices], 'o', color='red', markersize=10, 
                     label='Problematic Antennas', zorder=10)

        plt.xlabel("Antenna Number")
        plt.ylabel("Delay (nanoseconds)")
        plt.title("Delay Calibration Solutions")
        plt.grid(True)
        plt.legend()
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved delay plot to {os.path.basename(plot_path)}")
        return True
    except Exception as e:
        logger.error(f"Failed to generate delay plot: {e}")
        return False

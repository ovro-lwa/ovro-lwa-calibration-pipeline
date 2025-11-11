#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
process_archive.py
Orchestrates the 4-stage (Scatter-Gather-Aggregate) processing
of OVRO-LWA archive data.

--mode subband_work: (Array Task) Runs on a worker node.
                     Submits local Stage 2 (Peel) and Stage 3 (Gather).
--mode stage_2_task: (Local Array Task) Peels a single MS file.
--mode stage_3_gather: (Local Job) Concatenates, flags, and writes receipt.
--mode final_image:  (Central Job) Aggregates all MS files via scp
                     and runs final UV-Join imaging.
"""

import os
import sys
import argparse
import logging
import shutil
import re
import socket # For getting hostname
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import glob
import multiprocessing
import subprocess # For submitting slurm jobs

# --- Add script directory to path ---
# SCRIPT_DIR is now passed as an argument
# This must be done *before* other imports
cli_args_for_path = sys.argv
script_dir_from_arg = None
if '--script_dir' in cli_args_for_path:
    try:
        script_dir_from_arg = cli_args_for_path[cli_args_for_path.index('--script_dir') + 1]
        if script_dir_from_arg not in sys.path:
            sys.path.insert(0, script_dir_from_arg)
    except Exception as e:
        print(f"Warning: Could not parse --script_dir early: {e}", file=sys.stderr)

try:
    import pipeline_utils 
    import pipeline_config as config
    from data_preparation import fix_ms_field_id, identify_bad_antennas_mnc, apply_flags_with_casa
    from astropy.time import Time
    import astropy.units as u
    from astropy.coordinates import SkyCoord, AltAz
    from astropy.io import fits
except ImportError as e:
    print(f"FATAL: Failed to import required modules: {e}", file=sys.stderr)
    print(f"Searched sys.path (including): {script_dir_from_arg}")
    sys.exit(1)

# --- Globals ---
ARCHIVE_BASE_DIR = '/lustre/pipeline/night-time/averaged/'
ALL_POSSIBLE_SUB_BANDS = [
    '13MHz', '18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz',
    '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz'
]
BANDS_TO_IMAGE = [b for b in ALL_POSSIBLE_SUB_BANDS if b not in ('13MHz', '82MHz')]
INTEGRATION_DURATION_SEC = 10.031
PEELING_CONDA_ENV = 'julia060'
PEELING_RFI_CONDA_ENV = 'ttcal_dev' 
EXIT_CODE_SKIP = 99
LOCAL_TASK_MAP_FILE = 'peeling_task_map.txt'

# Use 32 processes to match the --cpus-per-task=32 request
POOL_PROCESSES = 32

# NEW: Define parallel peel jobs and memory
# 48G was still not enough (Code -9 OOM). Trying 96G.
# 384G / 4 jobs = 96G per job.
PEEL_JOB_SLOTS = 4
PEEL_JOB_MEM_G = 96


# --- Custom Argparse Type for Datetime (MODIFIED) ---
def valid_datetime(s):
    """Custom argparse type for validating datetime strings."""
    import re # Ensure re is available
    format_str = '%Y-%m-%d:%H:%M:%S'
    
    # NEW: Use regex to find the *first* match for the datetime pattern.
    # This will ignore any junk characters, quotes, or whitespace.
    match = re.search(r'(\d{4}-\d{2}-\d{2}:\d{2}:\d{2}:\d{2})', s)
    
    if match:
        cleaned_s = match.group(1)
        try:
            return datetime.strptime(cleaned_s, format_str)
        except ValueError:
            # This should almost never happen if the regex matched
            msg = f"Date string '{cleaned_s}' matched regex but failed parsing."
            raise argparse.ArgumentTypeError(msg)
    else:
        msg = f"Not a valid datetime: '{s}'. Could not find pattern '{format_str}'"
        raise argparse.ArgumentTypeError(msg)

# ==============================================================================
# === Logging Utility (Unchanged) ===
# ==============================================================================
def setup_task_logger(log_name, log_filepath):
    """Configures and returns a logger instance."""
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    fh = logging.FileHandler(log_filepath, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
         logger.addHandler(ch)
    return logger

# ==============================================================================
# === Core Processing Functions (Unchanged logic, MODIFIED calls) ===
# ==============================================================================
# ... (find_archive_files_for_subband, copy_and_prepare_files, concatenate_sub_band, run_fix_field_id unchanged) ...
def find_archive_files_for_subband(start_dt, end_dt, sub_band, logger, input_dir=None):
    """Finds archive MS files for a sub-band within the time range."""
    file_list = []
    filename_pattern = re.compile(r'(\d{8})_(\d{6})_' + re.escape(sub_band) + r'(?:|_averaged)\.ms')
    if input_dir:
        logger.info(f"Searching custom input directory for sub-band {sub_band}: {input_dir}")
        search_pattern = os.path.join(input_dir, f'*{sub_band}*.ms')
        found_files = glob.glob(search_pattern)
        logger.debug(f"Glob pattern '{search_pattern}' found {len(found_files)} potential files.")
        for f_path in found_files:
            filename = os.path.basename(f_path)
            match = filename_pattern.search(filename)
            if match:
                date_str_file, time_str_file = match.groups()
                try:
                    file_midpoint_dt = datetime.strptime(date_str_file + time_str_file, '%Y%m%d%H%M%S')
                    if start_dt <= file_midpoint_dt < end_dt: file_list.append(f_path)
                    else: logger.debug(f"Skipping {filename}: Midpoint {file_midpoint_dt} outside range {start_dt} - {end_dt}")
                except ValueError: logger.warning(f"Could not parse timestamp from filename in input_dir: {filename}")
            elif filename.endswith(".ms"): logger.warning(f"Filename looks like MS but does not match pattern: {filename}")
    else: # Standard Archive
        logger.info(f"Searching standard archive structure for sub-band: {sub_band}")
        logger.info(f"Time Range: {start_dt.isoformat()} to {end_dt.isoformat()} UTC")
        current_hour_dt = start_dt.replace(minute=0, second=0, microsecond=0)
        loop_end_dt = end_dt
        if end_dt.minute == 0 and end_dt.second == 0 and end_dt.microsecond == 0: loop_end_dt -= timedelta(microseconds=1)
        loop_end_hour_dt = loop_end_dt.replace(minute=0, second=0, microsecond=0)
        while current_hour_dt <= loop_end_hour_dt:
            date_str, hour_str = current_hour_dt.strftime('%Y-%m-%d'), current_hour_dt.strftime('%H')
            target_dir = os.path.join(ARCHIVE_BASE_DIR, sub_band, date_str, hour_str)
            if os.path.isdir(target_dir):
                logger.debug(f"Scanning directory: {target_dir}")
                try:
                    for f in os.listdir(target_dir):
                        match = filename_pattern.search(f)
                        if match:
                            date_str_file, time_str_file = match.groups()
                            try:
                                file_midpoint_dt = datetime.strptime(date_str_file + time_str_file,'%Y%m%d%H%M%S')
                                if start_dt <= file_midpoint_dt < end_dt: file_list.append(os.path.join(target_dir, f))
                            except ValueError: logger.warning(f"Could not parse timestamp from file: {f}")
                except OSError as e: logger.warning(f"Could not list directory {target_dir}: {e}")
            else: logger.debug(f"Skipping non-existent directory: {target_dir}")
            current_hour_dt += timedelta(hours=1)
    logger.info(f"Search complete for {sub_band}. Found {len(file_list)} files within time range.")
    return file_list

def copy_and_prepare_files(archive_file_list, working_dir, sub_band, logger):
    """Copies MS files from source location to temporary (NVMe) working directory."""
    logger.info(f"--- Step 2: Copying {len(archive_file_list)} files for {sub_band} to NVMe ---")
    logger.info(f"Destination base: {working_dir}")
    ms_raw_copy_dir = os.path.join(working_dir, 'ms_data', f"{sub_band}_raw_copies")
    os.makedirs(ms_raw_copy_dir, exist_ok=True)
    copied_file_list = []
    for src_path in tqdm(archive_file_list, desc=f'Copying {sub_band}', unit='file', disable=None):
        dest_path = os.path.join(ms_raw_copy_dir, os.path.basename(src_path))
        if os.path.exists(dest_path): logger.warning(f"Destination exists, removing: {dest_path}"); shutil.rmtree(dest_path)
        try: shutil.copytree(src_path, dest_path, symlinks=False); copied_file_list.append(dest_path)
        except Exception as e: logger.error(f"Failed to copy {src_path} to {dest_path}: {e}"); raise RuntimeError(f"Failed to copy archive file: {src_path}")
    logger.info(f"Finished copying {len(copied_file_list)} files for {sub_band} to {ms_raw_copy_dir}.")
    return copied_file_list

def concatenate_sub_band(processed_file_list, working_dir, sub_band, logger, is_peeling_mode=False):
    """Concatenates processed files (on NVMe) into a single MS (on NVMe)."""
    step_num = "5" if is_peeling_mode else "3"; logger.info(f"--- Step {step_num}: Concatenating {sub_band} on NVMe ---")
    if not processed_file_list: logger.warning(f"No files provided for {sub_band}, skipping."); return None
    concat = pipeline_utils.get_casa_task('concat');
    if not concat: logger.error("CASA task 'concat' not found."); return None
    valid_files = [f for f in processed_file_list if os.path.isdir(f)]
    if len(valid_files) != len(processed_file_list): logger.warning(f"Found {len(processed_file_list) - len(valid_files)} missing MS paths before concat for {sub_band}.")
    if not valid_files: logger.error(f"No valid MS files found to concatenate for {sub_band}."); return None
    sorted_file_list = sorted(valid_files)
    ms_data_dir = os.path.join(working_dir, 'ms_data')
    concat_ms_path = os.path.join(ms_data_dir, f'{sub_band}_concat.ms')
    if os.path.exists(concat_ms_path): logger.warning(f"Removing existing concat file: {concat_ms_path}"); shutil.rmtree(concat_ms_path)
    try:
        logger.info(f"Concatenating {len(sorted_file_list)} files into {os.path.basename(concat_ms_path)}...")
        concat(vis=sorted_file_list, concatvis=concat_ms_path, dirtol='', copypointing=False)
        logger.info(f"Cleaning up {len(sorted_file_list)} individual source files/dirs...")
        parent_dir = None
        for f in sorted_file_list:
            if os.path.exists(f): parent_dir = os.path.dirname(f); shutil.rmtree(f)
        if parent_dir and os.path.exists(parent_dir):
            try: os.rmdir(parent_dir); logger.info(f"Removed empty source directory: {parent_dir}")
            except OSError: logger.warning(f"Could not remove source directory (not empty?): {parent_dir}")
    except Exception as e:
        logger.error(f"Failed to concatenate {sub_band}: {e}", exc_info=True)
        if os.path.exists(concat_ms_path):
            logger.warning(f"Attempting to remove potentially corrupted concat MS: {concat_ms_path}")
            try: shutil.rmtree(concat_ms_path)
            except Exception as cleanup_e: logger.error(f"Failed cleanup: {cleanup_e}")
        return None
    logger.info(f"Concatenation complete for {sub_band}.")
    return concat_ms_path

def run_fix_field_id(ms_path, logger):
    """Runs fix_ms_field_id on the specified MS."""
    logger.debug(f"Fixing FIELD_ID for: {os.path.basename(ms_path)}")
    try: fix_ms_field_id(ms_path); logger.debug(f"FIELD_ID fixed."); return True
    except ImportError: logger.error("MS table library missing."); return False
    except Exception as e: logger.error(f"Failed FIELD_ID fixing: {e}", exc_info=True); return False

def apply_bandpass_calibration(ms_path, bp_table_path, sub_band, caltable_spw_map, logger):
    """Applies bandpass calibration using dynamic SPW map."""
    logger.debug(f"Applying Bandpass Cal to: {os.path.basename(ms_path)}")
    applycal = pipeline_utils.get_casa_task('applycal');
    if not applycal: logger.error("CASA task 'applycal' not found."); return False
    spw_id = caltable_spw_map.get(sub_band)
    if spw_id is None: logger.warning(f"{sub_band} not in cal table map. Skipping applycal."); return True
    logger.debug(f"Mapping {sub_band} MS to Cal Table SPW ID {spw_id}")
    try: applycal(vis=ms_path, gaintable=[bp_table_path], spwmap=[spw_id], interp='nearest', calwt=False, flagbackup=True)
    except Exception as e:
        if "specify spw indices <= maximum available" in str(e):
             max_spw = -1; match = re.search(r"\((\d+)\)", str(e));
             if match: max_spw = int(match.group(1))
             logger.error(f"Applycal failed: Requested SPW ID {spw_id} out of bounds for {bp_table_path} (max SPW is {max_spw}). Map: {caltable_spw_map}")
        else: logger.error(f"Failed applycal: {e}", exc_info=True)
        return False
    logger.debug(f"Applycal successful.")
    return True

def set_common_phase_center(ms_path, center_coord, logger):
    """Changes the phase center of the specified MS file."""
    logger.debug(f"Setting Phase Center for: {os.path.basename(ms_path)}")
    ra_str = center_coord.ra.to_string(unit=u.hourangle, sep='hms', precision=3)
    dec_str = center_coord.dec.to_string(unit=u.deg, sep='dms', precision=3)
    logger.debug(f"Target Center: {center_coord.to_string('hmsdms')}")
    command = [config.CHGCENTRE_PATH, ms_path, ra_str, dec_str]
    success = pipeline_utils.run_command(command, task_name=f"chgcentre ({os.path.basename(ms_path)})", logger=logger)
    if not success: logger.error(f"chgcentre failed."); return False
    logger.debug(f"Phase center set.")
    return True

# --- Antenna Flagging Function (Unchanged) ---
def run_antenna_flagging(ms_path, context, script_dir, logger):
    """
    Identifies and flags bad antennas using MNC and config lists.
    """
    logger.info(f"Running Antenna Flagging for: {os.path.basename(ms_path)}")
    try:
        mnc_context = {
            'script_dir': script_dir, # Use passed-in script_dir
            'obs_info': {'obs_mid_time': context['midpoint_time']}
        }
        mnc_bad_antennas = identify_bad_antennas_mnc(mnc_context)
        antennas_to_flag = set()

        if mnc_bad_antennas is None:
            logger.warning("MNC identification failed/skipped. Falling back to ADDITIONAL_BAD_ANTENNAS.")
            antennas_to_flag.update(config.ADDITIONAL_BAD_ANTENNAS)
        elif len(mnc_bad_antennas) == 0:
            logger.info("MNC returned 0 bad antennas. Applying FALLBACK_BAD_ANTENNAS.")
            antennas_to_flag.update(config.FALLBACK_BAD_ANTENNAS)
        else:
            logger.info("Combining MNC results with ADDITIONAL_BAD_ANTENNAS.")
            antennas_to_flag.update(mnc_bad_antennas)
            antennas_to_flag.update(config.ADDITIONAL_BAD_ANTENNAS)

        if antennas_to_flag:
            logger.info(f"Applying flags to {len(antennas_to_flag)} antennas.")
            logger.debug(f"Antennas to flag: {sorted(list(antennas_to_flag))}")
            apply_flags_with_casa(ms_path, sorted(list(antennas_to_flag)))
        else:
            logger.info("No antennas met criteria for health flagging.")
        return True
    except Exception as e:
        logger.error(f"Error during antenna flagging: {e}", exc_info=True)
        return False
    
def run_aoflagger_on_sub_band(concat_ms_path, sub_band, logger):
    """Runs AOFlagger on the concatenated sub-band MS."""
    logger.info(f"--- Step 6: Running AOFlagger on {sub_band} ---"); logger.info(f"Operating on: {os.path.basename(concat_ms_path)}")
    if not config.AOFLAGGER_EXECUTABLE or not os.path.exists(config.AOFLAGGER_EXECUTABLE): logger.error(f"AOFlagger not found: {config.AOFLAGGER_EXECUTABLE}"); return False
    if not config.AOFLAGGER_STRATEGY_PATH or not os.path.exists(config.AOFLAGGER_STRATEGY_PATH): logger.error(f"AOFlagger strategy not found: {config.AOFLAGGER_STRATEGY_PATH}"); return False
    aoflagger_cmd = [config.AOFLAGGER_EXECUTABLE, '-strategy', config.AOFLAGGER_STRATEGY_PATH, concat_ms_path]
    success = pipeline_utils.run_command(aoflagger_cmd, task_name=f"AOFlagger ({os.path.basename(concat_ms_path)})", logger=logger)
    if not success: logger.error(f"AOFlagger failed."); return False
    logger.info(f"AOFlagger finished."); return True

# --- run_peeling UPDATED (uses shell=True) ---
def run_peeling(ms_path, model_path, logger):
    """
    Runs the ttcal.jl astrophysical peeling tool on a single MS file.
    Uses shell=True to properly activate conda env and avoid nested conda run.
    """
    logger.info(f"Starting Astrophysical Peeling for: {os.path.basename(ms_path)}")
    if not os.path.exists(model_path):
        logger.error(f"Astro peel model not found: {model_path}")
        return False
        
    task_name = f"ttcal.jl zest astro ({os.path.basename(ms_path)})"
    cmd_string = (
        f"source ~/.bashrc && "
        f"conda activate {PEELING_CONDA_ENV} && "
        f"ttcal.jl zest {ms_path} {model_path} "
        f"--beam constant --minuvw 10 --maxiter 30 --tolerance 1e-4"
    )
    
    logger.debug(f"Command: {cmd_string}")
    
    try:
        process = subprocess.Popen(
            cmd_string, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8', 
            errors='replace', 
            shell=True,  # <-- This is the key
            executable='/bin/bash' # <-- Ensures we use bash
        )
        
        # Stream the output to the DEBUG log
        for line in iter(process.stdout.readline, ''): 
            logger.debug(f'[{task_name}] {line.strip()}')
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code:
            # We already logged the error, just raise it
            logger.error(f"Task '{task_name}' failed (Code {return_code}).")
            return False
            
        logger.info(f"Finished Astrophysical Peeling for: {os.path.basename(ms_path)}."); 
        return True
        
    except Exception as e:
        logger.error(f"Failed to execute peeling: {e}", exc_info=True)
        return False

# --- run_peeling_rfi UPDATED (uses shell=True) ---
def run_peeling_rfi(ms_path, model_path, logger):
    """
    Runs the ttcal.jl RFI peeling tool on a single MS file.
    Uses shell=True to properly activate conda env and avoid nested conda run.
    """
    logger.info(f"Starting RFI Peeling for: {os.path.basename(ms_path)}")
    if not os.path.exists(model_path):
        logger.error(f"RFI peel model not found: {model_path}")
        return False

    task_name = f"ttcal.jl zest RFI ({os.path.basename(ms_path)})"
    cmd_string = (
        f"source ~/.bashrc && "
        f"conda activate {PEELING_RFI_CONDA_ENV} && "
        f"ttcal.jl zest {ms_path} {model_path} "
        f"--beam constant --minuvw 10 --maxiter 30 --tolerance 1e-4"
    )

    logger.debug(f"Command: {cmd_string}")

    try:
        process = subprocess.Popen(
            cmd_string, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8', 
            errors='replace', 
            shell=True,  # <-- This is the key
            executable='/bin/bash' # <-- Ensures we use bash
        )
        
        # Stream the output to the DEBUG log
        for line in iter(process.stdout.readline, ''): 
            logger.debug(f'[{task_name}] {line.strip()}')
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code:
            # We already logged the error, just raise it
            logger.error(f"Task '{task_name}' failed (Code {return_code}).")
            return False
            
        logger.info(f"Finished RFI Peeling for: {os.path.basename(ms_path)}."); 
        return True
        
    except Exception as e:
        logger.error(f"Failed to execute peeling: {e}", exc_info=True)
        return False

# ==============================================================================
# === IMAGING FUNCTIONS (MODIFIED to pass logger) ===
# ==============================================================================
# ... (run_subband_imaging, run_final_imaging, run_coadd_imaging unchanged) ...
def run_subband_imaging(ms_path, subband_name, qa_dir, logger, intervals_out=None, num_integrations=0):
    """
    Runs WSClean on a *single* sub-band MS. (Multi-node path)
    Handles time interval splitting using -intervals-out.
    """
    logger.info(f"--- Step 8 (Sub-band Imaging): Running WSClean for {subband_name} ---")
    if not config.WSCLEAN_PATH or not os.path.exists(config.WSCLEAN_PATH): 
        logger.error(f"WSClean not found: {config.WSCLEAN_PATH}"); return False
        
    original_threads = os.environ.get("OPENBLAS_NUM_THREADS"); os.environ["OPENBLAS_NUM_THREADS"] = "1"; logger.debug("Set OPENBLAS_NUM_THREADS=1.")
    
    interval_args = []
    if intervals_out is None:
        logger.info("Imaging mode: Single interval (default)")
    elif intervals_out == -1: 
        if num_integrations > 0:
            logger.info(f"Imaging mode: Per-integration ({num_integrations} intervals)")
            interval_args = ['-intervals-out', str(num_integrations)]
        else:
            logger.warning("Per-integration imaging requested, but num_integrations is 0. Defaulting to single interval.")
    else: 
        try:
            n_intervals = int(intervals_out)
            if n_intervals > 0:
                logger.info(f"Imaging mode: Splitting into {n_intervals} intervals")
                interval_args = ['-intervals-out', str(n_intervals)]
            else:
                 logger.warning(f"Invalid intervals-out value '{intervals_out}'. Defaulting to single interval.")
        except ValueError:
             logger.warning(f"Could not parse intervals-out value '{intervals_out}'. Defaulting to single interval.")

    wsclean_success = True
    try:
        # 1. Stokes I (Full, local-rms)
        logger.info(f"Starting Stokes I (Full) for {subband_name}...")
        name_i_full = os.path.join(qa_dir, f"{subband_name}-I-full")
        cmd_i_full = [
            config.WSCLEAN_PATH,
            '-pol', 'I', '-multiscale', '-multiscale-scale-bias', '0.9',
            '-niter', '50000', '-mgain', '0.85', '-horizon-mask', '10deg',
            '-mem', '50', '-local-rms', '-no-update-model-required',
            '-data-column', 'CORRECTED_DATA', '-size', '4096', '4096',
            '-scale', '0.03125', '-weight', 'briggs', '0',
            '-name', name_i_full
        ]
        cmd_i_full.extend(interval_args); cmd_i_full.append(ms_path)
        if not pipeline_utils.run_command(cmd_i_full, task_name=f"WSClean (I-Full {subband_name})", logger=logger): 
            logger.error(f"WSClean (I-Full {subband_name}) failed."); wsclean_success = False

        # 2. Stokes I (Tapered, auto-mask)
        if wsclean_success: 
            logger.info(f"Starting Stokes I (Tapered) for {subband_name}...")
            name_i_taper = os.path.join(qa_dir, f"{subband_name}-I-tapered")
            cmd_i_taper = [
                config.WSCLEAN_PATH,
                '-pol', 'I', '-multiscale', '-multiscale-scale-bias', '0.9',
                '-niter', '10000', '-taper-inner-tukey', '30',
                '-auto-threshold', '0.5', '-auto-mask', '3',
                '-mgain', '0.85', '-horizon-mask', '10deg', '-mem', '50',
                '-no-update-model-required', '-data-column', 'CORRECTED_DATA',
                '-size', '4096', '4096', '-scale', '0.03125', '-weight', 'briggs', '0',
                '-name', name_i_taper
            ]
            cmd_i_taper.extend(interval_args); cmd_i_taper.append(ms_path)
            if not pipeline_utils.run_command(cmd_i_taper, task_name=f"WSClean (I-Taper {subband_name})", logger=logger): 
                logger.error(f"WSClean (I-Taper {subband_name}) failed."); wsclean_success = False
        
        # 3. Stokes V (Tapered, auto-mask)
        if wsclean_success: 
            logger.info(f"Starting Stokes V (Tapered) for {subband_name}...")
            name_v_taper = os.path.join(qa_dir, f"{subband_name}-V-tapered")
            cmd_v_taper = [
                config.WSCLEAN_PATH,
                '-pol', 'V', '-multiscale', '-multiscale-scale-bias', '0.9',
                '-niter', '50000', '-taper-inner-tukey', '30',
                '-auto-threshold', '0.5', '-auto-mask', '3',
                '-mgain', '0.85', '-horizon-mask', '10deg', '-mem', '50',
                '-no-update-model-required', '-fit-spectral-pol', '3',
                '-data-column', 'CORRECTED_DATA', '-size', '4096', '4096',
                '-scale', '0.03125', '-weight', 'briggs', '0',
                '-name', name_v_taper
            ]
            cmd_v_taper.extend(interval_args); cmd_v_taper.append(ms_path)
            if not pipeline_utils.run_command(cmd_v_taper, task_name=f"WSClean (V-Taper {subband_name})", logger=logger): 
                logger.error(f"WSClean (V-Taper {subband_name}) failed."); wsclean_success = False

    finally:
        if original_threads is None:
            if "OPENBLAS_NUM_THREADS" in os.environ: del os.environ["OPENBLAS_NUM_THREADS"]
        else: os.environ["OPENBLAS_NUM_THREADS"] = original_threads
        logger.debug("Restored OPENBLAS_NUM_THREADS.")
    if wsclean_success: logger.info(f"WSClean sub-band imaging complete for {subband_name}.")
    else: logger.error(f"One or more WSClean steps failed for {subband_name}.")
    return wsclean_success

def run_final_imaging(ms_list, image_name_prefix, logger, intervals_out=None, num_integrations=0):
    """Runs WSClean with -join-channels on a list of MS files. (Single-node path)"""
    logger.info(f"--- Step 9 (UV-Join Imaging): Running WSClean Full-Sky Imaging ---")
    if not config.WSCLEAN_PATH or not os.path.exists(config.WSCLEAN_PATH): 
        logger.error(f"WSClean not found: {config.WSCLEAN_PATH}"); return False
    original_threads = os.environ.get("OPENBLAS_NUM_THREADS"); os.environ["OPENBLAS_NUM_THREADS"] = "1"; logger.debug("Set OPENBLAS_NUM_THREADS=1.")
    
    interval_args = []
    if intervals_out is None:
        logger.info("Imaging mode: Single interval (default)")
    elif intervals_out == -1: 
        if num_integrations > 0:
            logger.info(f"Imaging mode: Per-integration ({num_integrations} intervals)")
            interval_args = ['-intervals-out', str(num_integrations)]
        else:
            logger.warning("Per-integration imaging requested, but num_integrations is 0. Defaulting to single interval.")
    else: 
        try:
            n_intervals = int(intervals_out)
            if n_intervals > 0:
                logger.info(f"Imaging mode: Splitting into {n_intervals} intervals")
                interval_args = ['-intervals-out', str(n_intervals)]
            else:
                 logger.warning(f"Invalid intervals-out value '{intervals_out}'. Defaulting to single interval.")
        except ValueError:
             logger.warning(f"Could not parse intervals-out value '{intervals_out}'. Defaulting to single interval.")
    
    wsclean_success = True
    try:
        # 1. Stokes I (Full, local-rms)
        logger.info("Starting Stokes I (Full, UV-Join)..."); name_i_full = f"{image_name_prefix}-I-full"
        cmd_i_full = [
            config.WSCLEAN_PATH,
            '-pol', 'I', '-multiscale', '-multiscale-scale-bias', '0.9', 
            '-niter', '250000', '-mgain', '0.85', '-horizon-mask', '10deg',
            '-mem', '50', '-local-rms', '-no-update-model-required',
            '-fit-spectral-pol', '3', '-data-column', 'CORRECTED_DATA',
            '-size', '4096', '4096', '-scale', '0.03125', '-weight', 'briggs', '0',
            '-join-channels', '-channels-out', '3', '-name', name_i_full
        ]; 
        cmd_i_full.extend(interval_args); cmd_i_full.extend(ms_list)
        if not pipeline_utils.run_command(cmd_i_full, task_name="WSClean (I-Full, UV-Join)", logger=logger): 
            logger.error("WSClean (I-Full, UV-Join) failed."); wsclean_success = False
        
        # 2. Stokes I (Tapered, auto-mask)
        if wsclean_success: 
            logger.info("Starting Stokes I (Tapered, UV-Join)..."); name_i_taper = f"{image_name_prefix}-I-tapered"
            cmd_i_taper = [
                config.WSCLEAN_PATH,
                '-pol', 'I', '-multiscale', '-multiscale-scale-bias', '0.9', 
                '-niter', '250000', '-taper-inner-tukey', '30',
                '-auto-threshold', '0.5', '-auto-mask', '3',
                '-mgain', '0.85', '-horizon-mask', '10deg', '-mem', '50',
                '-no-update-model-required', '-fit-spectral-pol', '3',
                '-data-column', 'CORRECTED_DATA', '-size', '4096', '4096',
                '-scale', '0.03125', '-weight', 'briggs', '0',
                '-join-channels', '-channels-out', '3', '-name', name_i_taper
            ]; 
            cmd_i_taper.extend(interval_args); cmd_i_taper.extend(ms_list)
            if not pipeline_utils.run_command(cmd_i_taper, task_name="WSClean (I-Taper, UV-Join)", logger=logger): 
                logger.error("WSClean (I-Taper, UV-Join) failed."); wsclean_success = False

        # 3. Stokes V (Tapered, auto-mask)
        if wsclean_success: 
            logger.info("Starting Stokes V (Tapered, UV-Join)..."); name_v_taper = f"{image_name_prefix}-V-tapered"
            cmd_v_taper = [
                config.WSCLEAN_PATH,
                '-pol', 'V', '-multiscale', '-multiscale-scale-bias', '0.9', 
                '-niter', '50000', '-taper-inner-tukey', '30',
                '-auto-threshold', '0.5', '-auto-mask', '3',
                '-mgain', '0.85', '-horizon-mask', '10deg', '-mem', '50',
                '-no-update-model-required', '-fit-spectral-pol', '3',
                '-data-column', 'CORRECTED_DATA', '-size', '4096', '4096',
                '-scale', '0.03125', '-weight', 'briggs', '0',
                '-join-channels', '-channels-out', '3', '-name', name_v_taper
            ]; 
            cmd_v_taper.extend(interval_args); cmd_v_taper.extend(ms_list)
            if not pipeline_utils.run_command(cmd_v_taper, task_name="WSClean (V-Taper, UV-Join)", logger=logger): 
                logger.error("WSClean (V-Taper, UV-Join) failed."); wsclean_success = False
    finally:
        if original_threads is None:
            if "OPENBLAS_NUM_THREADS" in os.environ: del os.environ["OPENBLAS_NUM_THREADS"]
        else: os.environ["OPENBLAS_NUM_THREADS"] = original_threads
        logger.debug("Restored OPENBLAS_NUM_THREADS.")
    if wsclean_success: logger.info("WSClean UV-Join imaging complete.")
    else: logger.error("One or more WSClean UV-Join steps failed.")
    return wsclean_success

def run_coadd_imaging(qa_dir, logger):
    """Finds available sub-band FITS images and stacks them."""
    logger.info(f"--- Step 9 (Co-add Mode): Stacking FITS images from {qa_dir} ---")
    image_types = {'I-full': '-I-full-MFS-image.fits', 'I-tapered': '-I-tapered-MFS-image.fits', 'V-tapered': '-V-tapered-MFS-image.fits'}
    bands_available = [];
    for band in BANDS_TO_IMAGE:
         if glob.glob(os.path.join(qa_dir, f"{band}-I-full*-MFS-image.fits")): bands_available.append(band)
    if not bands_available: logger.error("No sub-band FITS images found. Cannot co-add."); return False
    logger.info(f"Found images for {len(bands_available)} bands (based on I-full): {bands_available}")
    logger.warning("Co-adding logic currently only stacks the first time interval (e.g., -t0000) if intervals were used.")
    for type_key, file_suffix in image_types.items():
        logger.info(f"Stacking image type: {type_key}"); image_stack, header_template = [], None
        files_to_stack = [];
        for sub_band in bands_available:
            search_pattern = os.path.join(qa_dir, f"{sub_band}{type_key.replace('I-', '-I-')}*-MFS-image.fits")
            found_images = sorted(glob.glob(search_pattern))
            if found_images:
                 files_to_stack.append(found_images[0])
                 if len(found_images) > 1: logger.debug(f"Found {len(found_images)} intervals for {sub_band} {type_key}. Using first: {found_images[0]}")
            else: logger.warning(f"Missing expected FITS for {sub_band} {type_key}. Skipping.")
        if not files_to_stack: logger.error(f"No valid FITS found for type {type_key}."); continue
        logger.info(f"Found {len(files_to_stack)} files to stack for {type_key}.")
        def get_freq_sort_key(fp):
            bn = os.path.basename(fp);
            for i, band_n in enumerate(ALL_POSSIBLE_SUB_BANDS):
                if bn.startswith(band_n): return i
            return -1
        files_to_stack.sort(key=get_freq_sort_key)
        for f_path in files_to_stack:
            try:
                with fits.open(f_path) as hdul:
                    data = np.squeeze(hdul[0].data)
                    if data.ndim != 2: logger.warning(f"Unexpected data shape {hdul[0].data.shape}->{data.shape} in {f_path}. Skipping."); continue
                    if header_template is None:
                        header_template = hdul[0].header
                        if data.shape != (header_template['NAXIS2'], header_template['NAXIS1']): logger.error(f"Inconsistent dims! Exp: {(header_template['NAXIS2'], header_template['NAXIS1'])}, Got: {data.shape}."); image_stack = []; break
                    elif data.shape != (header_template['NAXIS2'], header_template['NAXIS1']): logger.warning(f"Dim mismatch in {f_path}! Exp: {(header_template['NAXIS2'], header_template['NAXIS1'])}, Got: {data.shape}. Skipping."); continue
                    image_stack.append(data)
            except Exception as e: logger.error(f"Failed to read/process {f_path}: {e}")
        if image_stack:
            try:
                final_cube = np.array(image_stack); logger.info(f"Created final cube: {final_cube.shape}")
                for key in list(header_template.keys()):
                    if key.startswith(('CRPIX3','CRVAL3','CDELT3','CTYPE3','CUNIT3','NAXIS3')): del header_template[key]
                header_template['NAXIS'] = 3; header_template['NAXIS3'] = final_cube.shape[0]
                header_template.set('COMMENT', f"Freq axis order: {bands_available}", after='NAXIS3')
                output_filename = os.path.join(qa_dir, f"FullSky_MFS_Archived-{type_key}.fits")
                fits.writeto(output_filename, final_cube, header_template, overwrite=True)
                logger.info(f"Saved co-added image: {output_filename}")
            except Exception as e: logger.error(f"Failed to stack/save {type_key}: {e}", exc_info=True)
        else: logger.warning(f"No valid data to stack for {type_key}.")
    logger.info("--- FITS Co-adding Finished ---"); return True

# ==============================================================================
# === NEW Per-File Pre-Peel Parallel Helper (Stage 1) ===
# ==============================================================================

def _process_one_file_prepeel(args_tuple):
    """
    Helper function for Stage 1 parallel pre-peeling.
    Runs antenna flagging and applycal.
    """
    # Unpack arguments
    ms_file, center_coord, context, sub_band, bp_table_path, caltable_spw_map, script_dir = args_tuple
    
    pid = os.getpid()
    log_prefix = f"[PID {pid}, File {os.path.basename(ms_file)}]"
    
    # We must re-setup a basic logger *within* the child process
    logger = logging.getLogger(f'ArchiveWorker_PrePeelChild_{pid}')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(f'%(asctime)s - [PID {pid}] - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.propagate = False
        
    try:
        # 1. Antenna Flagging
        logger.info("Running antenna flagging...")
        if not run_antenna_flagging(ms_file, context, script_dir, logger): 
            raise RuntimeError(f"Antenna flagging failed")

        # 2. Applycal
        logger.info("Running applycal...")
        if not apply_bandpass_calibration(ms_file, bp_table_path, sub_band, caltable_spw_map, logger): 
            raise RuntimeError(f"Applycal failed")
        
        logger.info("Finished per-file PRE-PEEL processing.")
        return ms_file # Return the path on success
    except Exception as e:
        logger.error(f"FATAL ERROR in pre-peel process {log_prefix}: {e}", exc_info=True)
        return None


# ==============================================================================
# === NEW Mode A: Sub-band Work (Local Orchestrator) (MODIFIED) ===
# ==============================================================================

def run_mode_subband_work(args):
    """
    (Mode A) Called by worker_task.sh for one sub-band.
    Runs local Stage 1 (Pre-peel) and submits local
    Stage 2 (Peel) and Stage 3 (Gather).
    """
    sub_band = ALL_POSSIBLE_SUB_BANDS[args.task_id]
    qa_dir = os.path.join(args.working_dir, 'QA')
    log_filepath = os.path.join(qa_dir, f'archive_processing_S1_Orchestrator_{sub_band}.log')
    logger = setup_task_logger(f'Archive_S1_Orch_{args.task_id:02d}', log_filepath)

    logger.info(f"--- MODE: subband_work (S1 Orchestrator) ---")
    logger.info(f"Task ID: {args.task_id}, Sub-band: {sub_band}")
    logger.info(f"NVMe Dir: {args.working_dir}")
    logger.info(f"Lustre Receipt Dir: {args.lustre_receipt_dir}")

    # --- 1. Find and Copy Files ---
    if args.input_dir:
         logger.info(f"Checking for band presence in {args.input_dir}...")
         band_present = glob.glob(os.path.join(args.input_dir, f'*{sub_band}*.ms'))
         if not band_present: 
             logger.warning(f"{sub_band} not found in {args.input_dir}. Skipping."); 
             sys.exit(EXIT_CODE_SKIP)

    archive_file_list = find_archive_files_for_subband(
        args.start_time, args.end_time, sub_band, logger, args.input_dir
    )
    if not archive_file_list:
        logger.warning(f"No files found for {sub_band}. Exiting."); 
        sys.exit(EXIT_CODE_SKIP)
    
    copied_file_list = copy_and_prepare_files(
        archive_file_list, args.working_dir, sub_band, logger
    )
    if not copied_file_list: 
        raise RuntimeError("Copying failed.")
    
    # --- 2. Build Task List for Pre-peel Pool ---
    caltable_spw_map = pipeline_utils.get_caltable_spw_map(args.bandpass_table)
    if not caltable_spw_map: raise RuntimeError("Failed to generate SPW map.")

    if args.custom_phase_center: 
        center_coord = SkyCoord(args.custom_phase_center, frame='icrs', unit=(u.hourangle, u.deg))
    else:
        midpoint_dt = args.start_time + (args.end_time - args.start_time) / 2
        midpoint_time = Time(midpoint_dt, scale='utc', location=config.OVRO_LOCATION)
        zenith_frame = AltAz(obstime=midpoint_time, location=config.OVRO_LOCATION)
        center_coord = SkyCoord(alt=90*u.deg, az=0*u.deg, frame=zenith_frame).icrs
    
    midpoint_time_obj = args.start_time + (args.end_time - args.start_time) / 2
    context = {'midpoint_time': Time(midpoint_time_obj, scale='utc')} 

    task_args_list = []
    for ms_file in copied_file_list:
        task_args = (
            ms_file, center_coord, context, sub_band, 
            args.bandpass_table, caltable_spw_map, args.script_dir
        )
        task_args_list.append(task_args)

    # --- 3. Run Local Parallel Pre-peeling (AntFlag + Applycal) ---
    logger.info(f"--- Starting local PRE-PEEL pool ({POOL_PROCESSES} processes) for {len(task_args_list)} files... ---")
    
    with multiprocessing.Pool(processes=POOL_PROCESSES) as pool:
        results = list(tqdm(pool.imap(_process_one_file_prepeel, task_args_list), 
                            total=len(task_args_list), desc=f"Pre-peeling {sub_band}"))

    processed_files = [res for res in results if res is not None]
    if len(processed_files) != len(task_args_list):
        num_failed = len(task_args_list) - len(processed_files)
        raise RuntimeError(f"Parallel pre-peel step failed for {num_failed} files.")
    
    logger.info("--- Local PRE-PEEL processing complete. ---")

    # --- 4. Write *Local* Task Map for Stage 2 ---
    task_map_path = os.path.join(args.working_dir, LOCAL_TASK_MAP_FILE)
    logger.info(f"Writing local task map to {task_map_path}")
    with open(task_map_path, 'w') as f:
        for fpath in processed_files:
            f.write(f"{fpath}\n")
    
    num_tasks = len(processed_files)
    if num_tasks == 0:
        logger.warning("No tasks to peel. Skipping Stage 2 and 3 submission.")
        return

    # --- 5. Dynamically Submit Local Stage 2 (Peeling) ---
    logger.info(f"--- Submitting LOCAL STAGE 2 (Peel) as a {num_tasks}-task array job ---")
    logger.info(f"--- Throttling to {PEEL_JOB_SLOTS} parallel jobs, each with {PEEL_JOB_MEM_G}G RAM ---")
    
    current_hostname = socket.gethostname()
    stage2_job_name = f"S2_Peel_{sub_band}"
    stage2_log_out = os.path.join(args.working_dir, 'QA', 'slurm-S2_Peel-%A_%a.out')
    stage2_log_err = os.path.join(args.working_dir, 'QA', 'slurm-S2_Peel-%A_%a.err')
    
    python_cmd = (
        f"conda run -n py38_orca_nkosogor python \"{os.path.join(args.script_dir, 'process_archive.py')}\" "
        f"--mode stage_2_task "
        f"--working_dir \"{args.working_dir}\" " # Pass NVMe task dir
        f"--script_dir \"{args.script_dir}\" "
        f"--peel_sky_model \"{args.peel_sky_model}\" "
        f"--peel_rfi_model \"{args.peel_rfi_model}\" "
        f"{args.peel_sky and '--peel-sky' or ''} "
        f"{args.peel_rfi and '--peel-rfi' or ''} "
    )
    
    sbatch_cmd_stage2 = [
        'sbatch',
        f'--job-name={stage2_job_name}',
        f'--output={stage2_log_out}', f'--error={stage2_log_err}',
        '--partition=general',
        f'--nodelist={current_hostname}', # Constrain to this node
        '--nodes=1', '--ntasks=1', '--cpus-per-task=1', 
        f'--mem={PEEL_JOB_MEM_G}G', # <-- MODIFIED
        '--time=04:00:00',
        f'--array=0-{num_tasks - 1}%{PEEL_JOB_SLOTS}', # <-- MODIFIED
        '--wrap', python_cmd
    ]
    
    logger.info(f"Stage 2 sbatch command: {' '.join(sbatch_cmd_stage2)}")
    
    try:
        result = subprocess.run(sbatch_cmd_stage2, check=True, capture_output=True, text=True)
        stage2_job_id = result.stdout.strip().split()[-1]
        logger.info(f"Successfully submitted Stage 2. Job ID: {stage2_job_id}")
    except subprocess.CalledProcessError as e:
        logger.error("--- FAILED TO SUBMIT STAGE 2 ---")
        logger.error(f"Stderr: {e.stderr}")
        raise RuntimeError("Failed to submit Stage 2 (Peel) job array.")

    # --- 6. Dynamically Submit Local Stage 3 (Gather) ---
    logger.info(f"--- Submitting LOCAL STAGE 3 (Gather) dependent on Job ID {stage2_job_id} ---")

    stage3_job_name = f"S3_Gather_{sub_band}"
    stage3_log_out = os.path.join(args.working_dir, 'QA', 'slurm-S3_Gather-%j.out')
    stage3_log_err = os.path.join(args.working_dir, 'QA', 'slurm-S3_Gather-%j.err')
    
    python_cmd_stage3 = (
        f"conda run -n py38_orca_nkosogor python \"{os.path.join(args.script_dir, 'process_archive.py')}\" "
        f"--mode stage_3_gather "
        f"--working_dir \"{args.working_dir}\" "
        f"--lustre_receipt_dir \"{args.lustre_receipt_dir}\" "
        f"--sub_band \"{sub_band}\" "
        f"--script_dir \"{args.script_dir}\" "
        f"--custom_phase_center \"{center_coord.to_string('hmsdms')}\" "
    )

    sbatch_cmd_stage3 = [
        'sbatch',
        f'--job-name={stage3_job_name}',
        f'--output={stage3_log_out}', f'--error={stage3_log_err}',
        '--partition=general',
        f'--nodelist={current_hostname}', # Constrain to this node
        '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--mem=64G',
        '--time=02:00:00',
        f'--dependency=afterok:{stage2_job_id}',
        '--wrap', python_cmd_stage3
    ]

    logger.info(f"Stage 3 sbatch command: {' '.join(sbatch_cmd_stage3)}")
    
    try:
        result = subprocess.run(sbatch_cmd_stage3, check=True, capture_output=True, text=True)
        stage3_job_id = result.stdout.strip().split()[-1]
        logger.info(f"Successfully submitted Stage 3. Job ID: {stage3_job_id}")
    except subprocess.CalledProcessError as e:
        logger.error("--- FAILED TO SUBMIT STAGE 3 ---")
        logger.error(f"Stderr: {e.stderr}")
        raise RuntimeError("Failed to submit Stage 3 (Gather) job.")

    logger.info(f"--- S1 Orchestrator for {sub_band} Finished ---")


# ==============================================================================
# === NEW Mode: Stage 2 Task (Local Peeling) ===
# ==============================================================================

def run_mode_stage_2_task(args):
    """
    (Mode: stage_2_task) Peels a single file on its local node.
    """
    try:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    except Exception as e:
        print(f"FATAL: Could not get SLURM_ARRAY_TASK_ID: {e}", file=sys.stderr)
        sys.exit(1)
    
    qa_dir = os.path.join(args.working_dir, 'QA', 'stage_2_logs')
    os.makedirs(qa_dir, exist_ok=True)
    log_filepath = os.path.join(qa_dir, f'task_{task_id:05d}.log')
    logger = setup_task_logger(f'Archive_S2_Task_{task_id:05d}', log_filepath)

    logger.info(f"--- STAGE 2 (Peel) Task {task_id} Starting ---")
    
    # --- 1. Find my assigned file ---
    task_map_path = os.path.join(args.working_dir, LOCAL_TASK_MAP_FILE)
    if not os.path.exists(task_map_path):
        logger.error(f"Task map file not found: {task_map_path}")
        raise RuntimeError("Task map not found.")
    
    try:
        with open(task_map_path, 'r') as f:
            all_files = f.read().splitlines()
        
        if task_id >= len(all_files):
            logger.error(f"Task ID {task_id} is out of bounds for task map (len: {len(all_files)})")
            raise RuntimeError("Task ID out of bounds.")
        
        ms_file = all_files[task_id]
        if not os.path.exists(ms_file):
            logger.error(f"Assigned file does not exist: {ms_file}")
            raise RuntimeError(f"Assigned file not found: {ms_file}")
            
    except Exception as e:
        logger.error(f"Failed to read task map: {e}", exc_info=True)
        raise
    
    logger.info(f"Assigned file: {ms_file}")

    # --- 2. Run Peeling (if enabled) ---
    try:
        if args.peel_sky:
            if not run_peeling(ms_file, args.peel_sky_model, logger):
                raise RuntimeError(f"Astro Peeling failed: {ms_file}")
        else:
            logger.info("Astro peeling not enabled. Skipping.")
            
        if args.peel_rfi:
            if not run_peeling_rfi(ms_file, args.peel_rfi_model, logger):
                raise RuntimeError(f"RFI Peeling failed: {ms_file}")
        else:
            logger.info("RFI peeling not enabled. Skipping.")
    
    except Exception as e:
        logger.error(f"Peeling failed for {ms_file}: {e}", exc_info=True)
        raise
    
    logger.info(f"--- STAGE 2 (Peel) Task {task_id} Finished ---")


# ==============================================================================
# === NEW Mode: Stage 3 Gather (Local Concat) ===
# ==============================================================================

def run_mode_stage_3_gather(args):
    """
    (Mode: stage_3_gather) Concatenates, post-processes,
    and writes Lustre receipt.
    """
    sub_band = args.sub_band
    qa_dir = os.path.join(args.working_dir, 'QA')
    log_filepath = os.path.join(qa_dir, f'archive_processing_S3_Gather_{sub_band}.log')
    logger = setup_task_logger(f'Archive_S3_Gather_{sub_band}', log_filepath)

    logger.info(f"--- STAGE 3 (Gather) for {sub_band} Starting ---")
    logger.info(f"NVMe Dir: {args.working_dir}")

    # --- 1. Load the task map to find all (now peeled) files ---
    task_map_path = os.path.join(args.working_dir, LOCAL_TASK_MAP_FILE)
    if not os.path.exists(task_map_path):
        raise RuntimeError(f"Task map file not found: {task_map_path}")
    
    with open(task_map_path, 'r') as f:
        all_peeled_files = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Found {len(all_peeled_files)} peeled files from task map.")
    
    # --- 2. Concatenate and Post-process ---
    try:
        concat_ms_path = concatenate_sub_band(all_peeled_files, args.working_dir, sub_band, logger, is_peeling_mode=True)
        if not concat_ms_path: raise RuntimeError("Concat failed.")
        
        if not run_fix_field_id(concat_ms_path, logger): 
            raise RuntimeError("Fix ID on concat failed.") 
        
        center_coord = SkyCoord(args.custom_phase_center, frame='icrs', unit=(u.hourangle, u.deg))
        if not set_common_phase_center(concat_ms_path, center_coord, logger): 
            raise RuntimeError(f"Phase center failed on concat") 
            
        if not run_aoflagger_on_sub_band(concat_ms_path, sub_band, logger): 
            raise RuntimeError("AOFlagger failed.")
            
    except Exception as e:
        logger.error(f"Failed post-peel processing for {sub_band}: {e}", exc_info=True)
        raise

    # --- 3. Write Lustre Receipt File ---
    current_hostname = socket.gethostname()
    receipt_file_path = os.path.join(args.lustre_receipt_dir, f"{sub_band}.receipt")
    
    # This is the *parent* NVMe dir for the *whole task*, for later cleanup
    task_parent_dir = args.working_dir
    
    receipt_content = f"{current_hostname},{concat_ms_path},{task_parent_dir}"
    
    logger.info(f"Writing receipt to Lustre: {receipt_file_path}")
    logger.info(f"Receipt content: {receipt_content}")
    try:
        with open(receipt_file_path, 'w') as f:
            f.write(receipt_content)
    except Exception as e:
        logger.error(f"Failed to write receipt file: {e}", exc_info=True)
        raise

    logger.info(f"--- STAGE 3 (Gather) for {sub_band} Finished ---")


# ==============================================================================
# === NEW Mode B: Final Image (Central Aggregator) (MODIFIED) ===
# ==============================================================================

def run_mode_final_image(args):
    """
    (Mode: final_image) Aggregates MS files from worker nodes
    via scp, runs UV-Join imaging, and cleans up.
    """
    qa_dir = os.path.join(args.working_dir, 'QA')
    log_filepath = os.path.join(qa_dir, 'archive_processing_S4_Final_Image.log')
    logger = setup_task_logger('Archive_S4_Imager', log_filepath)

    logger.info("--- MODE: final_image (S4 Aggregator) ---")
    logger.info(f"NVMe Dir: {args.working_dir}")
    logger.info(f"Lustre Receipt Dir: {args.lustre_receipt_dir}")

    receipt_files = sorted(glob.glob(os.path.join(args.lustre_receipt_dir, "*.receipt")))
    if not receipt_files:
        raise RuntimeError(f"No receipt files found in {args.lustre_receipt_dir}")
    
    logger.info(f"Found {len(receipt_files)} receipts.")
    
    local_ms_list = []
    receipt_data = [] # Store for cleanup
    
    # --- 1. Aggregate MS files via scp ---
    for receipt_path in receipt_files:
        try:
            with open(receipt_path, 'r') as f:
                content = f.read().strip()
            
            hostname, remote_ms_path, remote_task_dir = content.split(',')
            sub_band = os.path.basename(receipt_path).replace('.receipt', '')
            logger.info(f"Aggregating {sub_band} from {hostname}...")
            
            local_dest_path = os.path.join(args.working_dir, 'ms_data', os.path.basename(remote_ms_path))
            
            scp_cmd = ['scp', '-o', 'StrictHostKeyChecking=no', 
                       f'gh@{hostname}:{remote_ms_path}', local_dest_path]
            
            logger.debug(f"Executing: {' '.join(scp_cmd)}")
            # MODIFIED: Add better error logging
            result = subprocess.run(scp_cmd, check=False, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode != 0:
                logger.error(f"scp command failed: {' '.join(scp_cmd)}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"scp failed to copy {remote_ms_path}")
            
            if not os.path.exists(local_dest_path):
                raise RuntimeError(f"scp completed but file not found: {local_dest_path}")
                
            local_ms_list.append(local_dest_path)
            receipt_data.append({'sub_band': sub_band, 'hostname': hostname, 'remote_task_dir': remote_task_dir})
            logger.info(f"Successfully copied {sub_band}.")
            
        except Exception as e:
            logger.error(f"Failed to process receipt {receipt_path}: {e}")
            # Continue trying to aggregate other files
    
    if not local_ms_list:
        raise RuntimeError("Failed to aggregate any MS files.")
        
    # --- 2. Run Final UV-Join Imaging ---
    logger.info("Aggregation complete. Starting UV-Join Imaging.")
    
    ms_list_to_image = [ms for ms in local_ms_list if not any(b in os.path.basename(ms) for b in ['13MHz', '82MHz'])]
    logger.info(f"Found {len(ms_list_to_image)} MS files for UV-Join imaging.")
    
    # Determine num_integrations (needed by run_final_imaging)
    num_integrations = 0
    if args.intervals_out == -1:
        try:
            # Get from one of the MS files
            msmd = pipeline_utils.get_ms_metadata(ms_list_to_image[0])
            if msmd and 'num_integrations' in msmd:
                num_integrations = msmd['num_integrations']
                logger.info(f"Determined {num_integrations} integrations from MS metadata.")
        except Exception as e:
            logger.warning(f"Could not get integration count from MS: {e}")

    if not ms_list_to_image:
        logger.warning("No MS files to image. Skipping WSClean.")
    else:
        image_prefix = os.path.join(qa_dir, 'FullSky_MFS_Archived_UVJoin')
        if not run_final_imaging(ms_list_to_image, image_prefix, logger, 
                                 intervals_out=args.intervals_out, 
                                 num_integrations=num_integrations): 
            raise RuntimeError("WSClean UV-Join imaging failed.")

    # --- 3. Remote Cleanup ---
    logger.info("Imaging complete. Cleaning up remote worker NVMe directories.")
    for data in receipt_data:
        try:
            logger.info(f"Cleaning {data['sub_band']} data from {data['hostname']}:{data['remote_task_dir']}")
            ssh_cmd = ['ssh', '-o', 'StrictHostKeyChecking=no', 
                       f'gh@{data["hostname"]}', f"rm -rf {data['remote_task_dir']}"]
            
            logger.debug(f"Executing: {' '.join(ssh_cmd)}")
            # MODIFIED: Add better error logging
            result = subprocess.run(ssh_cmd, check=False, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode != 0:
                logger.warning(f"SSH command failed: {' '.join(ssh_cmd)}")
                logger.warning(f"STDOUT: {result.stdout}")
                logger.warning(f"STDERR: {result.stderr}")
            else:
                logger.info(f"Cleanup successful for {data['sub_band']}.")
        except Exception as e:
            logger.warning(f"Failed to clean up {data['sub_band']} on {data['hostname']}: {e}")

    logger.info("--- MODE: final_image (S4 Aggregator) Finished ---")


# ==============================================================================
# === Main Execution ===
# ==============================================================================

def main():
    parser = argparse.ArgumentParser( description="OVRO-LWA Night-Time Archive Processing (4-Stage Workflow)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # --- Mode ---
    parser.add_argument("--mode", type=str, required=True, 
                        choices=['subband_work', 'stage_2_task', 'stage_3_gather', 'final_image', 'run_coadd'],
                        help="The execution mode for the script.")

    # --- Common arguments ---
    parser.add_argument("--working_dir", type=str, required=True, help="Path to temporary (NVMe) directory.")
    parser.add_argument("--script_dir", type=str, help="Path to pipeline script directory (for imports).")
    parser.add_argument("--start_time", type=valid_datetime, help="Start Time (UTC). Format: 'YYYY-MM-DD:HH:MM:SS'")
    parser.add_argument("--end_time", type=valid_datetime, help="End Time (UTC). Format: 'YYYY-MM-DD:HH:MM:SS'")
    parser.add_argument("--bandpass_table", type=str, help="Path to the bandpass calibration table.")
    parser.add_argument("--input-dir", type=str, default=None, help="Optional: Path to directory containing all MS files.")
    parser.add_argument("--custom_phase_center", type=str, default=None, help="Custom phase center (ICRS).")
    parser.add_argument("--peel-sky", action='store_true', help="Enable astrophysical peeling.")
    parser.add_argument("--peel-rfi", action='store_true', help="Enable RFI peeling.")
    parser.add_argument("--peel_sky_model", type=str, help="Full path to astro peel model JSON.")
    parser.add_argument("--peel_rfi_model", type=str, help="Full path to RFI peel model JSON.")
    parser.add_argument("--intervals-out", type=str, nargs='?', const=-1, default=None,
                        help="Split imaging into intervals. No value: 1 per integration. [N]: N intervals.")

    # --- Mode-specific arguments ---
    parser.add_argument("--task_id", type=int, help="SLURM_ARRAY_TASK_ID (for mode: subband_work)")
    parser.add_argument("--sub_band", type=str, help="Sub-band name (for mode: stage_3_gather)")
    parser.add_argument("--lustre_receipt_dir", type=str, help="Path to Lustre receipt directory.")
    parser.add_argument("--lustre_qa_dir", type=str, help="Path to Lustre QA directory (for mode: final_image).")

    args = parser.parse_args()
    
    # --- Setup Logger (path depends on mode) ---
    logger = None
    try:
        if args.mode == 'subband_work':
            if args.task_id is None: parser.error("--task_id is required for --mode subband_work")
            sub_band = ALL_POSSIBLE_SUB_BANDS[args.task_id]
            qa_dir = os.path.join(args.working_dir, 'QA'); os.makedirs(qa_dir, exist_ok=True)
            log_filepath = os.path.join(qa_dir, f'archive_processing_S1_Orchestrator_{sub_band}.log')
            logger = setup_task_logger(f'Archive_S1_Orch_{args.task_id:02d}', log_filepath)
            run_mode_subband_work(args)

        elif args.mode == 'stage_2_task':
            # Logger is set up inside the function
            run_mode_stage_2_task(args)

        elif args.mode == 'stage_3_gather':
            if not args.sub_band: parser.error("--sub_band is required for --mode stage_3_gather")
            qa_dir = os.path.join(args.working_dir, 'QA'); os.makedirs(qa_dir, exist_ok=True)
            log_filepath = os.path.join(qa_dir, f'archive_processing_S3_Gather_{args.sub_band}.log')
            logger = setup_task_logger(f'Archive_S3_Gather_{args.sub_band}', log_filepath)
            run_mode_stage_3_gather(args)

        elif args.mode == 'final_image':
            qa_dir = os.path.join(args.working_dir, 'QA'); os.makedirs(qa_dir, exist_ok=True)
            log_filepath = os.path.join(qa_dir, 'archive_processing_S4_Final_Image.log')
            logger = setup_task_logger('Archive_S4_Imager', log_filepath)
            run_mode_final_image(args)

        elif args.mode == 'run_coadd':
            qa_dir = os.path.join(args.working_dir, 'QA'); os.makedirs(qa_dir, exist_ok=True)
            log_filepath = os.path.join(qa_dir, 'archive_processing_COADD.log'); 
            logger = setup_task_logger('ArchiveCoadd', log_filepath)
            logger.info(f"--- Started (CO-ADD MODE) ---"); logger.info(f"Reading FITS from: {qa_dir}")
            if not run_coadd_imaging(qa_dir, logger): raise RuntimeError("FITS Co-adding failed.")
            logger.info("--- Co-adding Finished ---")

    except argparse.ArgumentTypeError as e: print(f"FATAL Argument Error: {e}", file=sys.stderr); sys.exit(2)
    except Exception as e:
        if logger: logger.error(f"\nFATAL: Pipeline step failed: {e}", exc_info=True); logger.error(f"Processing halted. Data may remain in: {args.working_dir}")
        else: print(f"\nFATAL: Pipeline step failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

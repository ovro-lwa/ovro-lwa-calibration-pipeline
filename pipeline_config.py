# pipeline_config.py
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
import os
import sys
import glob
import logging

# Initialize a basic logger for config loading issues
config_logger = logging.getLogger('PipelineConfig')

# --- General Pipeline Configuration ---
# Name of the required conda environment to run the pipeline.
# Set to None to disable the check.
REQUIRED_CONDA_ENV = 'py38_orca_nkosogor'
PARENT_OUTPUT_DIR = '/fast/gh/calibration/'
# Directory for reference tables (Base path)
REFERENCE_CALIBRATION_BASE_DIR = '/lustre/gh/calibration/pipeline/reference'
# Path to the static flux model file (Used in SkyModel and QA)
FLUX_MODEL_NPZ_PATH = '/lustre/gh/calibration/pipeline/reference/fluxscale/primary_calibrator_flux_models.npz' 

# --- NEW: Data Structure Constants ---
# Master list of all possible sub-bands in their canonical frequency order
ALL_POSSIBLE_SUB_BANDS = [
    '13MHz', '18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz',
    '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz'
]

# Channel bandwidth (kHz)
CHAN_BW_KHZ = 23.926

# Number of channels per sub-band
NCHAN_PER_SUBBAND = 192

# Approximate sub-band bandwidth (MHz)
SUBBAND_BW_MHZ = NCHAN_PER_SUBBAND * CHAN_BW_KHZ * 1e-3 # Approx 4.6 MHz
# --- End NEW ---

# Hardcoded mapping of Correlator Number to SNAP2 Location 
# NOTE: This dictionary must contain the complete mapping for 352 antennas.
ANTENNA_MAPPING = {
    0: 'R3C1A', 1: 'R3C1B', 2: 'R3C2A', 3: 'R3C2B', 4: 'R3C3A', 5: 'R3C3B', 6: 'R3C4A', 7: 'R3C4B', 
    # ... (rest of mapping unchanged) ...
    344: 'R13C13A', 345: 'R13C13B', 346: 'R13C14A', 347: 'R13C14B', 348: 'R13C15A', 349: 'R13C15B', 350: 'R13C16A', 351: 'R13C16B', 
}

if len(ANTENNA_MAPPING) != 352:
    print(f"WARNING: ANTENNA_MAPPING is incomplete (Found {len(ANTENNA_MAPPING)} entries).", file=sys.stderr)
N_ANTENNAS = 352
OVRO_LOCATION = EarthLocation(lat=37.239780*u.deg, lon=-118.276250*u.deg, height=1222*u.m)
CONDA_ENV_MNC = 'development'
ADDITIONAL_BAD_ANTENNAS = [29, 37, 41, 42, 56, 92]
FALLBACK_BAD_ANTENNAS = [
    69, 131, 166, 169, 143, 176, 178, 198, 225, 180, 182, 290, 243, 242, 244, 
    295, 296, 298, 301, 300, 280, 309, 346, 37, 5, 41, 40, 42, 44, 92, 190, 
    125, 56, 29, 126
]
CHGCENTRE_PATH = os.environ.get('CHGCENTRE_BIN', '/opt/bin/chgcentre')
AOFLAGGER_EXECUTABLE = os.environ.get('AOFLAGGER_BIN', '/opt/bin/aoflagger')
WSCLEAN_PATH = os.environ.get('WSCLEAN_BIN', '/opt/bin/wsclean')
AOFLAGGER_STRATEGY_PATH = os.environ.get('AOFLAGGER_STRATEGY', '/lustre/ghellbourg/AOFlagger_strat_opt/LWA_opt_GH1.lua')
def get_mnc_helper_path(script_dir):
    return os.path.join(script_dir, 'get_bad_antennas_mnc.py')
BEAM_MODEL_H5 = '/lustre/pipeline/beam-models/OVRO-LWA_MROsoil_updatedheight.h5'
FLUX_THRESHOLD_JY = 1000.0
ELEVATION_LIMIT_DEG = 20.0
PRODUCE_SECONDARY_SOURCE_CSV = True
PRIMARY_SOURCES = {
    'CygA': {'ra': '19h59m28.356s', 'dec': '+40d44m02.09s', 'skycoord': SkyCoord('19h59m28.356s', '+40d44m02.09s', frame='icrs')},
    'CasA': {'ra': '23h23m24.000s', 'dec': '+58d48m54.00s', 'skycoord': SkyCoord('23h23m24.000s', '+58d48m54.00s', frame='icrs')},
    'TauA': {'ra': '05h34m31.94s', 'dec': '+22d00m52.2s', 'skycoord': SkyCoord('05h34m31.94s', '+22d00m52.2s', frame='icrs')},
    'VirA': {'ra': '12h30m49.42338s', 'dec': '+12d23m28.0439s', 'skycoord': SkyCoord('12h30m49.42338s', '+12d23m28.0439s', frame='icrs')}
}
SECONDARY_SOURCES = {
    '3C48': {'ra': '01h37m41.3s', 'dec': '+33d09m35s', 'coeffs': [43.874, -0.349, -0.374]},
    # ... (rest of sources) ...
    '3C446': {'ra': '22h25m47.2s', 'dec': '-04d55m45s', 'coeffs': [14.461, -0.425, -0.386]},
}
CAL_REFANT = '283'
CAL_PHASE_CENTER = 'zenith'
CAL_SINGLE_SOURCE_OVERRIDE = None
CAL_UVRANGE_DEFAULT = ">5lambda,<350lambda"
CAL_UVRANGE_VIRA = ">5lambda,<185lambda"
BANDPASS_SOLINT = 'inf'
BANDPASS_MIN_SNR = 3.0
DELAY_CAL_MIN_SNR = 3.0
DELAY_CAL_MIN_FREQ_MHZ = 41.0 
DELAY_MIN_ELEVATION_DEG = 40.0
def _find_latest_table_in_dir(directory, table_type):
    search_pattern = os.path.join(directory, f"*{table_type}*")
    matching_tables = [p for p in glob.glob(search_pattern) if os.path.isdir(p)]
    if not matching_tables: return None
    latest_table = sorted(matching_tables, key=os.path.getmtime, reverse=True)[0]
    return latest_table

def get_reference_table_path(table_type, lst_hour_str, search_radius_h=8):
    if table_type not in ['delay', 'bandpass']: config_logger.error(f"Invalid table type: {table_type}"); return None
    if logging.getLogger('OVRO_Pipeline').handlers: logger = logging.getLogger('OVRO_Pipeline.Config')
    else: logger = config_logger
    type_base_dir = os.path.join(REFERENCE_CALIBRATION_BASE_DIR, table_type)
    if not os.path.exists(type_base_dir): logger.warning(f"Ref dir not found: {type_base_dir}"); return None
    try: target_lst = int(lst_hour_str.replace('h', ''))
    except ValueError: logger.error(f"Invalid LST format: {lst_hour_str}"); return None
    available_lsts = {}
    if os.path.exists(type_base_dir):
        for item in os.listdir(type_base_dir):
            item_path = os.path.join(type_base_dir, item)
            if os.path.isdir(item_path) and item.endswith('h'):
                try:
                    lst_val = int(item.replace('h', ''))
                    if _find_latest_table_in_dir(item_path, table_type): available_lsts[lst_val] = item
                except ValueError: continue
    if not available_lsts: logger.warning(f"No valid LST ref dirs found in {type_base_dir}"); return None
    min_diff, nearest_lst = float('inf'), None
    for lst_val in available_lsts.keys():
        diff = abs(target_lst - lst_val); wrap_diff = min(diff, 24 - diff)
        if wrap_diff < min_diff: min_diff, nearest_lst = wrap_diff, lst_val
        elif wrap_diff == min_diff:
            if nearest_lst is None or abs(target_lst - lst_val) < abs(target_lst - nearest_lst): nearest_lst = lst_val
    if min_diff > search_radius_h: logger.warning(f"No ref LST found within +/-{search_radius_h}h of {lst_hour_str}."); return None
    selected_lst_str = available_lsts[nearest_lst]
    selected_dir = os.path.join(type_base_dir, selected_lst_str)
    latest_table = _find_latest_table_in_dir(selected_dir, table_type)
    if latest_table:
        if selected_lst_str != lst_hour_str: logger.info(f"Using nearest ref {table_type} table (LST {selected_lst_str}) for LST {lst_hour_str}.")
        else: logger.info(f"Found exact ref {table_type} table for LST {lst_hour_str}.")
        logger.debug(f"Selected ref table: {os.path.basename(latest_table)}")
        return latest_table
    else: logger.error("Table disappeared after check."); return None
DELAY_DIFF_THRESHOLD_NS = 100.0
SNAP2_FAILURE_THRESHOLD_PERCENT = 50.0 
WSCLEAN_BASE_PARAMS = {
    'pol': 'I', 'mgain': 0.85, 'horizon-mask': '10deg', 'taper-inner-tukey': 30,
    'no-update-model-required': True, 'quiet': True, 'data-column': 'CORRECTED_DATA'
}
WSCLEAN_SPECTRUM_PARAMS = {
    'size': '512 512', 'niter': 100, 'scale': 0.01, 'taper-inner-tukey': 30, 'weight': 'briggs 1',
    'fit-spectral-pol': 4, 'join-channels': True,
}
WSCLEAN_SCINTILLATION_PARAMS = {
    'size': '512 512', 'niter': 100, 'scale': 0.01, 'taper-inner-tukey': 30, 'weight': 'briggs 1',
}
WSCLEAN_FULLSKY_PARAMS = {
    'size': '4096 4096', 'scale': 0.03125, 'weight': 'briggs 0',
    'fit-spectral-pol': 4, 'niter': 10000, 'parallel-reordering': 10,
    'mem': 50, 'local-rms': True, 'taper-inner-tukey': 3, 'auto-threshold': 0.5, 'auto-mask': 3,
    'join-channels': True,
}

# --- NEW Bandpass Flagging Constants (from test_bp_flagging.py) ---
# Step 0: Normalization
NORMALIZATION_FREQ_RANGE_MHZ = (40.0, 55.0) # Range for per-ant normalization

# Step 1: Per-channel scatter flagging
CHANNEL_SCATTER_CLEAN_FREQ_RANGE_MHZ = (40.0, 60.0)
CHANNEL_SCATTER_THRESHOLD_MULTIPLIER = 7.0

# Step 2 & 3: Template creation and deviation flagging
TEMPLATE_SMOOTHING_KERNEL_SIZE = 51 # Kernel size for smoothing the template (must be odd)
DEVIATION_FLAG_SIGMA = 7.0

# Step 4: Per-antenna gain outlier flagging
GAIN_OUTLIER_SIGMA = 7.0

# Step 5: Iterative outlier flagging (re-using OUTLIER_FLAG_SIGMA/ITERATIONS)
# OUTLIER_FLAG_SIGMA = 3.0 
# OUTLIER_FLAG_ITERATIONS = 20 
BP_ITERATIVE_OUTLIER_FLAG_ITERATIONS = 15
BP_ITERATIVE_OUTLIER_FLAG_SIGMA = 3.0

# Step 6: Channel quorum flagging
CHANNEL_QUORUM_THRESHOLD = 0.5 # Flag channel if > 50% of antennas are flagged

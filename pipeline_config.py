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
    8: 'R3C5A', 9: 'R3C5B', 10: 'R3C6A', 11: 'R3C6B', 12: 'R3C7A', 13: 'R3C7B', 14: 'R3C8A', 15: 'R3C8B',
    16: 'R3C9A', 17: 'R3C9B', 18: 'R3C10A', 19: 'R3C10B', 20: 'R3C11A', 21: 'R3C11B', 22: 'R3C12A', 23: 'R3C12B',
    24: 'R3C13A', 25: 'R3C13B', 26: 'R3C14A', 27: 'R3C14B', 28: 'R3C15A', 29: 'R3C15B', 30: 'R3C16A', 31: 'R3C16B',
    32: 'R4C1A', 33: 'R4C1B', 34: 'R4C2A', 35: 'R4C2B', 36: 'R4C3A', 37: 'R4C3B', 38: 'R4C4A', 39: 'R4C4B',
    40: 'R4C5A', 41: 'R4C5B', 42: 'R4C6A', 43: 'R4C6B', 44: 'R4C7A', 45: 'R4C7B', 46: 'R4C8A', 47: 'R4C8B',
    48: 'R4C9A', 49: 'R4C9B', 50: 'R4C10A', 51: 'R4C10B', 52: 'R4C11A', 53: 'R4C11B', 54: 'R4C12A', 55: 'R4C12B',
    56: 'R4C13A', 57: 'R4C13B', 58: 'R4C14A', 59: 'R4C14B', 60: 'R4C15A', 61: 'R4C15B', 62: 'R4C16A', 63: 'R4C16B',
    64: 'R5C1A', 65: 'R5C1B', 66: 'R5C2A', 67: 'R5C2B', 68: 'R5C3A', 69: 'R5C3B', 70: 'R5C4A', 71: 'R5C4B',
    72: 'R5C5A', 73: 'R5C5B', 74: 'R5C6A', 75: 'R5C6B', 76: 'R5C7A', 77: 'R5C7B', 78: 'R5C8A', 79: 'R5C8B',
    80: 'R5C9A', 81: 'R5C9B', 82: 'R5C10A', 83: 'R5C10B', 84: 'R5C11A', 85: 'R5C11B', 86: 'R5C12A', 87: 'R5C12B',
    88: 'R5C13A', 89: 'R5C13B', 90: 'R5C14A', 91: 'R5C14B', 92: 'R5C15A', 93: 'R5C15B', 94: 'R5C16A', 95: 'R5C16B',
    96: 'R6C1A', 97: 'R6C1B', 98: 'R6C2A', 99: 'R6C2B', 100: 'R6C3A', 101: 'R6C3B', 102: 'R6C4A', 103: 'R6C4B',
    104: 'R6C5A', 105: 'R6C5B', 106: 'R6C6A', 107: 'R6C6B', 108: 'R6C7A', 109: 'R6C7B', 110: 'R6C8A', 111: 'R6C8B',
    112: 'R6C9A', 113: 'R6C9B', 114: 'R6C10A', 115: 'R6C10B', 116: 'R6C11A', 117: 'R6C11B', 118: 'R6C12A', 119: 'R6C12B',
    120: 'R6C13A', 121: 'R6C13B', 122: 'R6C14A', 123: 'R6C14B', 124: 'R6C15A', 125: 'R6C15B', 126: 'R6C16A', 127: 'R6C16B',
    128: 'R7C1A', 129: 'R7C1B', 130: 'R7C2A', 131: 'R7C2B', 132: 'R7C3A', 133: 'R7C3B', 134: 'R7C4A', 135: 'R7C4B',
    136: 'R7C5A', 137: 'R7C5B', 138: 'R7C6A', 139: 'R7C6B', 140: 'R7C7A', 141: 'R7C7B', 142: 'R7C8A', 143: 'R7C8B',
    144: 'R7C9A', 145: 'R7C9B', 146: 'R7C10A', 147: 'R7C10B', 148: 'R7C11A', 149: 'R7C11B', 150: 'R7C12A', 151: 'R7C12B',
    152: 'R7C13A', 153: 'R7C13B', 154: 'R7C14A', 155: 'R7C14B', 156: 'R7C15A', 157: 'R7C15B', 158: 'R7C16A', 159: 'R7C16B',
    160: 'R8C1A', 161: 'R8C1B', 162: 'R8C2A', 163: 'R8C2B', 164: 'R8C3A', 165: 'R8C3B', 166: 'R8C4A', 167: 'R8C4B',
    168: 'R8C5A', 169: 'R8C5B', 170: 'R8C6A', 171: 'R8C6B', 172: 'R8C7A', 173: 'R7C7B', 174: 'R8C8A', 175: 'R8C8B',
    176: 'R8C9A', 177: 'R8C9B', 178: 'R8C10A', 179: 'R8C10B', 180: 'R8C11A', 181: 'R8C11B', 182: 'R8C12A', 183: 'R8C12B',
    184: 'R8C13A', 185: 'R8C13B', 186: 'R8C14A', 187: 'R8C14B', 188: 'R8C15A', 189: 'R8C15B', 190: 'R8C16A', 191: 'R8C16B',
    192: 'R9C1A', 193: 'R9C1B', 194: 'R9C2A', 195: 'R9C2B', 196: 'R9C3A', 197: 'R9C3B', 198: 'R9C4A', 199: 'R9C4B',
    200: 'R9C5A', 201: 'R9C5B', 202: 'R9C6A', 203: 'R9C6B', 204: 'R9C7A', 205: 'R9C7B', 206: 'R9C8A', 207: 'R9C8B',
    208: 'R9C9A', 209: 'R9C9B', 210: 'R9C10A', 211: 'R9C10B', 212: 'R9C11A', 213: 'R9C11B', 214: 'R9C12A', 215: 'R9C12B',
    216: 'R9C13A', 217: 'R9C13B', 218: 'R9C14A', 219: 'R9C14B', 220: 'R9C15A', 221: 'R9C15B', 222: 'R9C16A', 223: 'R9C16B',
    224: 'R10C1A', 225: 'R10C1B', 226: 'R10C2A', 227: 'R10C2B', 228: 'R10C3A', 229: 'R10C3B', 230: 'R10C4A', 231: 'R10C4B',
    232: 'R10C5A', 233: 'R10C5B', 234: 'R10C6A', 235: 'R10C6B', 236: 'R10C7A', 237: 'R10C7B', 238: 'R10C8A', 239: 'R10C8B',
    240: 'R10C9A', 241: 'R10C9B', 242: 'R10C10A', 243: 'R10C10B', 244: 'R10C11A', 245: 'R10C11B', 246: 'R10C12A', 247: 'R10C12B',
    248: 'R10C13A', 249: 'R10C13B', 250: 'R10C14A', 251: 'R10C14B', 252: 'R10C15A', 253: 'R10C15B', 254: 'R10C16A', 255: 'R10C16B',
    256: 'R11C1A', 257: 'R11C1B', 258: 'R11C2A', 259: 'R11C2B', 260: 'R11C3A', 261: 'R11C3B', 262: 'R11C4A', 263: 'R11C4B',
    264: 'R11C5A', 265: 'R11C5B', 266: 'R11C6A', 267: 'R11C6B', 268: 'R11C7A', 269: 'R11C7B', 270: 'R11C8A', 271: 'R11C8B',
    272: 'R11C9A', 273: 'R11C9B', 274: 'R11C10A', 275: 'R11C10B', 276: 'R11C11A', 277: 'R11C11B', 278: 'R11C12A', 279: 'R11C12B',
    280: 'R11C13A', 281: 'R11C13B', 282: 'R11C14A', 283: 'R11C14B', 284: 'R11C15A', 285: 'R11C15B', 286: 'R11C16A', 287: 'R11C16B',
    288: 'R12C1A', 289: 'R12C1B', 290: 'R12C2A', 291: 'R12C2B', 292: 'R12C3A', 293: 'R12C3B', 294: 'R12C4A', 295: 'R12C4B',
    296: 'R12C5A', 297: 'R12C5B', 298: 'R12C6A', 299: 'R12C6B', 300: 'R12C7A', 301: 'R12C7B', 302: 'R12C8A', 303: 'R12C8B',
    304: 'R12C9A', 305: 'R12C9B', 306: 'R12C10A', 307: 'R12C10B', 308: 'R12C11A', 309: 'R12C11B', 310: 'R12C12A', 311: 'R12C12B',
    312: 'R12C13A', 313: 'R12C13B', 314: 'R12C14A', 315: 'R12C14B', 316: 'R12C15A', 317: 'R12C15B', 318: 'R12C16A', 319: 'R12C16B',
    320: 'R13C1A', 321: 'R13C1B', 322: 'R13C2A', 323: 'R13C2B', 324: 'R13C3A', 325: 'R13C3B', 326: 'R13C4A', 327: 'R13C4B',
    328: 'R13C5A', 329: 'R13C5B', 330: 'R13C6A', 331: 'R13C6B', 332: 'R13C7A', 333: 'R13C7B', 334: 'R13C8A', 335: 'R13C8B',
    336: 'R13C9A', 337: 'R13C9B', 338: 'R13C10A', 339: 'R13C10B', 340: 'R13C11A', 341: 'R13C11B', 342: 'R13C12A', 343: 'R13C12B',
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
    '3C67': {'ra': '02h24m29.5s', 'dec': '+31d13m03s', 'coeffs': [16.608, -0.610, -0.348]},
    '3C123': {'ra': '04h37m04.4s', 'dec': '+29d40m14s', 'coeffs': [64.586, -0.655, -0.294]},
    '3C138': {'ra': '05h21m10.0s', 'dec': '+16d38m22s', 'coeffs': [13.550, -0.544, -0.325]},
    '3C147': {'ra': '05h42m36.1s', 'dec': '+49d51m07s', 'coeffs': [47.136, -0.548, -0.309]},
    '3C190': {'ra': '08h01m29.4s', 'dec': '+14d14m43s', 'coeffs': [20.099, -0.633, -0.353]},
    '3C196': {'ra': '08h13m36.0s', 'dec': '+48d13m03s', 'coeffs': [36.345, -0.546, -0.373]},
    '3C216': {'ra': '09h09m33.7s', 'dec': '+42d53m46s', 'coeffs': [13.613, -0.493, -0.357]},
    '3C220.1': {'ra': '09h32m08.3s', 'dec': '+79d07m23s', 'coeffs': [10.327, -0.514, -0.355]},
    '3C220.3': {'ra': '09h36m10.4s', 'dec': '+36d07m03s', 'coeffs': [10.500, -0.647, -0.337]},
    '3C249.1': {'ra': '11h04m12.3s', 'dec': '+76d58m58s', 'coeffs': [8.278, -0.649, -0.306]},
    '3C286': {'ra': '13h31m08.3s', 'dec': '+30d30m33s', 'coeffs': [26.141, -0.347, -0.337]},
    '3C295': {'ra': '14h11m20.5s', 'dec': '+52d12m10s', 'coeffs': [56.503, -0.575, -0.371]},
    '3C298': {'ra': '14h19m08.2s', 'dec': '+06d28m35s', 'coeffs': [22.639, -0.666, -0.345]},
    '3C309.1': {'ra': '14h59m07.6s', 'dec': '+71d40m20s', 'coeffs': [13.999, -0.556, -0.326]},
    '3C348': {'ra': '16h51m08.1s', 'dec': '+04d59m31s', 'coeffs': [29.107, -0.697, -0.329]},
    '3C353': {'ra': '17h20m28.1s', 'dec': '-00d58m47s', 'coeffs': [85.831, -0.680, -0.303]},
    '3C380': {'ra': '18h29m31.8s', 'dec': '+48d44m46s', 'coeffs': [40.601, -0.560, -0.366]},
    '3C409': {'ra': '20h14m27.3s', 'dec': '+23d34m58s', 'coeffs': [32.384, -0.664, -0.316]},
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

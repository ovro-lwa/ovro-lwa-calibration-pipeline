# imaging.py
import os
import sys
import logging
from astropy.coordinates import AltAz, SkyCoord
import astropy.units as u
import warnings
import re # <-- Import re for parsing

# Import pipeline utilities and config
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import pipeline_utils
import pipeline_config as config

# Initialize logger
logger = pipeline_utils.get_logger('Imaging')

# ==============================================================================
# === Helper Functions ===
# ==============================================================================

def run_wsclean(ms_path, image_name_prefix, specific_params):
    """Constructs and executes the WSClean command with a specific environment."""
    
    # Temporarily set the environment variable for this specific command
    original_threads = os.environ.get("OPENBLAS_NUM_THREADS")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    try:
        params = config.WSCLEAN_BASE_PARAMS.copy()
        params.update(specific_params)
        cmd_parts = [config.WSCLEAN_PATH]
        
        for key, value in params.items():
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f"-{key}")
            elif value is not None:
                cmd_parts.append(f"-{key}")
                # MODIFICATION: Ensure channel-range is split into two arguments
                if key == 'channel-range':
                    cmd_parts.extend(str(value).split())
                else:
                    cmd_parts.extend(str(value).split())
                
        cmd_parts.extend(['-name', image_name_prefix, ms_path])
        command = cmd_parts
        success = pipeline_utils.run_command(command, task_name=f"WSClean ({os.path.basename(image_name_prefix)})", logger=logger)
    
    finally:
        # Restore the original environment variable to avoid side effects
        if original_threads is None:
            if "OPENBLAS_NUM_THREADS" in os.environ:
                del os.environ["OPENBLAS_NUM_THREADS"]
        else:
            os.environ["OPENBLAS_NUM_THREADS"] = original_threads
            
    return success

def change_phase_center(ms_path, coordinates, source_name="Target"):
    """Uses chgcentre to rotate the phase center of the MS."""
    if not hasattr(coordinates, 'to_string'):
        logger.error("Invalid coordinate format. Must be Astropy SkyCoord object.")
        return False
    coord_str_hmsdms = coordinates.to_string('hmsdms')
    parts = coord_str_hmsdms.split()
    if len(parts) >= 2:
        ra_arg = parts[0]
        dec_arg = parts[1]
    else:
        logger.error(f"Could not parse coordinates string from SkyCoord: {coord_str_hmsdms}")
        return False
    logger.info(f"Rotating phase center to {source_name} (RA: {ra_arg}, Dec: {dec_arg})...")
    command = [config.CHGCENTRE_PATH, ms_path, ra_arg, dec_arg]
    success = pipeline_utils.run_command(command, task_name=f"chgcentre ({source_name})", logger=logger)
    return success

# --- NEW DYNAMIC CHANNEL RANGE FUNCTION ---
def _determine_channel_ranges(context):
    """
    Dynamically determines channel ranges for 3-band imaging
    based on the sub-bands actually present and the level of channel averaging.
    """
    logger.info("Dynamically determining channel ranges for 3-band imaging...")
    obs_info = context.get('obs_info', {})
    unique_freqs_detected = obs_info.get('unique_freqs_detected')
    
    # --- MODIFICATION: Dynamically detect averaging ---
    total_chans = obs_info.get('total_channels_concatenated')
    num_spw = obs_info.get('num_spw_concatenated')

    if not (unique_freqs_detected and total_chans and num_spw):
        logger.error("Missing context info (freqs, total_chans, or num_spw). Cannot determine channel ranges.")
        return {}
        
    if num_spw == 0: # Avoid division by zero
        logger.error("Number of SPWs is zero. Cannot determine channel ranges.")
        return {}

    # Calculate actual channels per sub-band (handles averaging)
    nchan_per_spw = total_chans // num_spw
    logger.info(f"Detected {nchan_per_spw} channels per sub-band (Total: {total_chans} / {num_spw} SPWs).")
    # --- END MODIFICATION ---

    # 1. Parse detected frequencies and sort them to get SPW order
    freq_mapping = []
    for freq_str in unique_freqs_detected:
        match = re.match(r'(\d+)MHz', freq_str)
        if match:
            freq_val = int(match.group(1))
            freq_mapping.append((freq_val, freq_str))
        else:
            logger.warning(f"Could not parse frequency string: {freq_str}")

    freq_mapping.sort(key=lambda x: x[0]) # Sort by frequency

    # 2. Create a map of {freq_in_mhz: spw_id}
    spw_map = {freq_val: spw_id for spw_id, (freq_val, freq_str) in enumerate(freq_mapping)}
    logger.debug(f"Dynamic SPW Map (FreqMHz: SPW_ID): {spw_map}")
    
    # 3. Find SPW IDs for each band (using corrected definitions)
    # --- FIX 3: Corrected frequency ranges to be non-overlapping ---
    # Band 1 (18-41 MHz): SPWs 1 (18MHz) to 5 (36MHz).
    spws_low = [spw_map[f] for f in spw_map.keys() if 18 <= f < 41]
    # Band 2 (41-64 MHz): SPWs 6 (41MHz) to 10 (59MHz).
    spws_mid = [spw_map[f] for f in spw_map.keys() if 41 <= f < 64]
    # Band 3 (64-82 MHz): SPWs 11 (64MHz) to 15 (82MHz).
    spws_high = [spw_map[f] for f in spw_map.keys() if 64 <= f <= 82]
    # --- END FIX 3 ---

    # 4. Calculate channel ranges
    channel_ranges = {}
    
    if spws_low:
        min_spw, max_spw = min(spws_low), max(spws_low)
        # Use your corrected (inclusive) channel logic
        start_chan = min_spw * nchan_per_spw 
        end_chan = (max_spw + 1) * nchan_per_spw
        channel_ranges["18-41MHz"] = (f"{start_chan} {end_chan}", len(spws_low))
    
    if spws_mid:
        min_spw, max_spw = min(spws_mid), max(spws_mid)
        start_chan = min_spw * nchan_per_spw
        end_chan = (max_spw + 1) * nchan_per_spw
        channel_ranges["41-64MHz"] = (f"{start_chan} {end_chan}", len(spws_mid))

    if spws_high:
        min_spw, max_spw = min(spws_high), max(spws_high)
        start_chan = min_spw * nchan_per_spw
        end_chan = (max_spw + 1) * nchan_per_spw
        channel_ranges["64-82MHz"] = (f"{start_chan} {end_chan}", len(spws_high))

    logger.info(f"Determined channel ranges (Range, N_SPW): {channel_ranges}")
    return channel_ranges

# ==============================================================================
# === Main Imaging Function ===
# ==============================================================================

def run_imaging(context):
    """
    Executes imaging steps, creating subdirectories for image products.
    """
    logger.info("Starting Imaging.")
    overall_success = True

    ms_path = context.get('concat_ms')
    if not ms_path or not os.path.exists(ms_path):
        logger.error(f"Input MS not found: {ms_path}. Imaging cannot proceed.")
        return False
        
    qa_dir = context['qa_dir']
    modeled_sources = context.get('model_sources', [])
    time_identifier = context.get('time_identifier', 'unknown_time')

    obs_info = context.get('obs_info', {})
    intervals_out = obs_info.get('num_integrations_detected')
    # Use the dynamically detected channel count per SPW
    channels_per_spw = obs_info.get('total_channels_concatenated', 192*16) // obs_info.get('num_spw_concatenated', 16)
    channels_out_total = channels_per_spw * obs_info.get('num_spw_concatenated', 16)
    logger.info(f"Total channels for spectrum imaging: {channels_out_total} ({channels_per_spw} chan/spw)")

    # --- FIX 1: Use number of SPWs for channels-out ---
    channels_out_spws = obs_info.get('num_spw_concatenated', 16)
    if channels_out_spws <= 0:
        logger.error("Number of SPWs is zero or invalid. Imaging cannot proceed.")
        return False
    # --- END FIX 1 ---

    if not intervals_out or not channels_out_total:
        logger.error("Could not determine integrations/sub-bands. Imaging cannot proceed.")
        return False
        
    logger.info(f"Using dynamically detected parameters: Intervals Out={intervals_out}, Channels Out (SPWs)={channels_out_spws}")

    # --- Create subdirectories for image products ---
    spectrum_dir = context['web_spec_dir']
    scintillation_dir = context['web_scint_dir']
    allsky_dir = context['web_allsky_dir']
    
    # These directories are already created by pipeline_utils.initialize_context
    logger.info(f"Image products will be saved in QA_website subdirectories.")

    if not modeled_sources:
        logger.info("No sources modeled. Skipping targeted imaging.")
    
    for source in modeled_sources:
        logger.info(f"--- Imaging Source: {source} ---")
        details = config.PRIMARY_SOURCES.get(source)
        if not details or 'skycoord' not in details: 
            logger.warning(f"Coordinates not found for {source}. Skipping.")
            continue
        coords = details['skycoord']

        if not change_phase_center(ms_path, coords, source):
            logger.error(f"Failed to change phase center to {source}. Skipping.")
            overall_success = False
            continue

        logger.info(f"Imaging {source} spectrum...")
        image_prefix_spec = os.path.join(spectrum_dir, f"{source}_spectrum_{time_identifier}")
        params_spec = config.WSCLEAN_SPECTRUM_PARAMS.copy()
        # --- FIX 1: Set channels-out to number of sub-bands (e.g., 16) ---
        params_spec['channels-out'] = channels_out_spws
        
        if not run_wsclean(ms_path, image_prefix_spec, params_spec):
            logger.warning(f"WSClean (Spectrum) failed for {source}.")
            overall_success = False

        if intervals_out > 1:
            logger.info(f"Imaging {source} scintillation...")
            image_prefix_scint = os.path.join(scintillation_dir, f"{source}_scintillation_{time_identifier}")
            params_scint = config.WSCLEAN_SCINTILLATION_PARAMS.copy()
            params_scint['intervals-out'] = intervals_out
            # --- FIX 1: Set channels-out to number of sub-bands (e.g., 16) ---
            params_scint['channels-out'] = channels_out_spws
            if not run_wsclean(ms_path, image_prefix_scint, params_scint):
                logger.warning(f"WSClean (Scintillation) failed for {source}.")
                overall_success = False
        else:
            logger.info(f"Skipping scintillation imaging for {source} as only one integration was found.")

    logger.info("--- Starting Full Sky Imaging at Zenith ---")
    obs_mid_time = obs_info.get('obs_mid_time')
    if obs_mid_time:
        zenith_coord = SkyCoord(alt=90*u.deg, az=0*u.deg, frame='altaz',
                                obstime=obs_mid_time, location=config.OVRO_LOCATION)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="astropy._erfa")
            zenith_icrs = zenith_coord.transform_to('icrs')

        if change_phase_center(ms_path, zenith_icrs, "Zenith"):
            logger.info("Running WSClean (Full Sky) in 3 bands...")
            
            # --- MODIFIED: Use dynamic channel ranges ---
            try:
                channel_ranges = _determine_channel_ranges(context)
            except Exception as e:
                logger.error(f"Failed to determine dynamic channel ranges: {e}. Skipping all-sky imaging.")
                overall_success = False
                channel_ranges = {} # Ensure it's an empty dict to skip loops
            
            # 1. Low Band (18-41 MHz)
            if "18-41MHz" in channel_ranges:
                ch_range, n_spw = channel_ranges["18-41MHz"]
                image_prefix_low = os.path.join(allsky_dir, f"FullSky_zenith_18-41MHz_{time_identifier}")
                params_low = config.WSCLEAN_FULLSKY_PARAMS.copy()
                # --- FIX 2: Removed .pop('taper-inner-tukey', None) ---
                params_low['channel-range'] = ch_range
                params_low['channels-out'] = n_spw # Add channels-out
                if not run_wsclean(ms_path, image_prefix_low, params_low):
                    logger.warning("WSClean (Full Sky, 18-41 MHz) failed.")
                    overall_success = False
            else:
                logger.warning("Skipping 18-41 MHz all-sky image: No sub-bands found in this range.")

            # 2. Mid Band (41-64 MHz)
            if "41-64MHz" in channel_ranges:
                ch_range, n_spw = channel_ranges["41-64MHz"]
                image_prefix_mid = os.path.join(allsky_dir, f"FullSky_zenith_41-64MHz_{time_identifier}")
                params_mid = config.WSCLEAN_FULLSKY_PARAMS.copy()
                # --- FIX 2: Removed .pop('taper-inner-tukey', None) ---
                params_mid['channel-range'] = ch_range
                params_mid['channels-out'] = n_spw # Add channels-out
                if not run_wsclean(ms_path, image_prefix_mid, params_mid):
                    logger.warning("WSClean (Full Sky, 41-64 MHz) failed.")
                    overall_success = False
            else:
                logger.warning("Skipping 41-64 MHz all-sky image: No sub-bands found in this range.")

            # 3. High Band (64-82 MHz)
            if "64-82MHz" in channel_ranges:
                ch_range, n_spw = channel_ranges["64-82MHz"]
                image_prefix_high = os.path.join(allsky_dir, f"FullSky_zenith_64-82MHz_{time_identifier}")
                params_high = config.WSCLEAN_FULLSKY_PARAMS.copy()
                # --- FIX 2: Removed .pop('taper-inner-tukey', None) ---
                params_high['channel-range'] = ch_range
                params_high['channels-out'] = n_spw # Add channels-out
                if not run_wsclean(ms_path, image_prefix_high, params_high):
                    logger.warning("WSClean (Full Sky, 64-82 MHz) failed.")
                    overall_success = False
            else:
                logger.warning("Skipping 64-82 MHz all-sky image: No sub-bands found in this range.")
            
            # --- End of modification ---

        else:
            logger.error("Failed to change phase center to Zenith. Skipping full sky imaging.")
            overall_success = False
    else:
        logger.error("Midpoint time missing. Skipping full sky imaging.")
        overall_success = False

    logger.info("Imaging step finished.")
    return overall_success

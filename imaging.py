# imaging.py
import os
import sys
import logging
from astropy.coordinates import AltAz, SkyCoord
import astropy.units as u
import warnings

# Import pipeline utilities and config
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import pipeline_utils
import pipeline_config as config

# Initialize logger
logger = pipeline_utils.get_logger('Imaging')

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
                cmd_parts.extend(str(value).split())
                
        cmd_parts.extend(['-name', image_name_prefix, ms_path])
        command = cmd_parts
        success = pipeline_utils.run_command(command, task_name=f"WSClean ({os.path.basename(image_name_prefix)})")
    
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
    success = pipeline_utils.run_command(command, task_name=f"chgcentre ({source_name})")
    return success

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
    channels_out = obs_info.get('num_subbands_detected')

    if not intervals_out or not channels_out:
        logger.error("Could not determine integrations/sub-bands. Imaging cannot proceed.")
        return False
        
    logger.info(f"Using dynamically detected parameters: Channels Out={channels_out}, Intervals Out={intervals_out}")

    # --- Create subdirectories for image products ---
    spectrum_dir = os.path.join(qa_dir, 'spectrum_images')
    scintillation_dir = os.path.join(qa_dir, 'scintillation_images')
    os.makedirs(spectrum_dir, exist_ok=True)
    os.makedirs(scintillation_dir, exist_ok=True)
    logger.info(f"Image products will be saved in subdirectories within {qa_dir}")

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
        params_spec['channels-out'] = channels_out
        
        if not run_wsclean(ms_path, image_prefix_spec, params_spec):
            logger.warning(f"WSClean (Spectrum) failed for {source}.")
            overall_success = False

        if intervals_out > 1:
            logger.info(f"Imaging {source} scintillation...")
            image_prefix_scint = os.path.join(scintillation_dir, f"{source}_scintillation_{time_identifier}")
            params_scint = config.WSCLEAN_SCINTILLATION_PARAMS.copy()
            params_scint['intervals-out'] = intervals_out
            params_scint['channels-out'] = channels_out
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
            logger.info("Running WSClean (Full Sky)...")
            # Create a single MFS image in the main QA directory
            image_prefix_fullsky = os.path.join(qa_dir, f"FullSky_zenith_{time_identifier}")
            params_fullsky = config.WSCLEAN_FULLSKY_PARAMS.copy()
            # This ensures a single MFS image is created
            params_fullsky['join-channels'] = True 
            
            if not run_wsclean(ms_path, image_prefix_fullsky, params_fullsky):
                logger.warning("WSClean (Full Sky) failed.")
                overall_success = False
        else:
            logger.error("Failed to change phase center to Zenith. Skipping full sky imaging.")
            overall_success = False
    else:
        logger.error("Midpoint time missing. Skipping full sky imaging.")
        overall_success = False

    logger.info("Imaging step finished.")
    return overall_success


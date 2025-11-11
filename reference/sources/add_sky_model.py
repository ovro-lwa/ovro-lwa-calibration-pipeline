# add_sky_model.py

import os
import sys
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
import shutil
import logging
from tqdm import tqdm
import warnings

# Astropy imports
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
import astropy.units as u

# Import pipeline utilities and config
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import pipeline_utils
import pipeline_config as config

# Initialize logger
logger = pipeline_utils.get_logger('SkyModel')

# CASA Imports handled by pipeline_utils
CASA_AVAILABLE = pipeline_utils.CASA_AVAILABLE
CASA_IMPORTS = pipeline_utils.CASA_IMPORTS

# ==============================================================================
# === HELPER FUNCTIONS (Beam and Secondary Source Models) ===
# ==============================================================================

def get_sh12_flux_jy(source_coeffs, freq_hz):
    """Calculates flux density for SH12 secondary sources."""
    freq_mhz = np.atleast_1d(freq_hz) / 1e6
    x = np.log10(freq_mhz / 150.0)
    # Per SH12, S = S_150 * 10^(c1*x + c2*x^2 + ...), where S_150 is coeffs[0]
    log_poly_term = sum(source_coeffs[i] * (x**i) for i in range(1, len(source_coeffs)))
    return source_coeffs[0] * (10**log_poly_term)

def load_beam_interpolator(h5_path):
    """Loads the beam model from HDF5 and creates a 3D interpolator."""
    logger.info(f"Loading and interpolating beam model from {h5_path}...")
    if not os.path.exists(h5_path):
        logger.error(f"Beam H5 file not found: {h5_path}")
        raise FileNotFoundError(f"Beam H5 file not found: {h5_path}")
    try:
        with h5py.File(h5_path, 'r') as hf:
            fq_orig = hf['freq_Hz'][:]
            th_orig = hf['theta_pts'][:] 
            ph_orig = hf['phi_pts'][:]   
            Exth = hf['X_pol_Efields/etheta'][:]
            Exph = hf['X_pol_Efields/ephi'][:]
            Eyth = hf['Y_pol_Efields/etheta'][:]
            Eyph = hf['Y_pol_Efields/ephi'][:]
            
        PbX = np.abs(Exth)**2 + np.abs(Exph)**2
        PbY = np.abs(Eyth)**2 + np.abs(Eyph)**2
        GI_unnormalized = PbX + PbY
        
        fq_s_idx, th_s_idx, ph_s_idx = np.argsort(fq_orig), np.argsort(th_orig), np.argsort(ph_orig)
        fq_s, th_s, ph_s = fq_orig[fq_s_idx], th_orig[th_s_idx], ph_orig[ph_s_idx]
        
        GI_unnormalized_sorted = GI_unnormalized[fq_s_idx,:,:][:,th_s_idx,:][:,:,ph_s_idx]
        
        zenith_theta_idx = np.argmin(np.abs(th_s))
        GI_unnorm_zenith_gain_vs_freq = GI_unnormalized_sorted[:, zenith_theta_idx, 0]
        GI_unnorm_zenith_gain_vs_freq[GI_unnorm_zenith_gain_vs_freq == 0] = 1e-9
        
        GI_normalized = GI_unnormalized_sorted / GI_unnorm_zenith_gain_vs_freq[:, np.newaxis, np.newaxis]
        
        interp = RegularGridInterpolator(
            (fq_s, th_s, ph_s), 
            GI_normalized, 
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        logger.info("Beam interpolator created successfully.")
        return interp
    except Exception as e:
        logger.exception(f"Error processing beam file: {h5_path}")
        raise

# ==============================================================================
# === Main Logic ===
# ==============================================================================

def main(context, single_source_override=None, apply_model=True):
    """
    Calculates and optionally applies a pre-computed sky model to the MS.
    If apply_model is False, it only calculates the model and populates the
    context['model_sources'] list without running the 'ft' task.
    """
    if not CASA_AVAILABLE:
        logger.error("FATAL: CASA tools not available. Cannot run Add Sky Model step.")
        return False

    logger.info("Starting Add Sky Model step (using pre-computed models).")

    msfile_path = context.get('concat_ms')
    if not msfile_path or not os.path.exists(msfile_path):
        logger.error(f"FATAL: MS not found. Path: {msfile_path}")
        return False

    output_dir = context.get('qa_dir')

    # Initialize CASA tools
    try:
        me_factory = CASA_IMPORTS.get('measures')
        msmd_factory = CASA_IMPORTS.get('msmetadata')
        cl_factory = CASA_IMPORTS.get('componentlist')
        ft = CASA_IMPORTS.get('ft') 

        if not all([me_factory, msmd_factory, cl_factory, ft]):
             raise ImportError("One or more critical CASA tools/tasks are missing.")

        me = me_factory()
        msmd = msmd_factory()
        cl = cl_factory()
        
        qa_factory = CASA_IMPORTS.get('quantity') or CASA_IMPORTS.get('quanta')
        if qa_factory:
            qa = qa_factory()
        else:
            raise ImportError("Critical tool 'quantity' or 'quanta' not found.")
        
    except (KeyError, ImportError) as e:
        logger.exception(f"Failed to initialize CASA tool instances: {e}")
        return False

    logger.info(f"Loading reference flux models from {config.FLUX_MODEL_NPZ_PATH}")
    try:
        models = np.load(config.FLUX_MODEL_NPZ_PATH)
        ref_freqs_hz = models['frequencies_hz']
    except (FileNotFoundError, Exception) as e:
        logger.error(f"FATAL: Failed to load reference flux models: {e}")
        return False

    try:
        msmd.open(msfile_path)
        location = config.OVRO_LOCATION
        
        try:
            spw_ids = np.array(msmd.datadescids())
            if not spw_ids.size > 0: raise ValueError("No Data Description IDs found.")
        except Exception:
            logger.warning("Failed to retrieve Data Description IDs. Falling back to direct SPW IDs.")
            spw_ids = np.array(msmd.spwids())

        if not spw_ids.size > 0:
            logger.error("CRITICAL: No valid SPW IDs or Data Description IDs found in the MS.")
            msmd.done()
            return False

        all_chans_list = [msmd.chanfreqs(int(s)) for s in spw_ids if msmd.nchan(int(s)) > 0]
        if not all_chans_list:
             logger.error("No valid channels with frequencies found in the MS.")
             msmd.done()
             return False
        all_chans_hz = np.unique(np.concatenate(all_chans_list))

        all_times = msmd.timesforfield(0)
        if all_times.size == 0:
            logger.error("No time samples found for FIELD_ID 0.")
            msmd.done()
            return False

        midpoint_time_sec = (all_times[0] + all_times[-1]) / 2.0
        model_time_sec = all_times[np.argmin(np.abs(all_times - midpoint_time_sec))]
        model_time_mjd = model_time_sec / 86400.0
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            curr_time = Time(model_time_mjd, format='mjd')
            epoch = curr_time.decimalyear
        
    except Exception as e:
        logger.exception("Failed to read essential metadata from MS.")
        return False
    finally:
        if msmd: 
            try: msmd.done()
            except Exception: pass 

    logger.info(f"Calculating model for midpoint time: {curr_time.iso} (Epoch {epoch:.2f})")
    
    try:
        beam_interp = load_beam_interpolator(config.BEAM_MODEL_H5)
    except Exception as e:
        logger.exception(f"Failed to load or interpolate beam model: {e}")
        return False

    altaz_frame = AltAz(obstime=curr_time, location=config.OVRO_LOCATION)
    
    try:
        obs_pos_me = me.position('ITRF', qa.quantity(location.x.to_value(u.m), 'm'), qa.quantity(location.y.to_value(u.m), 'm'), qa.quantity(location.z.to_value(u.m), 'm'))
        me.doframe(obs_pos_me)
    except Exception as e:
        logger.exception("Failed to set up CASA measures frame (me.position/doframe).")
        return False

    comps_for_ft = []
    log_data_for_csv = []
    modeled_source_names = []
    
    date_str = curr_time.datetime.strftime('%Y%m%d')
    lst_str = context['obs_info']['lst_hour'] 
    csv_filename = os.path.join(output_dir, f"{date_str}_LST_{lst_str}_model_summary.csv")
    cl_filename = os.path.join(output_dir, f"{date_str}_LST_{lst_str}_model.cl")

    all_sources = {**config.PRIMARY_SOURCES, **config.SECONDARY_SOURCES}
    logger.info("--- Calculating Apparent Fluxes for All Sources ---")
    
    for src, details in tqdm(all_sources.items(), desc="Processing Sources", unit="source"):
        sky_coord = details.get('skycoord') or SkyCoord(details['ra'], details['dec'], frame='icrs', unit=(u.hourangle, u.deg))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            src_aa = sky_coord.transform_to(altaz_frame)

        if src_aa.alt.deg < config.ELEVATION_LIMIT_DEG:
            continue

        s_int_full = None
        is_primary = src in config.PRIMARY_SOURCES
        
        if is_primary:
            obs_indices = np.searchsorted(ref_freqs_hz, all_chans_hz)
            obs_indices = np.clip(obs_indices, 0, len(ref_freqs_hz) - 1)
            try:
                if src == 'CygA': s_int_full = models['cyga_flux_jy'][obs_indices]
                elif src == 'VirA': s_int_full = models['vira_flux_jy'][obs_indices]
                elif src == 'CasA':
                    s_ref = models['casa_flux_jy_ref'][obs_indices]
                    ref_epoch = models['reference_epoch'][0]
                    s_int_full = s_ref * (1 + -0.46 / 100.0)**(epoch - ref_epoch)
                elif src == 'TauA':
                    s_ref = models['taua_flux_jy_ref'][obs_indices]
                    ref_epoch = models.get('taua_reference_epoch', [1985.5])[0]
                    s_int_full = s_ref * (1 + -0.16 / 100.0)**(epoch - ref_epoch)
            except KeyError:
                logger.error(f"Missing model data for primary source {src}. Skipping.")
                continue
        else: 
            s_int_full = get_sh12_flux_jy(details['coeffs'], all_chans_hz)

        if s_int_full is not None:
            interp_coords = np.stack([all_chans_hz, np.full_like(all_chans_hz, src_aa.zen.rad), np.full_like(all_chans_hz, src_aa.az.rad)], axis=-1)
            gains_full = beam_interp(interp_coords)
            s_corr_full = s_int_full * gains_full
            
            log_data_for_csv.append({'src': src, 's_int': s_int_full, 's_corr': s_corr_full})

            if is_primary:
                if single_source_override and src != single_source_override: continue
                
                max_apparent_flux = np.max(s_corr_full)
                if max_apparent_flux >= config.FLUX_THRESHOLD_JY:
                    logger.info(f"  --> Including {src} in model. Max apparent flux: {max_apparent_flux:.0f} Jy.")
                    try:
                        direction_me = me.direction('ICRS', sky_coord.ra.to_string(unit=u.hourangle, sep='hms'), sky_coord.dec.to_string(unit=u.deg, sep='dms'))
                        comps_for_ft.append({'direction': direction_me, 'label': src, 'freqs_hz': all_chans_hz, 'fluxes_jy': s_corr_full})
                        modeled_source_names.append(src)
                    except Exception as e:
                        logger.error(f"Failed to create CASA direction object for {src}: {e}")
                        continue
                else:
                    logger.info(f"  --> Excluding {src}. Max apparent flux ({max_apparent_flux:.0f} Jy) below threshold.")

    if config.PRODUCE_SECONDARY_SOURCE_CSV and log_data_for_csv:
        logger.info(f"Writing model summary CSV: {os.path.basename(csv_filename)}")
        try:
            output_headers = ['Frequency_Hz'] + [item for data in log_data_for_csv for item in (f"{data['src']}_Intrinsic_Jy", f"{data['src']}_Apparent_Jy")]
            output_columns = [all_chans_hz] + [col for data in log_data_for_csv for col in (data['s_int'], data['s_corr'])]
            output_array = np.column_stack(output_columns)
            np.savetxt(csv_filename, output_array, header=','.join(output_headers), fmt='%.6e', delimiter=',')
        except Exception as e:
            logger.error(f"Failed to write CSV log file: {e}")
    elif not config.PRODUCE_SECONDARY_SOURCE_CSV:
        logger.info("Skipping model summary CSV generation based on pipeline_config setting.")

    if comps_for_ft:
        logger.info(f"Generating Component List (CL) file: {os.path.basename(cl_filename)}")
        if os.path.exists(cl_filename):
            shutil.rmtree(cl_filename, ignore_errors=True)
        
        try:
            try: cl.close()
            except Exception: pass 

            for comp_data in comps_for_ft:
                cl.addcomponent(label=comp_data['label'], dir=comp_data['direction'], flux=1.0, fluxunit='Jy', freq="100MHz", shape="point")
                cl.setspectrum(which=cl.length()-1, type='tabular', tabularfreqs=comp_data['freqs_hz'], tabularflux=comp_data['fluxes_jy'], tabularframe='LSRK')
            
            cl.rename(filename=cl_filename)
            cl.close()
            
            if apply_model:
                logger.info("Applying model using CASA task 'ft' (usescratch=True)...")
                ft(vis=msfile_path, complist=cl_filename, usescratch=True)
                logger.info("CASA ft task completed.")
            else:
                logger.info("apply_model=False. Skipping application of model to MS.")

            context['model_sources'] = modeled_source_names
            logger.info(f"Updated context with modeled sources: {modeled_source_names}")

        except Exception as e:
            logger.exception("Failed to create component list.")
            return False
    else:
        logger.warning("No primary calibrators met criteria. Model column remains empty.")
        context['model_sources'] = []
        return True if not apply_model else False

    logger.info("Add Sky Model step finished successfully.")
    return True

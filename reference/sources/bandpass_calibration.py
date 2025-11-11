# bandpass_calibration.py
import os
import sys
import numpy as np
import logging
import shutil

# Astropy imports for phase center calculation
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord

# Import pipeline utilities and config
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import pipeline_utils
import pipeline_config as config

# Initialize logger
logger = pipeline_utils.get_logger('BandpassCal')

# CASA Imports handled by pipeline_utils
CASA_AVAILABLE = pipeline_utils.CASA_AVAILABLE
CASA_IMPORTS = pipeline_utils.CASA_IMPORTS

# --- Imports for Bandpass Flagging and Plotting ---
try:
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend for saving files
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import casacore.tables as pt
    TABLE_TOOLS_AVAILABLE = True
except ImportError:
    TABLE_TOOLS_AVAILABLE = False

# --- MODIFIED: Removed scipy.ndimage and stats, Added medfilt ---
try:
    from scipy.signal import medfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy.signal.medfilt not found. Template smoothing will be disabled.")


# ==============================================================================
# === HELPER: Phase Center Correction ===
# ==============================================================================
#
# === FUNCTION _ensure_common_phase_center REMOVED ===
# This logic has been moved to pipeline_utils.set_phase_center
# and is now called from data_preparation.py
#

# ==============================================================================
# === Bandpass Flagging and QA (INTEGRATED - NEW 6-STEP STRATEGY) ===
# ==============================================================================

# --- REPLACED: load_bandpass_data from test_bp_flagging.py ---
def load_bandpass_data(caltable_path, context=None):
    """
    MODIFIED: Also saves total_chans and num_spw to the context
    to help detect channel averaging dynamically.
    """
    if not TABLE_TOOLS_AVAILABLE:
        logger.error("FATAL: casacore.tables not found. Cannot load bandpass data for flagging.")
        return None, None, None, None
    logger.info("Robustly loading bandpass data using manual iteration...")
    try:
        with pt.table(caltable_path, ack=False) as t, \
             pt.table(os.path.join(caltable_path, 'SPECTRAL_WINDOW'), ack=False) as t_spw:
            antennas = sorted(np.unique(t.getcol('ANTENNA1')))
            num_spw = t_spw.nrows()
            spw_ids = np.arange(num_spw)
            all_chan_freqs = t_spw.getcol('CHAN_FREQ')
            spw_freqs = {spw_id: freqs for spw_id, freqs in zip(spw_ids, all_chan_freqs)}
            spw_chans = {spw_id: len(freqs) for spw_id, freqs in spw_freqs.items()}
            total_chans = sum(spw_chans.values())
            
            # --- NEW: Save channel info to context ---
            if context and 'obs_info' in context:
                context['obs_info']['total_channels_concatenated'] = total_chans
                context['obs_info']['num_spw_concatenated'] = num_spw
                logger.info(f"Saved to context: total_channels={total_chans}, num_spw={num_spw}")
            # --- END NEW ---
                
            logger.info(f"Table structure: {len(antennas)} antennas, {len(spw_chans)} SPWs, {total_chans} total channels.")
            full_freq_axis = np.concatenate([spw_freqs[spw_id] for spw_id in sorted(spw_freqs.keys())])
            channel_offset = {}
            offset = 0
            for spw_id in sorted(spw_chans.keys()):
                channel_offset[spw_id] = offset
                offset += spw_chans[spw_id]
            num_pol = 2 
            full_gains = np.zeros((total_chans, len(antennas), num_pol), dtype=np.complex64)
            full_flags = np.zeros((total_chans, len(antennas), num_pol), dtype=bool)
            logger.info(f"Reading {t.nrows()} rows from main table...")
            # columns_to_read = ["ANTENNA1", "SPECTRAL_WINDOW_ID", "CPARAM", "FLAG"] # From test script, not needed for getcell
            for rownr in range(t.nrows()):
                ant_id = t.getcell("ANTENNA1", rownr)
                spw_id = t.getcell("SPECTRAL_WINDOW_ID", rownr)
                start_chan, end_chan = channel_offset[spw_id], channel_offset[spw_id] + spw_chans[spw_id]
                full_gains[start_chan:end_chan, ant_id, :] = t.getcell("CPARAM", rownr)
                full_flags[start_chan:end_chan, ant_id, :] = t.getcell("FLAG", rownr)
        logger.info(f"Successfully reconstructed data cubes with shape {full_gains.shape}")
        return full_gains, full_flags, full_freq_axis, antennas
    except Exception as e:
        logger.error("Failed during robust data loading for flagging.", exc_info=True)
        return None, None, None, None

# --- REPLACED: write_bandpass_flags from test_bp_flagging.py ---
def write_bandpass_flags(caltable_path, final_flags):
    # Modified to only write flags, not gains
    logger.info("Robustly writing modified flags back to table...")
    try:
        with pt.table(caltable_path, readonly=False, ack=False) as t, \
             pt.table(os.path.join(caltable_path, 'SPECTRAL_WINDOW'), ack=False) as t_spw:
            num_spw = t_spw.nrows()
            spw_ids = np.arange(num_spw)
            all_num_chan = t_spw.getcol('NUM_CHAN')
            spw_chans = {spw_id: nchan for spw_id, nchan in zip(spw_ids, all_num_chan)}
            channel_offset = {}
            offset = 0
            for spw_id in sorted(spw_chans.keys()):
                channel_offset[spw_id] = offset
                offset += spw_chans[spw_id]
            logger.info(f"Writing flags to {t.nrows()} rows in main table...")
            for rownr in range(t.nrows()):
                ant_id = t.getcell("ANTENNA1", rownr)
                spw_id = t.getcell("SPECTRAL_WINDOW_ID", rownr)
                start_chan, end_chan = channel_offset[spw_id], channel_offset[spw_id] + spw_chans[spw_id]
                flags_to_write = final_flags[start_chan:end_chan, ant_id, :]
                t.putcell("FLAG", rownr, flags_to_write)
        logger.info("Successfully wrote flags to table.")
        return True
    except Exception as e:
        logger.error("Failed during robust flag writing.", exc_info=True)
        return False

# --- REMOVED old helper functions ---
# _determine_rfi_cutoff_freq
# _flag_outlier_antennas
# _flag_and_interpolate_rfi_channels

def _consolidate_ranges(indices):
    """Helper for logging flag ranges."""
    if len(indices) == 0: return []
    ranges, start, end = [], indices[0], indices[0]
    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            end = indices[i]
        else:
            ranges.append((start, end))
            start = end = indices[i]
    ranges.append((start, end))
    return ranges

# --- REPLACED: flag_data_cubes from test_bp_flagging.py ---
# --- (and adapted to use pipeline_config) ---
def flag_data_cubes(gains, flags, freqs_hz):
    new_flags = flags.copy()
    amplitudes = np.abs(gains) # Raw amplitudes, used for Steps 2-6
    n_chan, n_ant, n_pol = amplitudes.shape
    freqs_mhz = freqs_hz / 1e6
    channel_indices = np.arange(n_chan)

    # --- NEW Step 0: Normalize Amplitudes for Scatter Flagging ---
    logger.info(f"--- Step 0: Normalizing amplitudes for scatter flagging ---")
    norm_indices = np.where((freqs_mhz >= config.NORMALIZATION_FREQ_RANGE_MHZ[0]) & 
                            (freqs_mhz <= config.NORMALIZATION_FREQ_RANGE_MHZ[1]))[0]
    median_gains = np.ones((n_ant, n_pol)) # Fallback is 1.0

    if len(norm_indices) > 10:
        for i in range(n_ant):
            for j in range(n_pol):
                # Use initial flags ('flags') to get a stable median
                amps_in_range = amplitudes[norm_indices, i, j]
                flags_in_range = flags[norm_indices, i, j]
                unflagged_norm_amps = amps_in_range[~flags_in_range]
                
                if len(unflagged_norm_amps) > 10:
                    median_val = np.median(unflagged_norm_amps)
                    if median_val > 1e-6: # Avoid division by zero
                        median_gains[i, j] = median_val
                    else:
                        logger.warning(f"Median gain for Ant {i} Pol {j} is near zero. Using 1.0 for normalization.")
                else:
                    logger.warning(f"Not enough clean data for Ant {i} Pol {j} to normalize. Using 1.0.")
        # Create normalized amplitudes array (n_chan, n_ant, n_pol)
        normalized_amplitudes = amplitudes / median_gains[np.newaxis, :, :]
    else:
        logger.warning("Not enough channels in normalization range. Skipping normalization.")
        normalized_amplitudes = amplitudes.copy() # Use unnormalized


    # --- Step 1: Per-channel RFI flagging based on scatter (MODIFIED) ---
    logger.info(f"--- Step 1: Flagging RFI channels based on scatter (using NORMALIZED data) ---")
    stds_per_channel = np.zeros(n_chan)
    scatter_flags_per_channel = np.zeros(n_chan, dtype=bool) # Track channels flagged here

    for i in range(n_chan):
        # Calculate std dev across all initially unflagged antennas and pols
        # *** USE NORMALIZED AMPLITUDES ***
        unflagged_amps = normalized_amplitudes[i, ~flags[i, :, :]] 
        
        if len(unflagged_amps) > 1:
            stds_per_channel[i] = np.std(unflagged_amps)

    clean_indices_step1 = np.where((freqs_mhz >= config.CHANNEL_SCATTER_CLEAN_FREQ_RANGE_MHZ[0]) & 
                                   (freqs_mhz <= config.CHANNEL_SCATTER_CLEAN_FREQ_RANGE_MHZ[1]))[0]
    
    if len(clean_indices_step1) > 10:
        # Get median scatter of *normalized* data
        median_clean_scatter = np.median(stds_per_channel[clean_indices_step1])
        if median_clean_scatter > 0:
            threshold = median_clean_scatter * config.CHANNEL_SCATTER_THRESHOLD_MULTIPLIER
            logger.info(f"Median clean-band (normalized) scatter: {median_clean_scatter:.4g}. RFI threshold: {threshold:.4g}")
            
            bad_channel_indices = np.where(stds_per_channel > threshold)[0]
            
            if len(bad_channel_indices) > 0:
                logger.info(f"Flagging {len(bad_channel_indices)} channels for all antennas due to high scatter.")
                scatter_flags_per_channel[bad_channel_indices] = True
                new_flags[bad_channel_indices, :, :] = True # Apply flags immediately
            else:
                logger.info("No high-scatter RFI channels found.")
        else:
            logger.warning("Median clean scatter is zero. Skipping scatter-based RFI flagging.")
    else:
        logger.warning("Not enough data in clean range to perform scatter-based RFI flagging.")

    # --- Step 2: Create robust template (UNCHANGED) ---
    # Note: This step correctly uses the *original* 'amplitudes'
    logger.info(f"--- Step 2: Creating robust bandpass template ---")
    
    median_template = np.zeros(n_chan)
    valid_template_channels = ~scatter_flags_per_channel # Channels NOT flagged in step 1
    
    for i in channel_indices[valid_template_channels]:
        # Uses original amplitudes
        unflagged_amps = amplitudes[i, ~new_flags[i, :, :]] # Use current flags
        if len(unflagged_amps) > 0:
            median_template[i] = np.median(unflagged_amps)

    valid_indices = channel_indices[valid_template_channels]
    invalid_indices = channel_indices[scatter_flags_per_channel]

    if len(valid_indices) > 1 and len(invalid_indices) > 0:
         logger.info(f"Interpolating template over {len(invalid_indices)} scatter-flagged channels.")
         median_template[invalid_indices] = np.interp(invalid_indices, valid_indices, median_template[valid_indices])
         median_template[median_template <= 0] = np.min(median_template[median_template > 0]) if np.any(median_template > 0) else 1.0 
    elif len(valid_indices) <= 1:
        logger.warning("Cannot interpolate template - too few valid channels. Using non-interpolated template.")
        median_template[median_template <= 0] = 1.0
    else:
        logger.info("No scatter-flagged channels to interpolate over in template.")
        median_template[median_template <= 0] = 1.0

    if SCIPY_AVAILABLE and len(valid_indices) > config.TEMPLATE_SMOOTHING_KERNEL_SIZE:
         logger.info(f"Smoothing template with median filter (kernel={config.TEMPLATE_SMOOTHING_KERNEL_SIZE}).")
         interp_template = median_template.copy()
         flagged_template_indices = np.where(median_template <= 0)[0] 
         unflagged_template_indices = np.where(median_template > 0)[0]
         if len(flagged_template_indices) > 0 and len(unflagged_template_indices) > 1:
              interp_template[flagged_template_indices] = np.interp(flagged_template_indices, unflagged_template_indices, median_template[unflagged_template_indices])
         
         median_template = medfilt(interp_template, kernel_size=config.TEMPLATE_SMOOTHING_KERNEL_SIZE)
    elif not SCIPY_AVAILABLE:
        logger.warning("Scipy not available, skipping template smoothing.")
    else:
        logger.warning("Not enough valid channels to smooth template.")


    # --- Step 3: Flag deviations from the template (UNCHANGED) ---
    # Note: This step correctly uses the *original* 'amplitudes'
    logger.info(f"--- Step 3: Flagging per-antenna deviations from template (sigma={config.DEVIATION_FLAG_SIGMA}) ---")
    total_deviation_flags = 0
    
    residuals = np.abs(amplitudes - median_template[:, np.newaxis, np.newaxis])
    
    for i in range(n_ant):
        if np.all(new_flags[:, i, :]): continue 
        for j in range(n_pol):
            ant_residuals = residuals[:, i, j]
            ant_flags = new_flags[:, i, j]
            unflagged_indices = np.where(~ant_flags)[0]
            
            if len(unflagged_indices) > 1:
                q1_res, q3_res = np.percentile(ant_residuals[unflagged_indices], [25, 75])
                iqr_res = q3_res - q1_res
                robust_std_res = iqr_res / 1.349
                
                if robust_std_res > 0:
                    threshold = config.DEVIATION_FLAG_SIGMA * robust_std_res
                    is_deviant = ant_residuals > threshold
                    
                    flags_to_add = np.logical_or(ant_flags, is_deviant)
                    total_deviation_flags += np.sum(flags_to_add) - np.sum(ant_flags)
                    new_flags[:, i, j] = flags_to_add

    logger.info(f"Step 3 complete. Flagged {total_deviation_flags} solutions deviating from the template.")

    # --- Step 4: Flag whole-antenna gain outliers (UNCHANGED) ---
    # Note: This step correctly uses the *original* 'amplitudes'
    logger.info(f"--- Step 4: Flagging whole-antenna gain outliers (sigma={config.GAIN_OUTLIER_SIGMA}) ---")
    
    clean_indices_step4 = clean_indices_step1 # Use the same clean range
    flagged_ant_count = 0

    if len(clean_indices_step4) > 10:
        mean_ant_gain = np.zeros((n_ant, n_pol))
        
        for i in range(n_ant):
            for j in range(n_pol):
                # Uses original amplitudes
                clean_amps = amplitudes[clean_indices_step4, i, j][~new_flags[clean_indices_step4, i, j]]
                if len(clean_amps) > 0:
                    mean_ant_gain[i, j] = np.mean(clean_amps)
        
        for j in range(n_pol):
            gains_pol = mean_ant_gain[:, j]
            valid_gains = gains_pol[gains_pol > 0] 
            
            if len(valid_gains) < 3:
                logger.warning(f"Pol {j}: Not enough valid antennas to check for gain outliers.")
                continue

            median_gain = np.median(valid_gains)
            iqr = np.percentile(valid_gains, 75) - np.percentile(valid_gains, 25)
            robust_std = iqr / 1.349
            
            if robust_std > 0:
                threshold = config.GAIN_OUTLIER_SIGMA * robust_std
                lower_bound = median_gain - threshold
                upper_bound = median_gain + threshold
                
                bad_ant_indices = np.where((gains_pol > 0) & 
                                           ((gains_pol < lower_bound) | (gains_pol > upper_bound)))[0]
                
                if len(bad_ant_indices) > 0:
                    logger.info(f"Pol {j}: Flagging {len(bad_ant_indices)} gain-outlier antennas (e.g., Ant {bad_ant_indices[0]}).")
                    for ant_idx in bad_ant_indices:
                        new_flags[:, ant_idx, j] = True
                        flagged_ant_count += 1
    else:
        logger.warning("Not enough data in clean range to perform gain-outlier flagging.")
    logger.info(f"Step 4 complete. Flagged {flagged_ant_count} total antenna/pols for bad gain.")
    
    # --- Step 5: Iterative Sigma Clipping (UNCHANGED) ---
    # Note: This step correctly uses the *original* 'amplitudes'
    sigma_threshold = config.BP_ITERATIVE_OUTLIER_FLAG_SIGMA
    n_iterations = config.BP_ITERATIVE_OUTLIER_FLAG_ITERATIONS
    if sigma_threshold and sigma_threshold > 0:
        logger.info(f"--- Step 5: Starting iterative sigma-clipping ({n_iterations} iterations, sigma={sigma_threshold}) ---")
        
        total_new_flags = 0
        for iteration in range(n_iterations):
            flags_in_iter = 0
            for chan_idx in range(n_chan):
                if np.all(new_flags[chan_idx, :, :]):
                    continue
                for pol_idx in range(n_pol):
                    # Uses original amplitudes
                    amps_this_chan = amplitudes[chan_idx, :, pol_idx]
                    flags_this_chan = new_flags[chan_idx, :, pol_idx]
                    unflagged_amps = amps_this_chan[~flags_this_chan]
                    
                    if len(unflagged_amps) < 2: continue

                    mean_amp = np.mean(unflagged_amps)
                    std_amp = np.std(unflagged_amps)
                    if std_amp == 0: continue

                    is_outlier = np.abs(amps_this_chan - mean_amp) > sigma_threshold * std_amp
                    is_new_outlier = np.logical_and(is_outlier, ~flags_this_chan)
                    
                    flags_in_iter += np.sum(is_new_outlier)
                    new_flags[chan_idx, :, pol_idx][is_new_outlier] = True
            
            logger.info(f"  Iteration {iteration + 1}: Flagged an additional {flags_in_iter} outlier solutions.")
            total_new_flags += flags_in_iter
            if flags_in_iter == 0:
                logger.info("  No new outliers found. Stopping iteration early.")
                break
        logger.info(f"Step 5 (iterative clipping) complete. Total new outlier flags: {total_new_flags}")

    # --- Step 6: Channel quorum flagging (UNCHANGED) ---
    logger.info(f"--- Step 6: Applying channel quorum flagging (Threshold: {config.CHANNEL_QUORUM_THRESHOLD*100}%) ---")
    
    flags_per_ant = np.any(new_flags, axis=2) 
    flagged_ant_count_chan = np.sum(flags_per_ant, axis=1)
    quorum = int(config.CHANNEL_QUORUM_THRESHOLD * n_ant)
    channels_to_flag_all = np.where(flagged_ant_count_chan > quorum)[0]
    
    if len(channels_to_flag_all) > 0:
        logger.info(f"Flagging all antennas in {len(channels_to_flag_all)} channels due to >{quorum} antennas being flagged.")
        new_flags[channels_to_flag_all, :, :] = True
    else:
        logger.info("No channels met the quorum flagging threshold.")
    
    return new_flags


# --- REPLACED/ADAPTED: summarize_and_plot_flags from test_bp_flagging.py ---
# --- Adapted to use context for QA directory and filenames ---
def summarize_and_plot_flags(caltable_path, initial_flags, final_flags, freqs_hz, antennas, context):
    """Saves the flag summary plot to the QA directory."""
    logger.info("--- Generating Flagging Summary ---")
    newly_flagged = final_flags & ~initial_flags
    total_new_flags = np.sum(newly_flagged)
    logger.info(f"Applied a total of {total_new_flags} new flags to the bandpass table.")

    # Log text summary of flags (will go to file)
    freqs_mhz = freqs_hz / 1e6
    summary_lines = ["\n" + "="*50, "Newly Flagged Solutions Summary", "="*50]
    found_flags = False
    for ant_idx in antennas:
        ant_flags_per_chan = np.any(newly_flagged[:, ant_idx, :], axis=1)
        flagged_chan_indices = np.where(ant_flags_per_chan)[0]
        if len(flagged_chan_indices) > 0:
            found_flags = True
            summary_lines.append(f"\nAntenna {ant_idx}:")
            ranges = _consolidate_ranges(flagged_chan_indices)
            for start_idx, end_idx in ranges:
                if start_idx == end_idx:
                    summary_lines.append(f"  - Flagged at {freqs_mhz[start_idx]:.2f} MHz")
                else:
                    summary_lines.append(f"  - Flagged from {freqs_mhz[start_idx]:.2f} to {freqs_mhz[end_idx]:.2f} MHz")
    if not found_flags: summary_lines.append("No new flags were applied.")
    summary_lines.append("="*50 + "\n")
    summary_string = "\n".join(summary_lines)
    
    # --- MODIFICATION: Find console handler and suppress verbose log ---
    console_handler = None
    for handler in logging.getLogger('OVRO_Pipeline').handlers: # Get root logger
        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
            console_handler = handler
            break

    if console_handler:
        original_level = console_handler.level
        console_handler.setLevel(logging.WARNING) # Temporarily set console to WARNING
        logger.info(summary_string)               # This will ONLY go to the file
        console_handler.setLevel(original_level)  # Restore original level
    else:
        # Fallback if no console handler found (e.g., in testing)
        logger.debug(summary_string) # Log at debug as a fallback
    # --- END MODIFICATION ---

    if not PLOTTING_AVAILABLE:
        logger.warning("Matplotlib not found. Skipping generation of flag summary plot.")
        return

    try:
        qa_dir = context['qa_dir']
        time_identifier = context.get('time_identifier', 'unknown')
        plot_basename = f"QA_plot_bandpass_flag_summary_{time_identifier}.png"
        plot_filename = os.path.join(qa_dir, plot_basename)
        
        flag_matrix = np.any(newly_flagged, axis=2)
        plt.figure(figsize=(12, 8))
        plt.imshow(flag_matrix.T, aspect='auto', origin='lower',
                   cmap='binary', interpolation='none',
                   extent=[freqs_hz[0]/1e6, freqs_hz[-1]/1e6, antennas[0]-0.5, antennas[-1]+0.5])
        plt.title(f'Newly Flagged Bandpass Solutions ({os.path.basename(caltable_path)})')
        plt.ylabel('Antenna Correlator Number'); plt.xlabel('Frequency (MHz)')
        plt.grid(alpha=0.2)
        plt.savefig(plot_filename)
        logger.info(f"Flag summary plot saved to: {plot_filename}")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate flag summary plot: {e}", exc_info=True)


# --- REMOVED ---
# plot_all_antennas_comparison
# --- MODIFIED: generate_per_antenna_bpplots_pdf ---
def generate_per_antenna_bpplots_pdf(initial_gains, initial_flags, final_flags, freqs_hz, antennas, pdf_output_path, png_output_dir):
    """
    MODIFIED: Generates multi-page PDF and individual PNGs.
    Note: 'final_gains' is not used by this function, but
    'final_flags' is, to show what data is flagged.
    """
    logger.info("Generating multi-page PDF of per-antenna bandpass solutions...")
    if not PLOTTING_AVAILABLE: return
    
    freqs_mhz = freqs_hz / 1e6
    try:
        with PdfPages(pdf_output_path) as pdf:
            for ant_idx, ant_num in enumerate(antennas):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
                
                # --- Amplitude Plot ---
                for pol_idx, pol_name in enumerate(['X', 'Y']):
                    # Initial (using initial_gains and initial_flags)
                    amp_i = np.abs(initial_gains[:, ant_idx, pol_idx])
                    amp_i[initial_flags[:, ant_idx, pol_idx]] = np.nan
                    ax1.plot(freqs_mhz, amp_i, alpha=0.5, label=f'Initial Pol {pol_name}')
                    
                    # Final (using initial_gains but final_flags)
                    amp_f = np.abs(initial_gains[:, ant_idx, pol_idx])
                    amp_f[final_flags[:, ant_idx, pol_idx]] = np.nan # Show what is flagged
                    ax1.plot(freqs_mhz, amp_f, alpha=1.0, label=f'Final (Flagged) Pol {pol_name}')
                
                ax1.set_ylabel('Gain Amplitude'); ax1.grid(True, alpha=0.3)
                ax1.legend(); ax1.set_yscale('log')

                # --- Phase Plot ---
                for pol_idx, pol_name in enumerate(['X', 'Y']):
                    # Initial
                    phase_i = np.angle(initial_gains[:, ant_idx, pol_idx], deg=True)
                    phase_i[initial_flags[:, ant_idx, pol_idx]] = np.nan
                    ax2.plot(freqs_mhz, phase_i, '.', alpha=0.5, label=f'Initial Pol {pol_name}')
                    # Final
                    phase_f = np.angle(initial_gains[:, ant_idx, pol_idx], deg=True)
                    phase_f[final_flags[:, ant_idx, pol_idx]] = np.nan
                    ax2.plot(freqs_mhz, phase_f, '.', alpha=1.0, label=f'Final (Flagged) Pol {pol_name}')
                
                ax2.set_ylabel('Gain Phase (deg)'); ax2.set_xlabel('Frequency (MHz)')
                ax2.grid(True, alpha=0.3); ax2.set_ylim(-180, 180)

                fig.suptitle(f'Antenna {ant_num} Bandpass Solutions (Post-Flagging)')
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                # Save to PDF
                pdf.savefig(fig)
                
                # Save individual PNG to web directory
                if png_output_dir:
                    png_path = os.path.join(png_output_dir, f'bp_ant_{ant_num:03d}.png')
                    fig.savefig(png_path, dpi=90) # Lower DPI for web
                
                plt.close(fig)
        logger.info(f"Saved per-antenna diagnostic PDF to {os.path.basename(pdf_output_path)}")
        if png_output_dir:
            logger.info(f"Saved individual antenna PNGs to {os.path.basename(png_output_dir)}")
    except Exception as e:
        logger.error(f"Failed to generate per-antenna plots: {e}", exc_info=True)


# --- MODIFIED: run_bandpass_flagging ---
def run_bandpass_flagging(input_caltable_path, tables_dir, context):
    """Orchestrates the bandpass flagging process and generates diagnostic plots."""
    logger.info("--- Starting Bandpass Table Flagging and Refinement (NEW 6-Step Strategy) ---")

    output_caltable_path = f"{input_caltable_path}.flagged"
    
    if os.path.exists(output_caltable_path):
        logger.warning(f"Output path {output_caltable_path} already exists. Removing it.")
        shutil.rmtree(output_caltable_path)
    try:
        logger.info(f"Copying original table to {output_caltable_path}")
        pt.tablecopy(input_caltable_path, output_caltable_path)
    except Exception as e:
        logger.error(f"Failed to copy calibration table for flagging: {e}", exc_info=True)
        return None

    # --- MODIFIED: Pass context to load_bandpass_data ---
    initial_gains, initial_flags, freqs_hz, antennas = load_bandpass_data(output_caltable_path, context=context)
    if initial_flags is None:
        logger.error("Could not load data from bandpass table. Aborting flagging.")
        return None
        
    # --- MODIFIED: Call new flagging function ---
    final_flags = flag_data_cubes(initial_gains, initial_flags, freqs_hz)
    
    # --- MODIFIED: Call new write function ---
    if not write_bandpass_flags(output_caltable_path, final_flags):
        logger.error("Failed to write flagged data. Aborting.")
        return None
        
    # --- MODIFIED: Call new plotting function ---
    summarize_and_plot_flags(output_caltable_path, initial_flags, final_flags, freqs_hz, antennas, context)
    
    # --- MODIFIED: Generate New Diagnostic Plots ---
    if PLOTTING_AVAILABLE:
        # Get new directory paths from context
        pdf_dir = context['ood_dir']
        png_dir = context['web_bp_dir']
        time_identifier = context.get('time_identifier', 'unknown')

        # Plot 1: Per-antenna amp/phase in a multi-page PDF (Saved to OOD)
        pdf_path = os.path.join(pdf_dir, f'bp_solutions_per_antenna_{time_identifier}.pdf')
        
        # This function now saves PNGs to png_dir *and* the PDF to pdf_path
        generate_per_antenna_bpplots_pdf(
            initial_gains, initial_flags, final_flags, 
            freqs_hz, antennas, 
            pdf_output_path=pdf_path, 
            png_output_dir=png_dir
        )
    
    # --- REMOVED: Old plot_all_antennas_comparison call ---
    
    logger.info("Bandpass flagging process completed successfully.")
    return output_caltable_path

# ==============================================================================
# === Main Logic (Unchanged, but calls new flagging) ===
# ==============================================================================

def run_bandpass_calibration(context):
    """
    Performs bandpass (B) calibration, flags the result, and applies the solution to the MS.
    """
    logger.info("Starting Bandpass Calibration (gaintype=B).")

    if not CASA_AVAILABLE:
        logger.error("CASA environment not available. Cannot run calibration tasks.")
        return False

    ms_path = context.get('concat_ms')
    if not ms_path or not os.path.exists(ms_path):
        logger.error(f"MS not found: {ms_path}")
        return False

    # <<< NEW: Verification step for phase center >>>
    logger.info("Verifying MS phase center...")
    try:
        if not pipeline_utils.check_phase_center(ms_path, context):
            logger.error("MS phase center verification failed. Halting calibration.")
            # This check is critical.
            return False
        logger.info("Phase center verified successfully.")
    except Exception as e:
        logger.exception("An unexpected error occurred during phase center verification.")
        return False

    # <<< REMOVED: Call to _ensure_common_phase_center >>>
    # This is now done in data_preparation.py

    tables_dir = context['tables_dir']
    
    try:
        obs_date_str = context['obs_info']['obs_date']
        lst_hour_str = context['obs_info']['lst_hour']
    except KeyError as e:
        logger.error(f"Missing essential observation info in context: {e}. Cannot generate table names.")
        return False

    bp_table_name = f"calibration_{obs_date_str}_{lst_hour_str}.B"
    bp_table_path = os.path.join(tables_dir, bp_table_name)

    if os.path.exists(bp_table_path):
        logger.info(f"Removing existing bandpass table: {bp_table_path}")
        try:
            shutil.rmtree(bp_table_path)
        except Exception as e:
            logger.error(f"Failed to remove existing bandpass table: {e}")
            return False

    uvrange_str = pipeline_utils.determine_calibration_uv_range(context)
    refant = config.CAL_REFANT

    try:
        bandpass = CASA_IMPORTS.get('bandpass')
        if not bandpass: raise EnvironmentError("CASA task 'bandpass' not available.")
            
        bandpass(
            vis=ms_path, caltable=bp_table_path, bandtype='B', refant=refant,
            uvrange=uvrange_str, solint='inf', combine='obs,scan,field',
            minsnr=config.BANDPASS_MIN_SNR, gaintable=[]
        )
        logger.info("CASA bandpass task completed.")
    except Exception as e:
        logger.error(f"CASA task 'bandpass' failed: {e}", exc_info=True)
        return False
    
    # --- This function call now uses the new 6-step flagging logic ---
    final_bp_table_path = run_bandpass_flagging(bp_table_path, tables_dir, context)
    
    if final_bp_table_path and os.path.exists(final_bp_table_path):
        table_to_apply = final_bp_table_path
        logger.info(f"Using flagged bandpass table for application: {os.path.basename(table_to_apply)}")
        context['calibration_tables']['bandpass_flagged'] = table_to_apply
    else:
        logger.warning("Bandpass flagging failed or produced no output. Applying original, unflagged table.")
        table_to_apply = bp_table_path

    context['calibration_tables']['bandpass'] = table_to_apply

    logger.info("Applying bandpass calibration table to the MS (CORRECTED_DATA column)...")
    try:
        applycal = CASA_IMPORTS.get('applycal')
        if not applycal: raise EnvironmentError("CASA task 'applycal' not available.")

        applycal(
            vis=ms_path, gaintable=[table_to_apply],
            interp=['nearest'], calwt=False, flagbackup=True
        )
        logger.info("CASA applycal task completed successfully.")
    except Exception as e:
        logger.error(f"Failed to apply bandpass table: {e}", exc_info=True)
        return False

    logger.info("Bandpass Calibration step finished successfully.")
    return True

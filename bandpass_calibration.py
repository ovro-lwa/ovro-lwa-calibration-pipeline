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

try:
    import scipy.ndimage
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ==============================================================================
# === HELPER: Phase Center Correction ===
# ==============================================================================

#
# === FUNCTION _ensure_common_phase_center REMOVED ===
# This logic has been moved to pipeline_utils.set_phase_center
# and is now called from data_preparation.py
#

# ==============================================================================
# === Bandpass Flagging and QA (INTEGRATED) ===
# ==============================================================================

def load_bandpass_data(caltable_path):
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

def write_bandpass_data(caltable_path, final_gains, final_flags):
    logger.info("Robustly writing modified gains and flags back to table...")
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
            logger.info(f"Writing gains and flags to {t.nrows()} rows in main table...")
            for rownr in range(t.nrows()):
                ant_id = t.getcell("ANTENNA1", rownr)
                spw_id = t.getcell("SPECTRAL_WINDOW_ID", rownr)
                start_chan, end_chan = channel_offset[spw_id], channel_offset[spw_id] + spw_chans[spw_id]
                gains_to_write = final_gains[start_chan:end_chan, ant_id, :]
                flags_to_write = final_flags[start_chan:end_chan, ant_id, :]
                t.putcell("CPARAM", rownr, gains_to_write)
                t.putcell("FLAG", rownr, flags_to_write)
        logger.info("Successfully wrote gains and flags to table.")
        return True
    except Exception as e:
        logger.error("Failed during robust gain/flag writing.", exc_info=True)
        return False

def _determine_rfi_cutoff_freq(gains, flags, freqs_hz):
    logger.info("Dynamically determining low-frequency RFI cutoff...")
    amplitudes = np.abs(gains)
    stds_per_channel = np.zeros(gains.shape[0])
    for i in range(gains.shape[0]):
        amps_this_chan = amplitudes[i,:,:]
        flags_this_chan = flags[i,:,:]
        unflagged_amps = amps_this_chan[~flags_this_chan]
        if len(unflagged_amps) > 1:
            stds_per_channel[i] = np.std(unflagged_amps)
    freqs_mhz = freqs_hz / 1e6
    clean_indices = np.where((freqs_mhz >= config.RFI_CLIFF_CLEAN_FREQ_RANGE_MHZ[0]) & (freqs_mhz <= config.RFI_CLIFF_CLEAN_FREQ_RANGE_MHZ[1]))[0]
    if len(clean_indices) < 10:
        logger.warning("Not enough data in clean range to determine RFI cliff. No cutoff applied.")
        return 0.0
    clean_band_stds = stds_per_channel[clean_indices]
    mean_of_stds, std_of_stds = np.mean(clean_band_stds), np.std(clean_band_stds)
    if mean_of_stds == 0:
        logger.warning("Baseline scatter is zero. Cannot determine RFI cliff. No cutoff applied.")
        return 0.0
    threshold = mean_of_stds + config.RFI_CLIFF_THRESHOLD_SIGMA * std_of_stds
    logger.info(f"Clean band scatter: mean={mean_of_stds:.4g}, std={std_of_stds:.4g}. Detection threshold: {threshold:.4g}")
    search_start_freq_mhz = config.RFI_CLIFF_CLEAN_FREQ_RANGE_MHZ[0] - 5.0
    search_indices = np.where(freqs_mhz < search_start_freq_mhz)[0]
    for i in reversed(search_indices):
        if stds_per_channel[i] > threshold:
            window_size = config.RFI_CLIFF_WINDOW_CHANS
            start_idx = max(0, i - window_size + 1)
            median_in_window = np.median(stds_per_channel[start_idx:i+1])
            if median_in_window > threshold:
                cutoff_freq = freqs_mhz[i]
                logger.info(f"Sustained RFI cliff detected at {cutoff_freq:.2f} MHz.")
                return cutoff_freq
    logger.warning("No sustained RFI cliff detected. No low-frequency cutoff will be applied.")
    return 0.0

def _flag_outlier_antennas(gains, flags, freqs_hz, antennas):
    logger.info(f"Checking for outlier antennas based on mean gain in {config.OUTLIER_ANTENNA_FREQ_RANGE_MHZ} MHz band...")
    new_flags = flags.copy()
    amplitudes = np.abs(gains)
    freqs_mhz = freqs_hz / 1e6
    clean_indices = np.where((freqs_mhz >= config.OUTLIER_ANTENNA_FREQ_RANGE_MHZ[0]) & (freqs_mhz <= config.OUTLIER_ANTENNA_FREQ_RANGE_MHZ[1]))[0]
    if len(clean_indices) < 10:
        logger.warning("Not enough channels in clean range to check for outlier antennas.")
        return new_flags
    mean_ant_amps, ant_indices = [], []
    for ant_idx, ant_num in enumerate(antennas):
        ant_amps_clean = amplitudes[clean_indices, ant_idx, :]
        ant_flags_clean = flags[clean_indices, ant_idx, :]
        unflagged_amps = ant_amps_clean[~ant_flags_clean]
        if len(unflagged_amps) > 0:
            mean_ant_amps.append(np.mean(unflagged_amps))
            ant_indices.append(ant_idx)
    if len(mean_ant_amps) < 3:
        logger.warning("Not enough unflagged antennas to perform outlier detection.")
        return new_flags
    mean_of_means, std_of_means = np.mean(mean_ant_amps), np.std(mean_ant_amps)
    if std_of_means == 0: return new_flags
    outlier_indices = np.where(np.abs(np.array(mean_ant_amps) - mean_of_means) > config.OUTLIER_ANTENNA_SIGMA * std_of_means)[0]
    if len(outlier_indices) > 0:
        for idx in outlier_indices:
            ant_to_flag_idx, ant_to_flag_num = ant_indices[idx], antennas[ant_indices[idx]]
            logger.warning(f"Flagging all data for outlier Antenna {ant_to_flag_num}. Mean amplitude was {mean_ant_amps[idx]:.3f} (vs. population mean {mean_of_means:.3f}, std {std_of_means:.3f}).")
            new_flags[:, ant_to_flag_idx, :] = True
    else:
        logger.info("No outlier antennas found.")
    return new_flags

def _flag_and_interpolate_rfi_channels(gains, flags, freqs_hz):
    if not SCIPY_AVAILABLE:
        logger.error("scipy is required for local RFI flagging but was not found. Skipping this step.")
        return gains, flags

    logger.info("Flagging channels with high scatter OR non-Gaussianity (kurtosis)...")
    new_gains, new_flags = gains.copy(), flags.copy()
    amplitudes = np.abs(gains)
    n_chan, n_ant, n_pol = gains.shape
    
    channel_norm_stds = np.zeros(n_chan)
    channel_kurtosis = np.zeros(n_chan)

    for i in range(n_chan):
        amps_this_chan = amplitudes[i, :, :]
        flags_this_chan = flags[i, :, :]
        unflagged_amps = amps_this_chan[~flags_this_chan]

        if len(unflagged_amps) > 4:
            mean_amp, std_amp = np.mean(unflagged_amps), np.std(unflagged_amps)
            if mean_amp > 1e-9:
                channel_norm_stds[i] = std_amp / mean_amp
            channel_kurtosis[i] = stats.kurtosis(unflagged_amps, fisher=True)

    window_size = config.NARROW_RFI_LOCAL_WINDOW_CHANS
    local_median_std = scipy.ndimage.median_filter(channel_norm_stds, size=window_size, mode='reflect')
    abs_dev_std = np.abs(channel_norm_stds - local_median_std)
    local_mad_std = scipy.ndimage.median_filter(abs_dev_std, size=window_size, mode='reflect')
    threshold_array_std = local_median_std + config.NARROW_RFI_SCATTER_SIGMA * (1.4826 * local_mad_std)
    
    global_median_std = np.median(channel_norm_stds[channel_norm_stds > 0])
    global_mad_std = np.median(np.abs(channel_norm_stds[channel_norm_stds > 0] - global_median_std))
    min_threshold_std = global_median_std + 5.0 * (1.4826 * global_mad_std)
    final_threshold_std = np.maximum(threshold_array_std, min_threshold_std)

    local_median_kurt = scipy.ndimage.median_filter(channel_kurtosis, size=window_size, mode='reflect')
    abs_dev_kurt = np.abs(channel_kurtosis - local_median_kurt)
    local_mad_kurt = scipy.ndimage.median_filter(abs_dev_kurt, size=window_size, mode='reflect')
    threshold_array_kurt = local_median_kurt + config.NARROW_RFI_KURTOSIS_SIGMA * (1.4826 * local_mad_kurt)
    final_threshold_kurt = np.maximum(threshold_array_kurt, 1.0)
    
    is_bad_scatter = channel_norm_stds > final_threshold_std
    is_bad_kurtosis = channel_kurtosis > final_threshold_kurt
    bad_chan_indices = np.where(is_bad_scatter | is_bad_kurtosis)[0]
    
    if len(bad_chan_indices) == 0:
        logger.info("No narrow-band RFI channels found to flag.")
        return new_gains, new_flags

    n_scatter, n_kurtosis = np.sum(is_bad_scatter), np.sum(is_bad_kurtosis)
    logger.warning(f"Found {len(bad_chan_indices)} RFI channels ({n_scatter} by scatter, {n_kurtosis} by kurtosis). Flagging and interpolating.")
    new_flags[bad_chan_indices, :, :] = True

    all_chans = np.arange(n_chan)
    for ant_idx in range(n_ant):
        for pol_idx in range(n_pol):
            good_indices_before = np.where(~flags[:, ant_idx, pol_idx])[0]
            if len(good_indices_before) < 2: continue
            good_gains_before = gains[good_indices_before, ant_idx, pol_idx]
            interp_values = np.interp(bad_chan_indices, good_indices_before, good_gains_before)
            new_gains[bad_chan_indices, ant_idx, pol_idx] = interp_values
            
    return new_gains, new_flags

def flag_data_cubes(gains, flags, freqs_hz, antennas):
    current_gains, current_flags = gains.copy(), flags.copy()
    
    if config.OUTLIER_FLAG_SIGMA and config.OUTLIER_FLAG_SIGMA > 0:
        logger.info(f"Starting iterative sigma-clipping ({config.OUTLIER_FLAG_ITERATIONS} iterations, sigma={config.OUTLIER_FLAG_SIGMA}).")
        amplitudes = np.abs(current_gains)
        n_chan, n_ant, n_pol = amplitudes.shape
        for iteration in range(config.OUTLIER_FLAG_ITERATIONS):
            flags_in_iter = 0
            for chan_idx in range(n_chan):
                for pol_idx in range(n_pol):
                    amps, flgs = amplitudes[chan_idx, :, pol_idx], current_flags[chan_idx, :, pol_idx]
                    if np.all(flgs): continue
                    unflagged_amps = amps[~flgs]
                    if len(unflagged_amps) < 2: continue
                    mean_amp, std_amp = np.mean(unflagged_amps), np.std(unflagged_amps)
                    if std_amp == 0: continue
                    is_outlier = np.abs(amps - mean_amp) > config.OUTLIER_FLAG_SIGMA * std_amp
                    flags_to_add = np.logical_or(flgs, is_outlier)
                    flags_in_iter += np.sum(flags_to_add) - np.sum(flgs)
                    current_flags[chan_idx, :, pol_idx] = flags_to_add
            logger.info(f"  Iteration {iteration + 1}: Flagged an additional {flags_in_iter} outlier solutions.")
            if flags_in_iter == 0:
                logger.info("  No new outliers found. Stopping iteration early.")
                break
        logger.info("Iterative clipping complete.")

    min_freq_mhz = _determine_rfi_cutoff_freq(current_gains, current_flags, freqs_hz)
    if min_freq_mhz and min_freq_mhz > 0:
        logger.info(f"Applying dynamically determined hard flag for solutions below {min_freq_mhz:.2f} MHz.")
        low_freq_indices = np.where(freqs_hz / 1e6 < min_freq_mhz)[0]
        if len(low_freq_indices) > 0:
            current_flags[low_freq_indices, :, :] = True

    current_flags = _flag_outlier_antennas(current_gains, current_flags, freqs_hz, antennas)
    final_gains, final_flags = _flag_and_interpolate_rfi_channels(current_gains, current_flags, freqs_hz)

    return final_gains, final_flags

def summarize_and_plot_flags(caltable_path, initial_flags, final_flags, freqs_hz, antennas, context):
    """Saves the flag summary plot to the QA directory so it's included in the PDF report."""
    logger.info("--- Generating Flagging Summary ---")
    newly_flagged = final_flags & ~initial_flags
    total_new_flags = np.sum(newly_flagged)
    logger.info(f"Applied a total of {total_new_flags} new flags to the bandpass table.")

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

def plot_all_antennas_comparison(initial_gains, initial_flags, final_gains, final_flags, freqs_hz, output_path):
    logger.info("Generating bandpass amplitude comparison plot for all antennas...")
    if not PLOTTING_AVAILABLE: return

    freqs_mhz = freqs_hz / 1e6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)
    
    try:
        # --- Before Flagging ---
        for ant_idx in range(initial_gains.shape[1]):
            for pol_idx in range(initial_gains.shape[2]):
                amp = np.abs(initial_gains[:, ant_idx, pol_idx])
                amp[initial_flags[:, ant_idx, pol_idx]] = np.nan
                ax1.plot(freqs_mhz, amp, alpha=0.1, color='C0')
        ax1.set_title('Before Flagging')
        ax1.set_xlabel('Frequency (MHz)'); ax1.set_ylabel('Gain Amplitude')
        ax1.grid(True, alpha=0.3); ax1.set_yscale('log')

        # --- After Flagging ---
        for ant_idx in range(final_gains.shape[1]):
            for pol_idx in range(final_gains.shape[2]):
                amp = np.abs(final_gains[:, ant_idx, pol_idx])
                amp[final_flags[:, ant_idx, pol_idx]] = np.nan
                ax2.plot(freqs_mhz, amp, alpha=0.1, color='C0')
        ax2.set_title('After Flagging')
        ax2.set_xlabel('Frequency (MHz)')
        ax2.grid(True, alpha=0.3)

        fig.suptitle('All Antenna Gain Amplitudes Comparison')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path)
        logger.info(f"Saved all-antenna comparison plot to {os.path.basename(output_path)}")
    except Exception as e:
        logger.error(f"Failed to generate all-antenna comparison plot: {e}", exc_info=True)
    finally:
        plt.close(fig)

def generate_per_antenna_bpplots_pdf(initial_gains, initial_flags, final_gains, final_flags, freqs_hz, antennas, output_path):
    logger.info("Generating multi-page PDF of per-antenna bandpass solutions...")
    if not PLOTTING_AVAILABLE: return
    
    freqs_mhz = freqs_hz / 1e6
    try:
        with PdfPages(output_path) as pdf:
            for ant_idx, ant_num in enumerate(antennas):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
                
                # --- Amplitude Plot ---
                for pol_idx, pol_name in enumerate(['X', 'Y']):
                    # Initial
                    amp_i = np.abs(initial_gains[:, ant_idx, pol_idx])
                    amp_i[initial_flags[:, ant_idx, pol_idx]] = np.nan
                    ax1.plot(freqs_mhz, amp_i, alpha=0.5, label=f'Initial Pol {pol_name}')
                    # Final
                    amp_f = np.abs(final_gains[:, ant_idx, pol_idx])
                    amp_f[final_flags[:, ant_idx, pol_idx]] = np.nan
                    ax1.plot(freqs_mhz, amp_f, alpha=1.0, label=f'Final Pol {pol_name}')
                ax1.set_ylabel('Gain Amplitude'); ax1.grid(True, alpha=0.3)
                ax1.legend(); ax1.set_yscale('log')

                # --- Phase Plot ---
                for pol_idx, pol_name in enumerate(['X', 'Y']):
                    # Initial
                    phase_i = np.angle(initial_gains[:, ant_idx, pol_idx], deg=True)
                    phase_i[initial_flags[:, ant_idx, pol_idx]] = np.nan
                    ax2.plot(freqs_mhz, phase_i, '.', alpha=0.5, label=f'Initial Pol {pol_name}')
                    # Final
                    phase_f = np.angle(final_gains[:, ant_idx, pol_idx], deg=True)
                    phase_f[final_flags[:, ant_idx, pol_idx]] = np.nan
                    ax2.plot(freqs_mhz, phase_f, '.', alpha=1.0, label=f'Final Pol {pol_name}')
                ax2.set_ylabel('Gain Phase (deg)'); ax2.set_xlabel('Frequency (MHz)')
                ax2.grid(True, alpha=0.3); ax2.set_ylim(-180, 180)

                fig.suptitle(f'Antenna {ant_num} Bandpass Solutions')
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig)
                plt.close(fig)
        logger.info(f"Saved per-antenna diagnostic PDF to {os.path.basename(output_path)}")
    except Exception as e:
        logger.error(f"Failed to generate per-antenna PDF: {e}", exc_info=True)


def run_bandpass_flagging(input_caltable_path, tables_dir, context):
    """Orchestrates the bandpass flagging process and generates diagnostic plots."""
    logger.info("--- Starting Bandpass Table Flagging and Refinement ---")

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

    initial_gains, initial_flags, freqs_hz, antennas = load_bandpass_data(output_caltable_path)
    if initial_flags is None:
        logger.error("Could not load data from bandpass table. Aborting flagging.")
        return None
        
    final_gains, final_flags = flag_data_cubes(initial_gains, initial_flags, freqs_hz, antennas)
    
    if not write_bandpass_data(output_caltable_path, final_gains, final_flags):
        logger.error("Failed to write flagged data. Aborting.")
        return None
        
    summarize_and_plot_flags(output_caltable_path, initial_flags, final_flags, freqs_hz, antennas, context)
    
    # --- Generate New Diagnostic Plots ---
    if PLOTTING_AVAILABLE:
        plot_subdir = os.path.join(context['qa_dir'], 'bandpass_plots')
        os.makedirs(plot_subdir, exist_ok=True)
        
        # Plot 1: All antennas overlaid, before vs after
        comp_plot_path = os.path.join(plot_subdir, 'bp_solutions_comparison.png')
        plot_all_antennas_comparison(initial_gains, initial_flags, final_gains, final_flags, freqs_hz, comp_plot_path)

        # Plot 2: Per-antenna amp/phase in a multi-page PDF
        pdf_path = os.path.join(plot_subdir, 'bp_solutions_per_antenna.pdf')
        generate_per_antenna_bpplots_pdf(initial_gains, initial_flags, final_gains, final_flags, freqs_hz, antennas, pdf_path)

    logger.info("Bandpass flagging process completed successfully.")
    return output_caltable_path

# ==============================================================================
# === Main Logic ===
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

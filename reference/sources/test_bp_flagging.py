#!/usr/bin/env python
# test_bp_flagging.py

import os
import sys
import argparse
import logging
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving files
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# --- Configuration for Flagging ---
# Step 0: Normalization
NORMALIZATION_FREQ_RANGE_MHZ = (40.0, 55.0) # NEW: Range for per-ant normalization

# Step 1: Per-channel scatter flagging
CHANNEL_SCATTER_CLEAN_FREQ_RANGE_MHZ = (40.0, 60.0)
CHANNEL_SCATTER_THRESHOLD_MULTIPLIER = 7.0 # MODIFIED as requested

# Step 2 & 3: Template creation and deviation flagging
TEMPLATE_SMOOTHING_KERNEL_SIZE = 51 # Kernel size for smoothing the template (must be odd)
DEVIATION_FLAG_SIGMA = 7.0        # MODIFIED as requested

# Step 4: Per-antenna gain outlier flagging
GAIN_OUTLIER_SIGMA = 7.0     # MODIFIED as requested

# Step 5: Iterative outlier flagging (from ref.py)
OUTLIER_FLAG_SIGMA = 3           # MODIFIED as requested
OUTLIER_FLAG_ITERATIONS = 15      # MODIFIED as requested

# Step 6: Channel quorum flagging
CHANNEL_QUORUM_THRESHOLD = 0.5 # Flag channel if > 50% of antennas are flagged


# --- Setup Logging and Table Tools ---
logger = logging.getLogger()

try:
    import casacore.tables as pt
    TABLE_TOOLS_AVAILABLE = True
except ImportError:
    TABLE_TOOLS_AVAILABLE = False
    pass

# --- Data I/O (Unchanged) ---
def load_bandpass_data(caltable_path):
    if not TABLE_TOOLS_AVAILABLE:
        logger.error("FATAL: casacore.tables not found. Please run in a CASA environment.")
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
            columns_to_read = ["ANTENNA1", "SPECTRAL_WINDOW_ID", "CPARAM", "FLAG"]
            for rownr in range(t.nrows()):
                ant_id = t.getcell("ANTENNA1", rownr)
                spw_id = t.getcell("SPECTRAL_WINDOW_ID", rownr)
                start_chan, end_chan = channel_offset[spw_id], channel_offset[spw_id] + spw_chans[spw_id]
                full_gains[start_chan:end_chan, ant_id, :] = t.getcell("CPARAM", rownr)
                full_flags[start_chan:end_chan, ant_id, :] = t.getcell("FLAG", rownr)
        logger.info(f"Successfully reconstructed data cubes with shape {full_gains.shape}")
        return full_gains, full_flags, full_freq_axis, antennas
    except Exception as e:
        logger.error("Failed during robust data loading.", exc_info=True)
        return None, None, None, None

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

# --- MODIFIED: Implements new 6-Step Flagging Logic ---
def flag_data_cubes(gains, flags, freqs_hz):
    new_flags = flags.copy()
    amplitudes = np.abs(gains) # Raw amplitudes, used for Steps 2-6
    n_chan, n_ant, n_pol = amplitudes.shape
    freqs_mhz = freqs_hz / 1e6
    channel_indices = np.arange(n_chan)

    # --- NEW Step 0: Normalize Amplitudes for Scatter Flagging ---
    logger.info(f"--- Step 0: Normalizing amplitudes for scatter flagging ---")
    norm_indices = np.where((freqs_mhz >= NORMALIZATION_FREQ_RANGE_MHZ[0]) & 
                            (freqs_mhz <= NORMALIZATION_FREQ_RANGE_MHZ[1]))[0]
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

    clean_indices_step1 = np.where((freqs_mhz >= CHANNEL_SCATTER_CLEAN_FREQ_RANGE_MHZ[0]) & 
                                   (freqs_mhz <= CHANNEL_SCATTER_CLEAN_FREQ_RANGE_MHZ[1]))[0]
    
    if len(clean_indices_step1) > 10:
        # Get median scatter of *normalized* data
        median_clean_scatter = np.median(stds_per_channel[clean_indices_step1])
        if median_clean_scatter > 0:
            threshold = median_clean_scatter * CHANNEL_SCATTER_THRESHOLD_MULTIPLIER
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

    if len(valid_indices) > TEMPLATE_SMOOTHING_KERNEL_SIZE:
         logger.info(f"Smoothing template with median filter (kernel={TEMPLATE_SMOOTHING_KERNEL_SIZE}).")
         interp_template = median_template.copy()
         flagged_template_indices = np.where(median_template <= 0)[0] 
         unflagged_template_indices = np.where(median_template > 0)[0]
         if len(flagged_template_indices) > 0 and len(unflagged_template_indices) > 1:
              interp_template[flagged_template_indices] = np.interp(flagged_template_indices, unflagged_template_indices, median_template[unflagged_template_indices])
         
         median_template = medfilt(interp_template, kernel_size=TEMPLATE_SMOOTHING_KERNEL_SIZE)


    # --- Step 3: Flag deviations from the template (UNCHANGED) ---
    # Note: This step correctly uses the *original* 'amplitudes'
    logger.info(f"--- Step 3: Flagging per-antenna deviations from template (sigma={DEVIATION_FLAG_SIGMA}) ---")
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
                    threshold = DEVIATION_FLAG_SIGMA * robust_std_res
                    is_deviant = ant_residuals > threshold
                    
                    flags_to_add = np.logical_or(ant_flags, is_deviant)
                    total_deviation_flags += np.sum(flags_to_add) - np.sum(ant_flags)
                    new_flags[:, i, j] = flags_to_add

    logger.info(f"Step 3 complete. Flagged {total_deviation_flags} solutions deviating from the template.")

    # --- Step 4: Flag whole-antenna gain outliers (UNCHANGED) ---
    # Note: This step correctly uses the *original* 'amplitudes'
    logger.info(f"--- Step 4: Flagging whole-antenna gain outliers (sigma={GAIN_OUTLIER_SIGMA}) ---")
    
    clean_indices_step4 = clean_indices_step1
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
                threshold = GAIN_OUTLIER_SIGMA * robust_std
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
    sigma_threshold = OUTLIER_FLAG_SIGMA
    if sigma_threshold and sigma_threshold > 0:
        logger.info(f"--- Step 5: Starting iterative sigma-clipping ({OUTLIER_FLAG_ITERATIONS} iterations, sigma={sigma_threshold}) ---")
        
        total_new_flags = 0
        for iteration in range(OUTLIER_FLAG_ITERATIONS):
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
    logger.info(f"--- Step 6: Applying channel quorum flagging (Threshold: {CHANNEL_QUORUM_THRESHOLD*100}%) ---")
    
    flags_per_ant = np.any(new_flags, axis=2) 
    flagged_ant_count_chan = np.sum(flags_per_ant, axis=1)
    quorum = int(CHANNEL_QUORUM_THRESHOLD * n_ant)
    channels_to_flag_all = np.where(flagged_ant_count_chan > quorum)[0]
    
    if len(channels_to_flag_all) > 0:
        logger.info(f"Flagging all antennas in {len(channels_to_flag_all)} channels due to >{quorum} antennas being flagged.")
        new_flags[channels_to_flag_all, :, :] = True
    else:
        logger.info("No channels met the quorum flagging threshold.")
    
    return new_flags

def _consolidate_ranges(indices):
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

# --- MODIFIED: Added console_handler parameter (Logging fix) ---
def summarize_and_plot_flags(caltable_path, initial_flags, final_flags, freqs_hz, antennas, console_handler):
    logger.info("--- Generating Flagging Summary ---")
    newly_flagged = final_flags & ~initial_flags
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
    
    # --- MODIFICATION: Suppress summary from console, but not from file (Logging fix) ---
    original_level = console_handler.level
    console_handler.setLevel(logging.WARNING) # Temporarily set console to WARNING
    logger.info(summary_string)               # This will ONLY go to the file
    console_handler.setLevel(original_level)  # Restore original level
    
    logger.info("Generating 2D flagging plot...") # This will still go to console
    plot_filename = f"{os.path.basename(caltable_path)}.flag_summary.png"
    flag_matrix = np.any(newly_flagged, axis=2)
    plt.figure(figsize=(12, 8))
    plt.imshow(flag_matrix.T, aspect='auto', origin='lower',
               cmap='binary', interpolation='none',
               extent=[freqs_mhz[0], freqs_mhz[-1], antennas[0]-0.5, antennas[-1]+0.5])
    plt.title(f'Newly Flagged Solutions for {os.path.basename(caltable_path)}')
    plt.ylabel('Antenna Correlator Number')
    plt.xlabel('Frequency (MHz)')
    plt.grid(alpha=0.2)
    plt.savefig(plot_filename)
    logger.info(f"Plot saved to: {plot_filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test script for flagging a CASA bandpass calibration table.")
    parser.add_argument("caltable", help="Path to the CASA calibration table directory (e.g., mycal.B)")
    args = parser.parse_args()
    caltable_path = args.caltable
    if not os.path.isdir(caltable_path):
        print(f"Error: Path provided is not a valid directory: {caltable_path}")
        sys.exit(1)
    log_filename = f"{os.path.basename(caltable_path.rstrip(os.sep))}.flagging.log"
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    # --- MODIFICATION: Keep a handle to the console_handler (Logging fix) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler) # <-- This provides INFO level to terminal
    
    if not TABLE_TOOLS_AVAILABLE:
        root_logger.info("casacore.tables is required but was not found.")
        sys.exit(1)
    output_caltable_path = f"{caltable_path.rstrip(os.sep)}.flagged"
    logger.info(f"Original table: {caltable_path}")
    logger.info(f"Output table:   {output_caltable_path}")
    if os.path.exists(output_caltable_path):
        logger.warning(f"Output path {output_caltable_path} already exists. Removing it.")
        shutil.rmtree(output_caltable_path)
    try:
        logger.info("Copying original table...")
        pt.tablecopy(caltable_path, output_caltable_path)
        logger.info("Copy complete.")
    except Exception as e:
        logger.error("Failed to copy calibration table.", exc_info=True)
        sys.exit(1)
    initial_gains, initial_flags, freqs_hz, antennas = load_bandpass_data(output_caltable_path)
    if initial_flags is None:
        sys.exit(1)
    final_flags = flag_data_cubes(initial_gains, initial_flags, freqs_hz)
    
    if not write_bandpass_flags(output_caltable_path, final_flags):
        sys.exit(1)
        
    # --- MODIFICATION: Pass console_handler to the function (Logging fix) ---
    summarize_and_plot_flags(output_caltable_path, initial_flags, final_flags, freqs_hz, antennas, console_handler)
    logger.info("Script finished successfully.")

if __name__ == "__main__":
    main()

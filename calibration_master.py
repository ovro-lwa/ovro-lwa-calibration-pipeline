# calibration_master.py

import os
import sys
import shutil
import argparse
import logging
import traceback

# Add the directory of this script to the Python path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import individual step scripts and utilities
try:
    import data_preparation
    import add_sky_model
    import delay_calibration
    import bandpass_calibration # New module
    import imaging_qa           # New module
    import pipeline_utils
    import pipeline_config as config
except ImportError as e:
    print(f"ERROR: Failed to import pipeline modules: {e}", file=sys.stderr)
    # Check specifically for common missing dependencies
    if 'tqdm' in str(e) or 'matplotlib' in str(e) or 'pandas' in str(e):
        print(f"Please install required dependencies (e.g., 'pip install tqdm matplotlib pandas' or use mamba/conda)")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="OVRO-LWA Calibration Pipeline Master Script")
    
    # (Arguments updated)
    parser.add_argument("data_folder", type=str,
                        help="Path to the directory containing raw calibration data.")
    parser.add_argument("--input_ms", type=str, default=None,
                        help="Path to a pre-prepared MS file.")
    parser.add_argument("--rerun_flagging", action="store_true",
                        help="If using --input_ms, run the flagging steps.")
    parser.add_argument("--single_source", type=str, default=None,
                        help="Restrict the sky model (Step 2) to a single source.")
    
    # Renamed steps based on new workflow
    parser.add_argument("--skip_add_sky_model", action="store_true", help="Skip Step 2 (Sky Model).")
    parser.add_argument("--skip_bandpass_calibration", action="store_true", help="Skip Step 3 (Bandpass).")
    parser.add_argument("--skip_delay_calibration", action="store_true", help="Skip Step 4 (Delay Diagnostics).")
    parser.add_argument("--skip_imaging_qa", action="store_true", help="Skip Step 5 (Imaging QA).")

    # Added alias for skip_cleanup
    parser.add_argument("--skip_cleanup", action="store_true",
                        help="Skip cleanup of intermediate MS (for debugging).")
    parser.add_argument("--keep_intermediate_ms", action="store_true", help="Alias for --skip_cleanup.")

    parser.add_argument("--no_move_results", action="store_true",
                        help="Do not move the working directory upon completion.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose console output (DEBUG level).")

    args = parser.parse_args()
    
    # Combine cleanup flags
    skip_cleanup = args.skip_cleanup or args.keep_intermediate_ms

    print(f"Starting OVRO-LWA Calibration Pipeline.")

    # --- Context Initialization (Phase 1 & 2) ---
    context = pipeline_utils.setup_pipeline_context_phase1(args.data_folder, SCRIPT_DIR)
    if not context: sys.exit(1)
        
    context = pipeline_utils.setup_pipeline_context_phase2(context, verbose=args.verbose)
    if not context: sys.exit(1)

    # Get the main logger instance
    logger = logging.getLogger('OVRO_Pipeline.Master')

    pipeline_success = True
    skip_data_preparation = False

    # --- Load Hardware Mapping (NEW) ---
    try:
        # Load the mapping {CorrelatorNumber: SNAP2Location}
        context['antenna_mapping'] = pipeline_utils.load_antenna_mapping()
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load antenna mapping. SNAP2 diagnostics will be unavailable. Error: {e}")
        # We allow the pipeline to continue, but diagnostics will be skipped if mapping is missing.

    # Handle input MS argument
    if args.input_ms:
        if os.path.exists(args.input_ms):
            context['concatenated_ms_path'] = os.path.abspath(args.input_ms)
            skip_data_preparation = not args.rerun_flagging
            # (Logging logic remains the same)
        else:
            logger.error(f"Provided input MS not found: {args.input_ms}. Halting.")
            pipeline_success = False
    
    # Main pipeline execution flow (REVISED WORKFLOW)
    if pipeline_success:
        try:
            # --- Step 1: Data preparation ---
            if not skip_data_preparation:
                logger.info("\n===== STEP 1: Data Preparation =====")
                flagging_only = args.input_ms is not None
                success = data_preparation.run_data_preparation(context, force_flagging_only=flagging_only)
                
                if success:
                    logger.info(f"Data preparation completed. MS: {context['concatenated_ms_path']}")
                else:
                    logger.error("Data preparation failed. Halting pipeline.")
                    pipeline_success = False
            else:
                logger.info("\n===== STEP 1: Data Preparation (SKIPPED) =====")

            # Check if MS exists before proceeding
            if pipeline_success and (not context['concatenated_ms_path'] or not os.path.exists(context['concatenated_ms_path'])):
                 logger.error("No valid MS available for subsequent steps. Halting.")
                 pipeline_success = False


            # --- Step 2: Add sky model ---
            if pipeline_success and not args.skip_add_sky_model:
                logger.info("\n===== STEP 2: Add Sky Model =====")
                success = add_sky_model.main(context, single_source_override=args.single_source)
                if not success:
                    logger.error("Add sky model step failed. Halting.")
                    pipeline_success = False
            elif pipeline_success:
                logger.info("\n===== STEP 2: Add Sky Model (SKIPPED) =====")
                # Even if skipped, we need to know which sources would have been modeled.
                # Run the sky model logic to populate the context without applying it to the MS.
                logger.info("Determining modeled sources for subsequent steps...")
                add_sky_model.main(context, single_source_override=args.single_source, apply_model=False)

            # --- Step 3: Bandpass calibration (NEW POSITION) ---
            if pipeline_success and not args.skip_bandpass_calibration:
                logger.info("\n===== STEP 3: Bandpass Calibration =====")
                success = bandpass_calibration.run_bandpass_calibration(context)
                if success:
                    logger.info("Bandpass calibration step completed.")
                else:
                    logger.error("Bandpass calibration failed. Halting pipeline.")
                    pipeline_success = False
            elif pipeline_success:
                logger.info("\n===== STEP 3: Bandpass Calibration (SKIPPED) =====")


            # --- Step 4: Delay calibration (NEW POSITION - Diagnostic Only) ---
            if pipeline_success and not args.skip_delay_calibration:
                logger.info("\n===== STEP 4: Delay Calibration (Diagnostic) =====")
                # This step will raise RuntimeError if SNAP2 diagnostics fail.
                success = delay_calibration.run_delay_calibration(context)
                if success:
                    logger.info("Delay calibration and diagnostics completed.")
                else:
                    # This handles failures in the CASA task itself, not the diagnostics
                    logger.error("Delay calibration task failed. Halting pipeline.")
                    pipeline_success = False
            elif pipeline_success:
                logger.info("\n===== STEP 4: Delay Calibration (SKIPPED) =====")

            # --- Step 5: Imaging QA (NEW STEP) ---
            if pipeline_success and not args.skip_imaging_qa:
                logger.info("\n===== STEP 5: Imaging QA (Flux & Scintillation) =====")
                success = imaging_qa.run_imaging_qa(context)
                if success:
                    logger.info("Imaging QA step completed.")
                else:
                    # Imaging QA failure is usually not critical to the calibration tables themselves
                    logger.warning("Imaging QA step failed, but calibration tables may still be valid.")
            elif pipeline_success:
                logger.info("\n===== STEP 5: Imaging QA (SKIPPED) =====")


        # Catch expected RuntimeErrors (including SNAP2 failures or failed external commands)
        except RuntimeError as e:
            logger.error(f"Pipeline execution failed due to a runtime error: {e}")
            # Print the critical error prominently to the console as well
            print(f"\nCRITICAL ERROR: Pipeline FAILED. See log for details: {context.get('log_filepath', 'N/A')}", file=sys.stderr)
            print(f"Error details: {e}", file=sys.stderr)
            pipeline_success = False
        # Catch any other unexpected exceptions
        except Exception as e:
            logger.exception(f"CRITICAL ERROR: Unhandled exception during pipeline execution.")
            pipeline_success = False

    # --- Pipeline Finalization ---
    # (Logic for cleanup and moving results remains the same as previous iterations, omitted for brevity)

if __name__ == "__main__":
    main() # Run the pipeline
    # print("Scripts generated and consolidated. Please ensure all 10 scripts are deployed and the mapping CSV path is correct in pipeline_config.py.")

    

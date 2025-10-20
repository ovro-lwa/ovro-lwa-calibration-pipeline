# calibration_master.py

import os
import sys
import shutil
import argparse
import logging
import traceback
from datetime import datetime
import glob

# Add the directory of this script to the Python path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ---
# === PRE-FLIGHT CHECK ===
# ---
# Import pipeline_config and run the check *first*.
try:
    import pipeline_config as config
except ImportError as e:
    print(f"FATAL ERROR: Failed to load 'pipeline_config.py'.", file=sys.stderr)
    print(f"This is likely because you are not in the required conda environment.", file=sys.stderr)
    print(f"Please ensure packages like 'numpy' and 'astropy' are available.", file=sys.stderr)
    print(f"Original error: {e}", file=sys.stderr)
    sys.exit(1)

if config.REQUIRED_CONDA_ENV:
    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    if current_env is None or os.path.basename(current_env) != config.REQUIRED_CONDA_ENV:
        print(f"FATAL ERROR: This pipeline must be run in the '{config.REQUIRED_CONDA_ENV}' conda environment.", file=sys.stderr)
        print(f"Currently active environment: '{current_env or 'None'}'", file=sys.stderr)
        print(f"Please run 'conda activate {config.REQUIRED_CONDA_ENV}' and try again.", file=sys.stderr)
        sys.exit(1)
# --- END PRE-FLIGHT CHECK ---


# Import individual step scripts and utilities
try:
    import data_preparation
    import add_sky_model
    import delay_calibration
    import bandpass_calibration
    import imaging
    import qa
    import pipeline_utils
except ImportError as e:
    # This block should now only catch *unexpected* import errors
    print(f"ERROR: Failed to import pipeline modules: {e}", file=sys.stderr)
    if 'tqdm' in str(e) or 'matplotlib' in str(e) or 'pandas' in str(e):
        print(f"Please install required dependencies (e.g., 'pip install tqdm matplotlib pandas' or use mamba/conda)")
    sys.exit(1)

# ==============================================================================
# === Pipeline Finalization ===
# ==============================================================================

def finalize_pipeline(context, success, skip_cleanup, no_move_results):
    """
    Handles the final steps of the pipeline: cleanup, determining success/failure,
    and moving the working directory to the final destination.
    """
    logger = logging.getLogger('OVRO_Pipeline.Master.Finalize')
    logger.info("\n===== PIPELINE FINALIZATION =====")

    # Update context status
    context['status'] = 'SUCCESS' if success else 'FAILURE'
    context['end_time'] = datetime.now()
    duration = context['end_time'] - context['start_time']
    logger.info(f"Pipeline finished with status: {context['status']}. Total duration: {duration}.")

    # --- Cleanup Logic ---
    if skip_cleanup:
        logger.info("Skipping all cleanup as requested via command-line flag.")
    elif not success:
        logger.warning("Pipeline failed. All intermediate files will be kept for debugging.")
        concat_ms = context.get('concat_ms')
        if concat_ms:
            logger.warning(f"  -> MS Path: {concat_ms}")
    else:
        # This block runs only if success is True and skip_cleanup is False
        logger.info("Pipeline succeeded. Cleaning up intermediate products.")

        # 1. Clean up concatenated MS
        concat_ms = context.get('concat_ms')
        if concat_ms and os.path.exists(concat_ms):
            logger.info(f"Cleaning up concatenated MS: {os.path.basename(concat_ms)}")
            try:
                shutil.rmtree(concat_ms)
            except Exception as e:
                logger.error(f"Failed to clean up concatenated MS: {e}")
        else:
            logger.info("No concatenated MS found to clean up.")

        # 2. Clean up intermediate WSClean images (psf, residual, model)
        qa_dir = context.get('qa_dir')
        if qa_dir and os.path.isdir(qa_dir):
            suffixes_to_delete = ['psf', 'residual', 'model']
            files_deleted_count = 0
            logger.info("Searching for intermediate WSClean images to delete...")
            # Search in main QA dir and subdirectories
            for suffix in suffixes_to_delete:
                search_pattern = os.path.join(qa_dir, '**', f'*-{suffix}.fits')
                files_to_delete = glob.glob(search_pattern, recursive=True)

                for f in files_to_delete:
                    try:
                        os.remove(f)
                        files_deleted_count += 1
                        logger.debug(f"Deleted: {os.path.basename(f)}")
                    except OSError as e:
                        logger.warning(f"Error deleting file {f}: {e}")

            if files_deleted_count > 0:
                logger.info(f"Deleted {files_deleted_count} intermediate image files.")
        else:
            logger.warning("QA directory not found. Skipping cleanup of intermediate images.")

    # --- Move Results Directory ---
    if no_move_results:
        logger.info(f"Skipping move of results directory. Data remains in: {context['working_dir']}")
        return

    working_dir = context['working_dir']
    final_parent_structure = context['final_parent_structure']

    status_dir_name = 'successful' if success else 'unsuccessful'
    status_dir = os.path.join(final_parent_structure, status_dir_name)

    processing_timestamp = os.path.basename(working_dir)
    final_destination = os.path.join(status_dir, processing_timestamp)

    logger.info(f"Moving working directory to final destination: {final_destination}")

    try:
        os.makedirs(status_dir, exist_ok=True)

        if os.path.exists(final_destination):
            logger.warning(f"Destination directory already exists. Overwriting: {final_destination}")
            shutil.rmtree(final_destination)

        shutil.move(working_dir, final_destination)
        logger.info("Move completed successfully.")

        intermediate_parent = os.path.dirname(working_dir)
        # Check if parent exists and is empty before trying to remove
        if intermediate_parent and os.path.exists(intermediate_parent) and not os.listdir(intermediate_parent):
            try:
                os.rmdir(intermediate_parent)
                logger.debug(f"Removed empty intermediate directory: {intermediate_parent}")
            except OSError: # Ignore error if directory is somehow not empty or permissions issue
                pass

    except Exception as e:
        logger.error(f"CRITICAL: Failed to move working directory to final destination. Data remains in {working_dir}. Error: {e}")

# ==============================================================================
# === Main Execution ===
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="OVRO-LWA Calibration Pipeline Master Script")

    # Required argument
    parser.add_argument("data_folder", type=str,
                        help="Path to the directory containing raw calibration data.")

    # Optional arguments for resuming/reprocessing
    parser.add_argument("--input_ms", type=str, default=None,
                        help="Path to a pre-prepared MS file. Skips concatenation/FIELD_ID fixing.")
    parser.add_argument("--rerun_flagging", action="store_true",
                        help="If using --input_ms, force re-running the flagging steps.")
    parser.add_argument("--skip_mnc_flagging", action="store_true",
                        help="Skip antenna flagging based on MNC health checks.")

    # Optional arguments for model customization
    parser.add_argument("--single_source", type=str, default=None,
                        help="Restrict the sky model (Step 2) to a single named source.")

    # Arguments for skipping steps
    parser.add_argument("--skip_add_sky_model", action="store_true", help="Skip Step 2 (Sky Model).")
    parser.add_argument("--skip_bandpass_calibration", action="store_true", help="Skip Step 3 (Bandpass).")
    parser.add_argument("--skip_delay_calibration", action="store_true", help="Skip Step 4 (Delay Diagnostics).")
    parser.add_argument("--skip_imaging", action="store_true", help="Skip Step 5 (Imaging).")
    parser.add_argument("--skip_qa", action="store_true", help="Skip Step 6 (QA).")

    # Arguments for finalization control
    parser.add_argument("--skip_cleanup", action="store_true",
                        help="Skip cleanup of intermediate MS (for debugging).")
    parser.add_argument("--keep_intermediate_ms", action="store_true", help="Alias for --skip_cleanup.")
    parser.add_argument("--no_move_results", action="store_true",
                        help="Do not move the working directory to the final successful/unsuccessful location.")

    # Other options
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose console output (DEBUG level).")

    args = parser.parse_args()

    skip_cleanup = args.skip_cleanup or args.keep_intermediate_ms

    print(f"Starting OVRO-LWA Calibration Pipeline.") # First non-debug print

    # --- Context Initialization ---
    context = None
    try:
        # The single-call context setup
        context = pipeline_utils.initialize_context(args.data_folder, config.PARENT_OUTPUT_DIR, SCRIPT_DIR)

        # Get the root logger configured by initialize_context
        logger = logging.getLogger('OVRO_Pipeline')

        # Adjust console log level based on the verbose flag
        if args.verbose:
            for handler in logger.handlers:
                # We identify the console handler by checking if it's a StreamHandler writing to stdout/stderr
                if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
                    handler.setLevel(logging.DEBUG)
                    logger.debug("Verbose logging enabled.")
                    break

        # Now get the specific logger for the Master script
        logger = logging.getLogger('OVRO_Pipeline.Master')

    except Exception as e:
        # Catch potential setup failures, like no MS files found
        print(f"CRITICAL ERROR: Pipeline context setup failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # If context initialization fails, we cannot proceed or finalize gracefully
        sys.exit(1)

    # --- Main Pipeline Logic ---
    pipeline_success = True
    skip_data_preparation = False

    try:
        # Load antenna mapping for diagnostics
        try:
            context['antenna_mapping'] = pipeline_utils.load_antenna_mapping()
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load antenna mapping. SNAP2 diagnostics will be unavailable. Error: {e}")
            context['antenna_mapping'] = None

        # Handle input MS argument
        if args.input_ms:
            if os.path.exists(args.input_ms):
                logger.info(f"Using provided input MS: {args.input_ms}")
                context['concat_ms'] = os.path.abspath(args.input_ms)
                # If input_ms is provided, we skip concatenation/fixing unless flagging is forced
                skip_data_preparation = not args.rerun_flagging
            else:
                logger.error(f"Provided input MS not found: {args.input_ms}. Halting.")
                pipeline_success = False

        # --- Pipeline Steps ---
        if pipeline_success:
            # Step 1: Data Preparation
            if not skip_data_preparation:
                logger.info("\n===== STEP 1: Data Preparation =====")
                flagging_only = args.input_ms is not None and args.rerun_flagging
                success = data_preparation.run_data_preparation(
                    context,
                    force_flagging_only=flagging_only,
                    skip_mnc_flagging=args.skip_mnc_flagging
                )

                if success:
                    logger.info(f"Data preparation completed. MS: {context.get('concat_ms')}")
                else:
                    logger.error("Data preparation failed. Halting pipeline.")
                    pipeline_success = False
            else:
                logger.info("\n===== STEP 1: Data Preparation (SKIPPED) =====")
                logger.info("Concatenation, FIELD_ID fixing, and flagging skipped.")


            # Critical check for MS availability
            if pipeline_success and (not context.get('concat_ms') or not os.path.exists(context.get('concat_ms'))):
                 logger.error("No valid MS available for subsequent steps. Halting.")
                 pipeline_success = False

            # Step 2: Add Sky Model
            if pipeline_success and not args.skip_add_sky_model:
                logger.info("\n===== STEP 2: Add Sky Model =====")
                success = add_sky_model.main(context, single_source_override=args.single_source)
                if not success:
                    logger.error("Add sky model step failed (no suitable sources or task error). Halting.")
                    pipeline_success = False
            elif pipeline_success:
                logger.info("\n===== STEP 2: Add Sky Model (SKIPPED) =====")
                add_sky_model.main(context, single_source_override=args.single_source, apply_model=False)

            # Step 3: Bandpass Calibration
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

            # Step 4: Delay Calibration (Diagnostic)
            if pipeline_success and not args.skip_delay_calibration:
                logger.info("\n===== STEP 4: Delay Calibration (Diagnostic) =====")
                delay_calibration.run_delay_calibration(context)
            elif pipeline_success:
                logger.info("\n===== STEP 4: Delay Calibration (SKIPPED) =====")

            # Step 5: Imaging
            if pipeline_success and not args.skip_imaging:
                logger.info("\n===== STEP 5: Imaging =====")
                success = imaging.run_imaging(context)
                if not success:
                    logger.warning("Imaging step failed, but calibration tables may still be valid.")
            elif pipeline_success:
                logger.info("\n===== STEP 5: Imaging (SKIPPED) =====")

            # Step 6: QA
            if pipeline_success and not args.skip_qa:
                logger.info("\n===== STEP 6: QA =====")
                qa.run_qa(context)
            elif pipeline_success:
                logger.info("\n===== STEP 6: QA (SKIPPED) =====")

    except Exception as e:
        # Catch any unhandled exceptions during execution
        logger.exception(f"CRITICAL ERROR: Unhandled exception during pipeline execution.")
        pipeline_success = False

    finally:
        # Ensure finalization always runs if context was initialized
        if context:
            finalize_pipeline(context, pipeline_success, skip_cleanup, args.no_move_results)

if __name__ == "__main__":
    main()

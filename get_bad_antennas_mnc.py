# get_bad_antennas_mnc.py
# Helper script to be executed in the 'development' conda environment.
import argparse
import json
import sys
import traceback
from astropy.time import Time

# Attempt imports specific to the 'development' environment
_mnc_available = False
_import_error = None
try:
    # We need these specific tools now
    from mnc import anthealth
    # Mapping is required to convert names to correlator numbers
    import lwa_antpos.mapping as mapping
    _mnc_available = True
except ImportError as e:
    _import_error = str(e)

def get_bad_correlators_from_anthealth(mjd_time):
    """Uses anthealth.get_badants and mapping to find bad correlators."""
    
    # CRITICAL: Suppress Astropy ERFA warnings (like "dubious year") which might clutter stderr
    try:
        import warnings
        from astropy.utils.exceptions import ErfaWarning
        warnings.filterwarnings('ignore', category=ErfaWarning)
    except ImportError:
        pass # Astropy/Erfa not available or too old

    # Ensure the input time is correctly handled. We receive MJD as a float.
    # FIX: Explicitly specify the format='mjd' to prevent misinterpretation as ISO.
    try:
        time_obj = Time(mjd_time, format='mjd')
    except Exception as e:
        print(f"ERROR: Failed to parse MJD time {mjd_time}: {e}", file=sys.stderr)
        raise
    
    # Call get_badants. It expects the MJD value.
    # The structure returned is (closest_mjd, list_of_bad_signal_names)
    try:
        closest_mjd, bad_signal_names = anthealth.get_badants('selfcorr', time=time_obj.mjd)
    except Exception as e:
        # Log the specific error from anthealth if it fails
        print(f"ERROR: anthealth.get_badants failed with MJD {time_obj.mjd}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise
    
    # Map signal names (e.g., 'LWA-001A') to correlator numbers.
    correlators = []
    for name in bad_signal_names:
        try:
            # Strip polarization identifiers ('A' or 'B')
            ant_name = name.rstrip('A').rstrip('B')
            corr_num = mapping.antname_to_correlator(ant_name)
            correlators.append(int(corr_num))
        except Exception as e:
            # Log mapping errors but continue
            print(f"WARNING: Could not map antenna name '{name}' to correlator: {e}", file=sys.stderr)
            
    # Return unique, sorted list of correlator numbers
    # Ensure closest_mjd is a standard float for JSON serialization
    return sorted(list(set(correlators))), float(closest_mjd)

def main():
    parser = argparse.ArgumentParser(description="Helper script to get bad antennas using MNC/ORCA.")
    # Input is MJD as a float
    parser.add_argument("mjd_time", type=float, help="The MJD time of the observation.")
    args = parser.parse_args()

    if not _mnc_available:
        print(f"ERROR_ENV: MNC/ORCA modules not available. Ensure 'mnc' and 'lwa_antpos' are installed. Import Error: {_import_error}", file=sys.stderr)
        sys.exit(1)

    mjd_time = args.mjd_time
    
    try:
        # Use the revised function
        bad_correlator_numbers, data_timestamp_mjd = get_bad_correlators_from_anthealth(mjd_time)
        
        # Output the result as a JSON string to stdout
        output = {
            "requested_mjd": mjd_time,
            "data_timestamp_mjd": data_timestamp_mjd,
            "bad_correlator_numbers": bad_correlator_numbers
        }
        # This print statement is the communication mechanism back to the main pipeline
        print(json.dumps(output))

    except Exception as e:
        # Errors are already logged in the helper function or above, just exit with failure code
        sys.exit(1)

if __name__ == "__main__":
    main()
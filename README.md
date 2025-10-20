# OVRO-LWA Calibration Pipeline

This repository contains the calibration and quality assurance (QA) pipeline for the Owens Valley Radio Observatory Long Wavelength Array (OVRO-LWA). The scripts are designed to process visibility data from the 352-antenna array, perform robust calibration, and generate calibration tables, images and diagnostic reports.

## Overview

The OVRO-LWA is a low-frequency radio interferometer operating from approximately 13 MHz to 87 MHz. This pipeline provides an end-to-end solution for taking raw Measurement Set (MS) data from a single observation and producing calibrated data ready for scientific imaging and analysis. The process includes data preparation, flagging, application of a sky model, delay and bandpass calibration, and automated QA analysis.

A key design principle of this pipeline is automated validation. At each major step, logs and QA plots are generated to ensure that the calibration is performing as expected. The entire process is orchestrated by `calibration_master.py`, and upon completion, results are organized into `successful` or `unsuccessful` directories based on the outcome.

## Key Features

* **End-to-End Automation**: A master script (`calibration_master.py`) orchestrates the entire workflow from raw data to final products.
* **Flexible Execution**: Command-line arguments allow for skipping specific steps, reprocessing from intermediate products (`--input_ms`), and customizing the sky model (`--single_source`) for debugging and analysis.
* **RFI Flagging**: Utilizes the observatory's antenna health database (MNC) and AOFlagger for Radio Frequency Interference (RFI) mitigation.
* **Dynamic Sky-Based Calibration**: Applies a sky model based on A-team calibrators (Cyg A, Vir A, Cas A, Tau A), correcting for the primary beam response at the time of observation.
* **Calibration Table Flagging**: Incorporates a post-processing module to analyze bandpass solutions, identify outlier antennas, flag residual RFI, and interpolate over flagged channels.
* **QA & Diagnostics**:
    * Performs diagnostic delay calibration to monitor digital system health.
    * Generates diagnostic images (per-source spectra, time-resolved scintillation, full-sky zenith) using **WSClean**.
    * Produces plots of delay solutions, calibrator spectra, and calibrator positions.
    * Creates a final **PDF summary report** with all QA plots for easy review.
* **Central Configuration**: A central configuration file (`pipeline_config.py`) controls all parameters, including a dynamic system for finding the most relevant reference calibration table based on LST.

## Data Characteristics

The pipeline is tailored to the specific data format of the OVRO-LWA:

* **Array**: 352-antenna dipole array.
* **Frequency Range**: ~13.4 MHz to ~86.9 MHz (73.5 MHz total bandwidth).
* **Channelization**: 3072 channels of 23.926 kHz each.
* **Data Structure**: Data are typically divided into 16 sub-bands, each containing 192 channels.
* **Integration Time**: 10.031 seconds per integration.
* **File Format**: Each MS file represents a single integration and a single sub-band.
* **Filename Convention**: `YYYYMMDD_HHMMSS_XXMHz.ms`, where the timestamp is the UTC mid-point of the integration and `XXMHz` is the approximate lowest frequency of the sub-band.

## Pipeline Workflow

The calibration process is executed by `calibration_master.py` as a series of sequential steps:

1.  **Context Initialization (`pipeline_utils.py`)**: The pipeline analyzes input filenames to determine the observation's time, duration, and LST. It creates a structured temporary working directory for all outputs.
2.  **Data Preparation (`data_preparation.py`)**: All input MS files are concatenated into a single MS. The `FIELD_ID` column is standardized, and RFI is mitigated using the MNC health database and AOFlagger.
3.  **Sky Model Application (`add_sky_model.py`)**: A sky model is calculated based on bright calibrators, corrected for the primary beam, and inserted into the `MODEL_DATA` column of the MS using the CASA `ft` task.
4.  **Bandpass Calibration (`bandpass_calibration.py`)**: The phase center is shifted to a suitable target (e.g., zenith). A per-channel bandpass solution is derived and undergoes rigorous post-processing to clean up poor solutions before being applied to the data.
5.  **Delay Calibration (`delay_calibration.py`)**: As a **diagnostic step**, this module solves for instrumental delays using only the first integration and higher-frequency sub-bands for speed and accuracy. The results are compared against a reference table to diagnose system health.
6.  **Imaging (`imaging.py`)**: A series of diagnostic images are produced using **WSClean**. The phase center is rotated to each primary calibrator to generate per-source spectral and time-resolved images. Finally, the phase center is moved to zenith for a deep, full-sky image.
7.  **Quality Assurance (`qa.py`)**: This final step analyzes the outputs from the calibration and imaging steps. It measures source fluxes and positions, generates all diagnostic plots, and compiles them into a final PDF summary report.
8.  **Finalization**: The master script cleans up the large intermediate MS file and moves the entire working directory to its final destination under a `successful` or `unsuccessful` folder, named with the processing timestamp.

## Dependencies & Installation

This pipeline relies on a specific software environment, including CASA 6, external astronomical tools, and several Python packages. For those internal to the OVRO-LWA project, these requirements are fulfilled within the Conda environment py38\_orca\_nkosogor.

### Core Dependencies

* **CASA 6.x**: Requires a modular installation of `casatools` and `casatasks`.
* **Python 3.8+**
* **WSClean**: The `wsclean` executable must be in the system's `PATH`.
* **AOFlagger**: The `aoflagger` executable must be in the system's `PATH`.
* **chgcentre**: The `chgcentre` executable must be available.

### Python Packages

The primary environment requires the following packages:

```
numpy
pandas
scipy
astropy
matplotlib
h5py
reportlab
tqdmi
```

### MNC Environment

A separate conda environment is required to query the antenna health database. The name of this environment must be set in `pipeline_config.py`.
```bash
conda create -n development python=3.8
conda activate development
pip install mnc lwa_antpos
```
## Configuration

All pipeline parameters are controlled via the `pipeline_config.py` file. This is the central hub for customizing the pipeline's behavior. Key configurable sections include:

* **Paths**: Base directories for outputs and reference tables, paths to external executables, and data models (flux, beam).
* **Flagging**: Lists of antennas to flag under different conditions (e.g., `ADDITIONAL_BAD_ANTENNAS`, `FALLBACK_BAD_ANTENNAS`).
* **Calibration**: The reference antenna, phase center target, UV ranges, and SNR thresholds.
* **Dynamic Reference Tables**: The `get_reference_table_path` function intelligently finds the best reference calibration table (delay or bandpass) based on the current observation's LST.
* **Imaging**: Default parameters for all WSClean imaging modes.
* **Bandpass Flagging**: Thresholds and window sizes for the sophisticated post-solution flagging algorithms.

## Usage

The pipeline is run from the command line using `calibration_master.py`.

### Basic Execution

To run the full pipeline on a directory of MS files:
```bash
python calibration_master.py /path/to/raw/ms_files/
```
### Common Command-Line Arguments

* `--input_ms /path/to/concat.ms`: Start the pipeline from an existing concatenated MS.
* `--rerun_flagging`: When used with `--input_ms`, forces the flagging sub-steps to run again.
* `--single_source CygA`: Restrict the sky model to only use Cygnus A.
* `--skip_imaging`: Skip the time-consuming WSClean imaging step.
* `--skip_cleanup`: Prevent the deletion of the large concatenated MS.
* `--no_move_results`: Leave the output in the temporary processing directory.
* `-v, --verbose`: Enable more detailed DEBUG-level logging to the console.

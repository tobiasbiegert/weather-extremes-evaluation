# Evaluation Protocols for AI Weather Forecasts and Extremes

Code accompanying the paper **“Can We Trust Evaluations of AI Weather Forecasts for Extremes? How Evaluation Protocols Shape Model Rankings”**.

The repository contains code to compare AI weather prediction models and a physics-based numerical weather prediction model under different evaluation protocols for extreme weather events. The experiments compare HRES and the operational variants of Pangu-Weather, GraphCast, and Aurora within a common setup.

## Overview

The main evaluation quantities implemented here are:

* `MSE`: mean squared error over all forecast cases.
* `owMSE`: MSE restricted to observed monthly record events.
* `twMSE`: threshold-weighted MSE based on the component scores of Taggart (2022).
* `PCRPS`: potential continuous ranked probability score after in-sample EasyUQ recalibration.
* `twPCRPS`: threshold-weighted PCRPS for monthly record thresholds.
* MSE, owMSE and twMSE decompositions into miscalibration (`MCB`), discrimination (`DSC`), and uncertainty (`UNC`).
* Hit, miss, and false-alarm counts and contributions to the twMSE.

## Repository structure

```text
download_2022_data.py
    Download and prepare the WeatherBench 2 forecasts and verification data.

compute_thresholds.py
    Compute monthly ERA5 record thresholds.

evaluate_mse_decomposition.py
    Compute MSE and owMSE scores and their decompositions.

evaluate_taggart_decomposition.py
    Compute twMSE scores, decompositions, and contingency-category contributions.

compute_contingency_tables.py
    Create pooled contingency-statistics and twMSE-share tables.

create_figures.py
    Create the figures shown in the paper and appendix from the computed results.

pcrps/
    compute_pcrps.py
        Run the PCRPS and twPCRPS computation pipeline.

    pcrps_helper.py
        Fit EasyUQ and evaluate PCRPS and twPCRPS.

jobs/
    run_mse_decomposition.sh
    run_taggart_decomposition.sh
    run_pcrps_temporal.sh
        Shell scripts containing the commands used for the experiments.

local_paths.env.example
    Template for local data, threshold, and result directory paths.

environment.yml
    Conda environment containing the required dependencies.
```

## Setup

The code expects local paths to the data, thresholds, and output directories. Create a local paths file from the example:

```bash
cp local_paths.env.example local_paths.env
```

Then edit `local_paths.env` and set the paths for your system.

Install the required Python environment with:

```bash
conda env create -f environment.yml
conda activate weather_extremes
```

## Usage

The shell scripts in `jobs/` contain the commands used for the experiments in the paper. They automatically change to the repository root and load `local_paths.env`.

### Prepare the data

Download and prepare the required WeatherBench 2 forecasts and verification data:

```bash
source local_paths.env
python download_2022_data.py
```

Compute the monthly ERA5 record thresholds:

```bash
source local_paths.env
python compute_thresholds.py
```

### Compute MSE and owMSE results

```bash
bash jobs/run_mse_decomposition.sh
```

### Compute twMSE results

```bash
bash jobs/run_taggart_decomposition.sh
```

### Create contingency and twMSE contribution tables

```bash
source local_paths.env
python compute_contingency_tables.py
```

### Compute PCRPS and twPCRPS results

```bash
bash jobs/run_pcrps_temporal.sh
```

### Create figures

```bash
source local_paths.env
python create_figures.py
```

## Data

The experiments use WeatherBench 2 forecast and verification datasets together with monthly record thresholds derived from ERA5.

## Paper

Tobias Biegert, Sam Allen, Zhongwei Zhang, and Sebastian Lerch (2026).
[Can We Trust Evaluations of AI Weather Forecasts for Extremes? How Evaluation Protocols Shape Model Rankings](https://openreview.net/forum?id=t3mlqVY2dn).
Accepted at the NeurIPS 2026 Trust-AI-Eval Workshop.

### Citation

If you use this code, please cite the accompanying paper:

```bibtex
@inproceedings{
biegert2026can,
title={Can We Trust Evaluations of {AI} Weather Forecasts for Extremes? How Evaluation Protocols Shape Model Rankings},
author={Tobias Biegert and Sam Allen and Zhongwei Zhang and Sebastian Lerch},
booktitle={NeurIPS 2026 Trust-AI-Eval Workshop},
year={2026},
url={https://openreview.net/forum?id=t3mlqVY2dn}
}
```

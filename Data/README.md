## Data Extraction and Processing

This repository contains scripts and notebooks for extracting and processing well log data from LAS and DLIS files, ultimately converting it into CSV format for easy manipulation and analysis.

### Overview

The data is extracted using functions in the 'Data/Raw Files/convert_file_format.py' script. An example of how to use these functions is provided in the Jupyter notebook [From LAS_DLIS to CSV](https://github.com/farah-rabie/GeosciencePlusAI/blob/main/Data/Raw%20Files/From%20LAS_DLIS%20to%20CSV.ipynb).

#### Data Description

The extracted data includes key well log parameters:

- **DEPTH**: The vertical distance below the surface, indicating the depth at which the measurements were taken.
- **BVW** (Bulk Volume of Water): The portion of the bulk volume of the formation occupied by water, important for assessing water saturation and fluid content.
- **KLOGH** (Permeability Log): Represents the permeability of the formation, which measures the ability of the rock to transmit fluids. This is a crucial parameter in reservoir engineering and fluid flow analysis.
- **VSH** (Shale Volume): The fraction of the formation made up of shale or clay, useful for identifying lithologies and estimating shale content.
- **GR** (Gamma Ray): Measures natural gamma radiation, helping to distinguish shale (high GR) from non-shale rocks (low GR), aiding in lithology classification.
- **NPHI** (Neutron Porosity): Measures porosity by interacting with hydrogen atoms, providing insights into the void space in the formation.
- **RHOB** (Bulk Density): The bulk density of the formation, used to calculate porosity and estimate formation characteristics.
- **DT** (Delta T or Sonic Travel Time): Measures acoustic wave travel time through the formation, providing insights into porosity and lithology.
- **PEF** (Photoelectric Factor): Measures the interaction of gamma rays with the formation, used to distinguish rock types, particularly mineral content.
- **RT** (Resistivity): Indicates the resistance of the formation to electrical flow, helping to distinguish between water and hydrocarbon zones.
- **LITHOLOGY**: The rock type (e.g., sandstone, shale, limestone), inferred from other log data as it is not explicitly provided in LAS or DLIS files.

All extracted data is stored in CSV files to facilitate further manipulation. If any data is missing, it has been documented in the file [here](https://github.com/farah-rabie/GeosciencePlusAI/blob/main/Data/data_info.txt). Lithology is not explicitly provided in the LAS or DLIS files. Therefore, it was derived from flag columns within these files and completed using additional information found in the files in the folder [here](https://github.com/farah-rabie/GeosciencePlusAI/tree/main/Data/Completion%20Logs).

#### Data Source

This dataset contains information for seven wells, sourced from the publicly available Volve field dataset, available for download [here](https://www.equinor.com/energy/volve-data-sharing).

### Files

- [convert_file_format](https://github.com/farah-rabie/GeosciencePlusAI/blob/main/Data/Raw%20Files/convert_file_format.py): Python script for data extraction and conversion.
- [From LAS_DLIS to CSV](https://github.com/farah-rabie/GeosciencePlusAI/blob/main/Data/Raw%20Files/From%20LAS_DLIS%20to%20CSV.ipynb): Jupyter notebook showing an example of how to use the script.
- [data_info](https://github.com/farah-rabie/GeosciencePlusAI/blob/main/Data/data_info.txt): File documenting any missing data details.

## Usage

1. Run the `convert_file_format.py` script to extract and convert LAS/DLIS files.
2. Use the `From LAS_DLIS to CSV.ipynb` notebook to see an example of the data extraction process.
3. Check the `data_info.txt` file for any missing data details.

## Licence

This repository is licensed under the MIT Licence. See `LICENSE` for more information.


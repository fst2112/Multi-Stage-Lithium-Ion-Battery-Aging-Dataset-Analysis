
# Multi-Stage Lithium-Ion Battery Aging Dataset Analysis

This repository contains code and resources for analyzing the aging dataset of lithium-ion batteries, as detailed in the Paper [Multi-Stage Lithium-Ion Battery Aging Dataset](link_to_publication). The primary objectives of this project include data loading, filtering and correcting outliers, extracting essential features, and creating visual representations of the data.

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
Lithium-ion batteries are critical components in many modern devices, from smartphones to electric vehicles. Understanding their aging process is vital for improving performance and longevity. This repository provides tools to:
- Load and preprocess battery aging data.
- Extract key features such as capacity, DCIR (Direct Current Internal Resistance), and OCV (Open Circuit Voltage).
- Create figures to visualize the data and extracted features.

## Data
The dataset used in this analysis can be found [here](https://doi.org/10.6084/m9.figshare.25975315). It consists of multiple files capturing various aspects of battery aging.

### Directory Structure
- `data/`: Contains the files that are created during analyses, e.g. table of capacities.

## Code Structure
- `notebooks/`: Jupyter notebooks for interactive analysis.
  - `basic_analyses.ipynb`: Notebook for loading and exploring the dataset.

- `src/`: Python scripts for modular code execution.
  - `data_import.py`: Script for loading data files.
  - `feature_extraction.py`: Script for feature extraction.


## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/fst2112/multi-stage-lithium-ion-battery-aging-dataset-analysis.git
   cd multi-stage-lithium-ion-battery-aging-dataset-analysis
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks or scripts as needed for your analysis.

## Contributing
Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) for more information.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

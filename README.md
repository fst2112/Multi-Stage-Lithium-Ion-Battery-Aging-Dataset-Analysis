
# Multi-Stage Lithium-Ion Battery Aging Dataset Analysis

This repository contains code and resources for analyzing the aging dataset of lithium-ion batteries, as detailed in the [Nature Scientific Data Descriptor](link_to_publication). The primary objectives of this project include data loading, filtering and correcting outliers, extracting essential features, and creating visual representations of the data.

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
- Filter and correct outliers in the dataset.
- Extract key features such as capacity, DCIR (Direct Current Internal Resistance), and OCV (Open Circuit Voltage).
- Create figures to visualize the data and extracted features.

## Data
The dataset used in this analysis can be found [here](link_to_dataset). It consists of multiple files capturing various aspects of battery aging.

### Directory Structure
- `data/raw/`: Contains the raw dataset files.
- `data/processed/`: Contains processed data files ready for analysis.

## Code Structure
- `notebooks/`: Jupyter notebooks for interactive analysis.
  - `data_loading.ipynb`: Notebook for loading and exploring the dataset.
  - `outlier_correction.ipynb`: Notebook for filtering and correcting outliers.
  - `feature_extraction.ipynb`: Notebook for extracting features from the dataset.
  - `visualization.ipynb`: Notebook for creating visualizations.

- `src/`: Python scripts for modular code execution.
  - `data_loading.py`: Script for loading data files.
  - `outlier_correction.py`: Script for filtering and correcting outliers.
  - `feature_extraction.py`: Script for feature extraction.
  - `visualization.py`: Script for creating figures.

- `figures/`: Directory to save generated figures.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lithium-ion-battery-aging-dataset-analysis.git
   cd lithium-ion-battery-aging-dataset-analysis
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

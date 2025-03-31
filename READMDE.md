![MIDOG 2025 logo](docs/MIDOG_2025_logo.jpg)

# MIDOG 2025 Challenge

Welcome to the official repository of the MIDOG 2025 Challenge (3rd iteration of the MIDOG Challenge series).

## Overview

This repository contains example implementations and utilities for participating in the MIDOG 2025 Challenge. The challenge focuses on advancing robust and automated mitosis detection across histopathology images of different tumor types.

🔗 **Official Challenge Website:** [MIDOG 2025](https://midog2025.deepmicroscopy.org/)

For challenge registration and detailed information about the competition, rules, and deadlines, please visit the official website.

## Community 

<a href="https://discord.gg/xEuqXjMqTk">
<img src="https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white" alt="Join our Discord server">
</a>

Join our Discord community to connect with other participants, ask questions, and share insights!

## Repository Contents

This repository includes:

- `MIDOG2025_01_Exploratory_Data_Analysis.ipynb`: A notebook introducing the MIDOGpp dataset with exploratory data analysis
- `MIDOG2025_02_Simple_Training.ipynb`: Example implementation of a basic training pipeline
- `requirements.txt`: Required Python packages for running the notebooks
- `utils/`: Utility functions and helper scripts

⚠️ **Important Note**: The provided notebooks are meant to serve as examples and starting points. We strongly encourage participants to be creative and develop their own implementations to achieve better results.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Openslide 

### Installation

1. Clone this repository:
```bash
git clone https://github.com/your-organization/midog2025-challenge.git
cd midog2025-challenge
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv midog_env
source midog_env/bin/activate  # On Windows: midog_env\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset Access

The [MIDOGpp](https://github.com/DeepMicroscopy/MIDOGpp) dataset used in this repository is available online and can be downloaded with the `download_MIDOGpp.py` script. 

## Example Notebooks

### 1. Dataset Exploration
The first notebook (`MIDOG2025_01_Exploratory_Data_Analysis.ipynb`) provides:
- Dataset structure overview
- Statistical analysis of mitosis distributions
- Visualization of sample images and annotations


### 2. Basic Training Implementation
The second notebook (`MIDOG2025_02_Simple_Training.ipynb`) demonstrates:
- How to prepare the dataset for training
- How to use this repository to create a model 
- A simple model training setup
- Visualizations of predictions


## Additional Resources

- [Challenge Overview](https://zenodo.org/records/15077361)
- [Datasets](https://midog2025.deepmicroscopy.org/datasets/)
- [Previous MIDOG 2022 Challenge Publication](https://www.sciencedirect.com/science/article/pii/S136184152400080X)



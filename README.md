# SF-RX

This repository contains the code used to reproduce the results from our research paper **"SF-Rx: A Multi-output Deep Neural Network-Based Framework Predicting Drug-Drug Interaction under Realistic Conditions for Safe Prescription"**. The software is organized into four folders, each corresponding to a specific task discussed in the paper. Below are detailed instructions and notes about the code structure and data.

## Folder Structure

- **[SF_RX_MODEL]**: Code and models for the **SF-RX** implementation, optimized for GPU environments.
- **[GNNs]**: Code for training GNNs and transformer models used in the paper.
- **[FEDERATED_LEARNING]**: Federated learning experiments with GPU parallelism.
- **[PERMUTATION_TEST]**: Permutation test for distributional shifts of scaffold structures.

## Data
- All required data is located in the `data` folder within each directory.
- For large files, Google Drive links are provided in the respective folders.
- **Note**: The original results in the paper were generated using proprietary DrugBank data, which cannot be shared. Instead, we created **toy datasets** by combining publicly available data from DrugBank and PDR.
- **Note**: **Drugs.com data was used with retrospective permission from Drugs.com. Original source data are not publicly available**
  
## Key Features
### GPU Optimization
- **SF-RX Model** and **Federated Learning** tasks are designed to run on GPU environments.
- Federated Learning assumes **4 GPUs** for parallel execution due to the computationally intensive nature of the FL experiments.
- To modify GPU settings, update the `parallelism` section in [`FEDERATED_LEARNING/experiment.py`](FEDERATED_LEARNING/experiment.py).

### Dependencies
- All software and library version requirements are listed in [`dependencies.txt`](dependencies.txt).

For any questions or issues, feel free to reach out to us via [shbae@drnoahbiotech.com], [dekim@drnoahbiotech.com], [jhyu@drnoahbiotech.com].

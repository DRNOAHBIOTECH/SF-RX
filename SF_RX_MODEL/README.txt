0. Data Loading

Due to GitHub's storage policy, the feature table for federated learning is stored in Google Drive. You can access it via the following link:

https://drive.google.com/drive/folders/10LexYtkgNTcyKEJFiLuF9hp8Uv-Z7VDJ?usp=sharing

Please download six parquet files from the link.

After downloading, place these files in the "SF_RX_MODEL/data" directory.


1. Run Python Scripts to Learn Model.

Execute the provided Python scripts to learn the SF-RX baseline and adjusted models.
Examples:

# python3 baseline_train.py --lv 2 --device 0
# python3 adjusted_train.py --lv 4 --device 2

The results including models and metrics by fold will be saved in the "SF_RX_MODEL/result" directory.

2. Run Python Scripts to calculate Gini Impurity.

Execute the provided Python scripts to learn the SF-RX baseline and adjusted models.
Examples:

# python3 baseline_gini.py --lv 2 --n_inference 30 --device 0
# python3 adjusted_gini.py --lv 4 --n_inference 30 --device 2

The results including inferences and plots by fold will be saved in the "SF_RX_MODEL/result/uncertainty" directory.

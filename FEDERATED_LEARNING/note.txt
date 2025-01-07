0. Data Loading

Due to GitHub's storage policy, the feature table for federated learning is stored in Google Drive. You can access it via the following link:

https://drive.google.com/drive/folders/1jetBBw3VaoGHm3oRUgnrPaaXWn2NfL3-?usp=sharing

Please download the following files from the link:

	1) drugscom_X.parquet
	2) pdr_X.parquet

After downloading, place these files in the "FEDERATED_LEARNING/data" directory.


1. Run Shell Scripts to Perform Experiments

Execute the provided shell scripts to run the federated learning experiments. The scripts will automatically perform the necessary computations and save the results.

# ./run_exp_cond1.sh
# ./run_exp_cond2.sh

The results will be saved in the "FEDERATED_LEARNING/result" directory.

If you encounter a permission issue try below:

# chmod +x *.sh


2. Analyze the Results

Once the experiments are complete, open and execute the Jupyter Notebook "check_results.ipynb" to analyze the results.
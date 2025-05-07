0. Data Loading

Due to GitHub's storage policy, the feature table for federated learning is stored in Google Drive. You can access it via the following link:

https://drive.google.com/drive/folders/1maQY7h8i80baqwbvQNaE4-jR8t8xypvf?usp=drive_link

Please download parquet files from the link.

After downloading, place these files in the "FEDERATED_LEARNING/data" directory.


1. Run Shell Scripts to Perform Experiments

Execute the provided shell scripts to run the federated learning experiments. The scripts will automatically perform the necessary computations and save the results.

# ./run_experiment.sh

The results will be saved in the "FEDERATED_LEARNING/result" directory.

If you encounter a permission issue try below:

# chmod +x *.sh


2. Analyze the Results

Once the experiments are complete, open and execute the Jupyter Notebook "check_results.ipynb" to analyze the results.

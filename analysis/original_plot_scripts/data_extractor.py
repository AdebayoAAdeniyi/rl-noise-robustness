import numpy as np
import os# The run directory held the .npz metric arrays this script reads. It was a
# cluster scratch path on the machine these plots were produced on; set
# RL_NOISE_RUN_DIR to wherever your own .npz files are.
working_dir = os.environ.get("RL_NOISE_RUN_DIR", ".")
os.chdir(working_dir)

files = os.listdir()
# Find all .npz files in the current directory
npz_files = [file for file in files if file.endswith(".npz")]

# Create the output directory if it doesn't exist
exp_results_dir = "expected_output_data"
os.makedirs(exp_results_dir, exist_ok=True)

# Process each .npz file
for file_name in npz_files:
    # Load the data from the .npz file
    data = np.load(file_name)

    # Extract the desired data
    results_exp_ret = data["results_exp_ret"]

    # Construct the save filename
    save_filename = os.path.join(exp_results_dir, f"exp_return_{file_name}")

    # Save the extracted data to a new .npz file
    np.savez(save_filename, results_exp_ret=results_exp_ret)
    print(f"Results saved to {save_filename}")


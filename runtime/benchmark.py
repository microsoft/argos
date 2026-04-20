import os
import random
import subprocess
import time

# Configuration
PROJECT_DIR = "argos"
DATASET_DIR = "argos/datasets/KPI"
timestamp = str(int(time.time()))
RESULT_DIR = os.path.join(PROJECT_DIR, "results", "mutation", timestamp)
os.makedirs(RESULT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(RESULT_DIR, "selected_datasets.txt")
LOG_FILE = os.path.join(RESULT_DIR, "benchmark_results.txt")
NUM_SAMPLES = 10

# Get all CSV files in the directory
dataset_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")]

KPI_metrics = [
    "1c35dbf57f55f5e4",
    # "da403e4e3f87c9e0",
    # "07927a9a18fa19ae",
    # "88cf3a776ba00e7c",
    # "a40b1df87e3f1c87"
]
# Select 10 random datasets
# selected_datasets = random.sample(dataset_files, NUM_SAMPLES)
selected_datasets = []
for file in dataset_files:
    # Check if the file name contains any of the KPI metrics
    if any(metric in file for metric in KPI_metrics):
        selected_datasets.append(file)
assert len(selected_datasets) == len(KPI_metrics), f"Expected {len(KPI_metrics)} datasets, but found {len(selected_datasets)}."

# Save selected datasets to a file
with open(OUTPUT_FILE, "w") as f:
    for dataset in selected_datasets:
        f.write(dataset + "\n")

# Run benchmark for each selected dataset
results = []

for dataset in selected_datasets:
    dataset_path = os.path.join(DATASET_DIR, dataset)
    command = [
        "python3", f"{PROJECT_DIR}/driver.py", 
        "--dataset_path", dataset_path, 
        "--mode", "train-evolution", 
        # "--mode", "train-evolution",
        "--result_path", RESULT_DIR, 
        "--chunk_size", "2500", 
        "--dataset_mode", "one-by-one", 
        "--llm_engine", "gpt-4o", 
        "--p_cores", "6",
        "--rule_per_group", "2"
    ]
    
    try:
        subprocess.run(command, check=True)
        results.append(f"SUCCESS: {dataset}")
    except subprocess.CalledProcessError as e:
        results.append(f"ERROR: {dataset} - {e}")

# Save results to log file
with open(LOG_FILE, "w") as f:
    for result in results:
        f.write(result + "\n")

print(f"Benchmark completed. Selected datasets saved in {OUTPUT_FILE}.")
print(f"Results logged in {LOG_FILE}.")

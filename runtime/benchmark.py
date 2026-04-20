import os
import subprocess
import time

PROJECT_DIR = os.environ["PROJECT_DIR"]
DATASET_PATH = (
    os.environ["DATASET_PATH"]
)
MODES = ["train-LLM-only-parallel", "train-evolution"]
MAX_ITER = 5

timestamp = str(int(time.time()))
RESULT_DIR = os.path.join(PROJECT_DIR, "test_run_out", "benchmark", timestamp)
os.makedirs(RESULT_DIR, exist_ok=True)

LOG_FILE = os.path.join(RESULT_DIR, "benchmark_results.txt")

results = []

for mode in MODES:
    mode_result_dir = os.path.join(RESULT_DIR, mode)
    os.makedirs(mode_result_dir, exist_ok=True)

    command = [
        "python3", f"{PROJECT_DIR}/driver.py",
        "--dataset_path", DATASET_PATH,
        "--mode", mode,
        "--result_path", mode_result_dir,
        "--chunk_size", "2500",
        "--dataset_mode", "one-by-one",
        "--llm_engine", "gpt-4o",
        "--p_cores", "6",
        "--rule_per_group", "2",
        "--top_k", "6",
        "--max_iter", str(MAX_ITER),
        "--timeout", "60",
    ]

    print(f"\n=== Running mode {mode} ===")
    print("Command:", " ".join(command))
    try:
        subprocess.run(command, check=True)
        results.append(f"SUCCESS: {mode}")
    except subprocess.CalledProcessError as e:
        results.append(f"ERROR: {mode} - {e}")

with open(LOG_FILE, "w") as f:
    for result in results:
        f.write(result + "\n")

print(f"\nBenchmark completed. Results logged in {LOG_FILE}.")
for r in results:
    print(r)

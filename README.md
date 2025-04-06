## Overview

Argos is a time series anomaly detection system that autonomously generates explainable anomaly rules using LLMs.

## Setup and Installation

### Python Environment

Argos is a python-based framework.
It is necessary for the users to install required python packages before using Argos.
We recommend installing packages in a conda virtual environment.
Follow the below steps for setup.

```
cd argos
conda create -n argos python=3.10
conda activate argos
pip install -r requirements.txt
```

### LLM API

Argos also requires a valid LLM API to run.
Currently Argos relies on AzureOpenAI client to interact with the LLM.
Set the following environment variables before use.

```
export OPENAI_AZURE_ENDPOINT=YOUR_AZURE_ENDPOINT
export OPENAI_AZURE_API_KEY=YOUR_API_KEY
export OPENAI_AZURE_API_VERSION=YOUR_API_VERSION
```

## Datasets

Argos assumes that a given time-series dataset will have the following structure.
There can be multiple metrics inside a dataset, and each metric can contain multiple time-series data in csv files.
Each csv file has three columns `value,label,index`, where `value` is the metric value, `label` is the anomaly label, and `index` is the index of the current data point.

```
  dataset_A
  ├── metric_1
  │   ├── time_series_1.csv
  │   ├── time_series_2.csv
  │  ...
  │   └── time_series_n.csv
  └── metric_2
  │   ├── time_series_1.csv
  │  ...
```

Argos provides a script that preprocesses a time-series dataset into the above format in `argos/utility/generate_csv.py`. 
Currently supported datasets are KPI dataset and Yahoo dataset.

### Dataset Mode
Argos supports two dataset modes: one-for-one and one-for-all. 

The one-for-one mode trains a set of anomaly rules for one time-series data.
The one-for-all mode trains a set of anomaly rules for one metric or a group of metrics.

The one-for-one mode is suitable for datasets with a
large number of continuous metrics while the one-for-all mode is useful when the user
wants to train a set of anomaly detection rules that generalize across multiple metrics or multiple time-series data. 


## Repo Structure

```
├── argos
│   ├── agent			  # Agent implementation
│   ├── baseline		  # Baseline implementation
│   ├── common		      # Common helper functions
│   ├── datasets		      # Dataset implementation
│   ├── eval_metrics      # Evaluation metrics implementation
│   ├── runtime			  # Training engine implementation
│   ├── selector		  # Top-k rule selector implementation
│   ├── utility		  	  # Scripts for processing datasets and results
|
│   └── driver.py
```

## Getting Started

Argos comes with two modes: with or without Aggregator. 
If you are using Argos without Aggregator, that means Argos will rely solely on LLM agents to train anomaly rules on the full training dataset.

For either case, you can run Argos through a driver program (`driver.py`) provided.

### Argos w/o Aggregator

For example, if you would like to train anomaly rules without Aggregator, using a chunk size of 2500 for each data chunk in the training iteration, with one-for-one dataset mode and top-5 rule selection, run the following command:

```
python3 driver.py --dataset_path=$dataset_path --mode=train-LLM-only --result_path=$result_path --chunk_size=2500 --top_k=5 --dataset_mode="one-by-one"
```

This will produce a list of rules for each training iteration along with training and testing performance in the `$result_path` folder.

### Argos w/ Aggregator

If you would like to proceed with Argos w/ Aggregator, that means you have an existing anomaly detection model.

First, you need to collect incorrect examples from the existing model and arrange them in a folder following the steps described here (TODO).

The training with Aggregator comes with two passes, one for training a set of rules to fix false positives and one for training a set of rules to fix false negatives.

For example, if you would like to train anomaly rules to fix false positives with Aggregator, using a chunk size of 100, with one-for-one dataset mode and top-1 rule selection, run the following command:

```
python3 driver.py --dataset_path=$dataset_path --mode=train-combined-fp --result_path=$result_path --chunk_size=1000 --top_k=1 --dataset_mode="one-by-one" --model_res_path=$model_res_path
```

Note that you need to additionally supply the processed model result path from your existing model in the argument.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

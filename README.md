Argos: Agentic Time-Series Anomaly Detection with Autonomous Rule Generation via LLMs
=====================================================================================

Overview
--------

Argos is an agentic system for autonomously generating explainable time-series anomaly rules using LLMs.

### What can Argos do

Argos employs LLMs to autonomously generate explainable and reproducible rules.
These rules can be used by humans or other systems to detect time-series anomalies in cloud infrastructure.

A detailed discussion of Argos, including how it was developed and tested, can be found in [Argos paper][paper link].

### Intended uses

- Argos is best suited for suggesting explainable and reproducible anomaly rules in monitoring systems for humans to review and apply.
- Argos is being shared with the research community to facilitate reproduction of our results and foster further research in this area.
- Argos is intended to be used by domain experts who are independently capable of evaluating the quality of outputs before acting on them.

### Out-of-scope uses

- Argos is not well suited for being deployed into real-world infrastructure to replace existing anomaly detection or monitoring systems directly.
We do not recommend using Argos in commercial or real-world applications without further testing and development. It is being released for research purposes.
- Argos was not designed or evaluated for all possible downstream purposes. Developers should consider its inherent limitations as they select use cases,
and evaluate and mitigate for accuracy, safety, and fairness concerns specific to each intended downstream use.
- Argos should not be used in highly regulated domains where inaccurate outputs could suggest actions that lead to injury or negatively impact an individual's legal, financial, or life opportunities.
We do not recommend using Argos in the context of high-risk decision making (e.g., in law enforcement, legal, finance, or healthcare).


Setup and Installation
----------------------

To begin using Argos, please checkout the code at https://github.com/microsoft/argos.

### Python Environment

Argos is a python-based framework.
It is necessary for the users to install required python packages before using Argos.
We recommend installing packages in a conda virtual environment.
Follow the below steps for setup.

```sh
cd argos
conda create -n argos python=3.10
conda activate argos
pip install -r requirements.txt
```

### LLM API

Argos also requires a valid LLM API to run.
Currently Argos relies on Azure OpenAI services to interact with the LLM.
Set the following environment variables before use.

```sh
export OPENAI_AZURE_ENDPOINT=YOUR_AZURE_ENDPOINT
export OPENAI_AZURE_API_KEY=YOUR_API_KEY
export OPENAI_AZURE_API_VERSION=YOUR_API_VERSION
```

### Datasets

Argos assumes that a given time-series dataset will have the following structure.
There can be multiple metrics inside a dataset, and each metric can contain multiple time-series data in csv files.
Each csv file has three columns `value,label,index`, where `value` is the metric value, `label` is the anomaly label, and `index` is the index of the current data point.

```sh
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

#### Dataset Mode

Argos supports two dataset modes: one-for-one and one-for-all.

The one-for-one mode trains a set of anomaly rules for one time-series data.
The one-for-all mode trains a set of anomaly rules for one metric or a group of metrics.

The one-for-one mode is suitable for datasets with a
large number of continuous metrics while the one-for-all mode is useful when the user
wants to train a set of anomaly detection rules that generalize across multiple metrics or multiple time-series data.

### Repo Structure

```sh
argos
├── agent           # Agent implementation
├── baseline        # Baseline implementation
├── common          # Common helper functions
├── datasets        # Dataset implementation
├── eval_metrics    # Evaluation metrics implementation
├── runtime         # Training engine implementation
├── selector        # Top-k rule selector implementation
├── utility         # Scripts for processing datasets and results
└── driver.py
```


Getting Started
---------------

Argos comes with two modes: with or without Aggregator.
If you are using Argos without Aggregator, that means Argos will rely solely on LLM agents to train anomaly rules on the full training dataset.

For either case, you can run Argos through a driver program (`driver.py`) provided.

### Argos w/o Aggregator

For example, if you would like to train anomaly rules without Aggregator, using a chunk size of 2500 for each data chunk in the training iteration, with one-for-one dataset mode and top-5 rule selection, run the following command:

```sh
python3 driver.py \
  --dataset_path=$dataset_path \
  --mode=train-LLM-only \
  --result_path=$result_path \
  --chunk_size=2500 \
  --top_k=5 \
  --dataset_mode="one-by-one"
```

This will produce a list of rules for each training iteration along with training and testing performance in the `$result_path` folder.

### Argos w/ Aggregator

If you would like to proceed with Argos w/ Aggregator, that means you have an existing anomaly detection model.

First, you need to collect incorrect examples from the existing model and arrange them in a folder following the steps described here (TODO).

The training with Aggregator comes with two passes, one for training a set of rules to fix false positives and one for training a set of rules to fix false negatives.

For example, if you would like to train anomaly rules to fix false positives with Aggregator, using a chunk size of 100, with one-for-one dataset mode and top-1 rule selection, run the following command:

```sh
python3 driver.py \
  --dataset_path=$dataset_path \
  --mode=train-combined-fp \
  --result_path=$result_path \
  --chunk_size=1000 \
  --top_k=1 \
  --dataset_mode="one-by-one" \
  --model_res_path=$model_res_path
```

Note that you need to additionally supply the processed model result path from your existing model in the argument.


Evaluation
----------

Argos was evaluated on its ability to detect time-series anomalies accurately in collected anomaly detection datasets.
A detailed discussion of our evaluation methods and results can be found in [Argos paper][paper link].

### Evaluation Methods

We used event-based F1 score with point adjustment to measure Argos's performance.

We compared the performance of Argos against 5 DL based methods (AnomalyTransformer, AutoRegression, FCVAE, LSTMAD, TFAD).
and 2 LLM based methods (LLMAD, SigLLM) using [KPI](https://github.com/NetManAIOps/KPI-Anomaly-Detection), [Yahoo](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70), and internally collected anomaly detection datasets.

The models used for evaluation were Azure OpenAI GPT-3.5, Azure OpenAI GPT-4-32k, and Azure OpenAI GPT-4o.
For more on these specific models, please see https://azure.microsoft.com/en-us/products/ai-services/openai-service.

Results may vary if Argos is used with a different model, based on their unique design, configuration and training.

### Evaluation Results

At a high level, we found that Argos outperformed state-of-the-art methods, increasing F1 scores by up to __28.3%__.


Limitations
-----------

- Argos was developed for research and experimental purposes. Further testing and validation are needed before considering its application in commercial or real-world scenarios.
- Argos was designed and tested using the English language. Performance in other languages may vary and should be assessed by someone who is both an expert in the expected outputs and a native speaker of that language.
- Outputs generated by AI may include factual errors, fabrication, or speculation. Users are responsible for assessing the accuracy of generated content. All decisions leveraging outputs of the system should be made with human oversight and not be based solely on system outputs.
- Argos inherits any biases, errors, or omissions produced by its base model. Developers are advised to choose an appropriate base LLM carefully, depending on the intended use case.
- Argos uses the Azure OpenAI GPT-4-32k/GPT4-4o models. See https://azure.microsoft.com/en-us/products/ai-services/openai-service to understand the capabilities and limitations of this model.
- Argos inherits any biases, errors, or omissions characteristic of its training data, which may be amplified by any AI-generated interpretations.


Best Practices
--------------

We strongly encourage users to use LLMs/MLLMs that support robust Responsible AI mitigations, such as Azure Open AI (AOAI) services.
Such services continually update their safety and RAI mitigations with the latest industry standards for responsible use.
For more on AOAI's best practices when employing foundations models for scripts and applications:
- [Blog post on responsible AI features in AOAI that were presented at Ignite 2023](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/announcing-new-ai-safety-amp-responsible-ai-features-in-azure/ba-p/3983686)
- [Overview of Responsible AI practices for Azure OpenAI models](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/overview)
- [Azure OpenAI Transparency Note](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/transparency-note)
- [OpenAI's Usage policies](https://openai.com/policies/usage-policies)
- [Azure OpenAI's Code of Conduct](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/code-of-conduct)

Users are responsible for sourcing their datasets legally and ethically. This could include securing appropriate copy rights,
ensuring consent for use of audio/images, and/or the anonymization of data prior to use in research.

Users are reminded to be mindful of data privacy concerns and are encouraged to review the privacy policies associated with any models and data storage solutions interfacing with Argos.

It is the user's responsibility to ensure that the use of Argos complies with relevant data protection regulations and organizational guidelines.


Contributing
------------

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Contact

We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected/offensive behavior in our technology,
please contact us at [argos-project@microsoft.com](mailto:argos-project@microsoft.com).
If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations.


Citations
---------

To cite Argos in your publications:

```bib
@article{argos,
  title={Argos: Agentic Time-Series Anomaly Detection with Autonomous Rule Generation via Large Language Models},
  author={Gu, Yile and Xiong, Yifan and Mace, Jonathan and Jiang, Yuting and Hu, Yigong and Kasikci, Baris and Cheng, Peng},
  journal={arXiv preprint arXiv:2501.14170},
  year={2025}
}
```

[paper link]: https://arxiv.org/pdf/2501.14170


License
-------
This project is licensed under the terms of the [MIT license](./LICENSE).


Trademarks
----------

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

# DRAGIN


## 📌 ATTENTION: A Critical Pinned Message for Reproduction

We sincerely want you to have a smooth experience reproducing our work. Most issues users encounter stem from two primary sources. To save you time and **frustration**, we **strongly urge** you to read this *before* running inference.

 1.  **Code Modifications:** Please do not make any changes to the code, no matter how small.
 2.  **Incomplete Elasticsearch (ES) Index:** This is the most common and difficult-to-diagnose problem, which is detailed below.

-----

 ### ⚠️ **MANDATORY CHECK: Verify Your Elasticsearch Index**

 After building the Wikipedia Index, and **before** you start the inference step, you **must** perform the following check to ensure the index is fully and correctly built.

 #### Check the Index Status then Verify the Output

 Run the following command in your terminal:

 ```bash
 curl -X GET "localhost:9200/_cat/indices?v"
 ```



A **correctly and completely built** index will look *exactly* like this. Pay close attention to the `docs.count` (\~21 million) and `store.size` (\~11.2GB):

 ```
 health status index uuid                  pri rep docs.count docs.deleted store.size pri.store.size dataset.size
 yellow open   wiki  w3L9eKIMT4m8XZN39JGn5w   1   1   21015324            0     11.2gb         11.2gb         11.2gb
 ```

If your `docs.count` is in the thousands (e.g., 10,000) instead of 21 million, your index is **incomplete**, and your reproduction will fail.



 ### The "Silent Failure" Warning

This is the most critical part, and the source of most indexing failures:

   * **ES Must Be Running:** Elasticsearch *must* remain running in the background at all times.
   * **Disk Space is Key:** You must have **at least 10%** of your total disk space free *after* accounting for the 11.2GB index.
   * **THE SILENT KILLER:** If your disk's free space drops too low (often below 10% or 5%), **Elasticsearch will automatically stop indexing**. It will do this **silently**, with **no errors and no logs** to indicate it has stopped.

This silent failure is why many reproductions go wrong. You may *think* the index is built, but it only processed a tiny fraction of the 21 million articles.

**Please, check your index and disk space carefully before proceeding!**

## 📌 Important Notes on Reproducibility

We are fully committed to helping you reproduce the results from our paper. We will respond to any email or issue regarding reproducibility to help you debug and replicate our main experimental findings.

Through extensive experience in assisting users, we've identified a common and particularly challenging scenario: when reproduced results are only slightly lower than those reported in the paper (e.g., a 2-4 point difference). Tracing the source of such minor deviations can be difficult.

**Based on our experience, if your results look different from ours, it's usually because of one of the four things listed below.** Each of these is a common issue that we've helped multiple users identify and fix:

------


### 1. Code Modifications (Especially for Single-GPU Setups)



#### **!!! Please do not make any modifications to the code!!!**

To correctly reproduce the results reported in our paper, you **must run the code as-is, without any alterations**.

All experiments in our paper were conducted on a multi-GPU setup. Therefore, to replicate the reported results, the program **must be run with exactly two GPUs**. We have observed that modifications made to enable single-GPU execution will lead to results that do not align with our paper, even if the code runs without errors.



### 2. Incomplete Elasticsearch Indexing

To reproduce the full results, please ensure that the entire corpus is completely indexed in Elasticsearch.

This process can be time-consuming (often taking several hours) and does not display a progress bar. It is highly susceptible to interruption. If the process is disrupted for any reason (e.g., network issues, system sleep), the resulting Elasticsearch index will be incomplete.

An incomplete index leads to a subtle drop in performance. This can be difficult to detect, as the base model's performance without RAG is already strong, which can mask the impact of a partially indexed corpus.



### 3. Using a Different Prompt Template



The LLaMA-2 model is highly sensitive to the prompt template. Using a different template, even if it seems to convey the same meaning, can cause significant performance change. Please ensure you are using the exact prompt template provided in our codebase.



### 4. Custom Evaluation Scripts or Dependency Issues



**Problem:** We found that some users replaced our evaluation script with their own, often due to a dependency conflict with `beir==1.0.1` in our previous `requirements.txt`. This would lead to significant differences in the final results.

**!!! This dependency issue was a bug in a previous version of our repository and has been fully resolved as of our October 13th update!!!**

To ensure a clean setup and avoid any conflicts, we strongly recommend the following steps:

1. **Delete or rename** your old local code directory to prevent file conflicts.
2. Clone the latest version of this repository using `git clone`.
3. Create a new virtual environment and install dependencies using the updated `requirements.txt` file from the new repository.



### New Detailed Reproduction Guide!

To further assist you, we have uploaded a comprehensive, step-by-step walkthrough to a new file:

- **[detailed_reproducibility_guide.md](https://github.com/oneal2000/DRAGIN/tree/main/detailed_reproducibility_guide.md)**

**This guide is significantly more detailed than a typical README. It provides command-by-command instructions and even includes the exact terminal output you should expect at each stage of a successful run.**



### Still Having Trouble?

The troubleshooting points listed above are a direct result of our email correspondence with the community. Each potential issue was identified after helping multiple researchers (more than three for each case) who were working to reproduce our results.

This experience has made it clear to me that even when all code for baselines and methods is made public, subtle differences in hardware, software environments, and other local factors can still lead to slight variations in results. We understand this can be frustrating.

**Therefore, if you have meticulously followed all the advice above and consulted our detailed guide but still encounter discrepancies, please do not hesitate to email us directly! We are more than happy to provide one-on-one guidance to help resolve the issue.**







## 📢 October 13. 2025 Update:


We have fixed several bugs in this version to improve usability and reproducibility:

**Bug Fixes:**

1. **Evaluation Script Renamed:** `evaluate.py` has been renamed to `eval.py`. In previous versions, users had to manually make this change to run the script.
2. **Dependency Version Update:** The `elasticsearch` version in `requirements.txt` has been updated to `7.9.1` to resolve potential errors.
3. **Model Version Update:** The `llama2-7b` and `llama2-13b` models have been switched to their Hugging Face (HF) versions. This change addresses potential errors and **reproducibility issues** found in the previous versions.

------



### ⚠️ **October 13. 2025: Important Notice on Code Modification**

We welcome the community to use DRAGIN as a foundation for new research. **Please feel free to modify and adapt the code for your projects. There are no licensing restrictions on any use.**

**However, if your goal is to reproduce the results in the main experiment table of our paper, please do not make any modifications to the code!**

This recommendation is based on our experience assisting users who initially faced issues with the reproducibility of our main experiment table, which were ultimately traced back to local code alterations.

To correctly reproduce the results reported in our paper, **in any case, under any circumstances, do not make any changes to the code!!!**

For example, if you think you need to modify the code to run it on a Single GPU, **absolutely do not change it**. If you change it, the results will not be reproducible. You **must use two GPUs** and run the code without making any modifications.

All experiments reported in our paper were conducted on multiple GPUs. Therefore, to reproduce the reported results, the program **must be run with exactly two GPUs**.



## **📢 September 16, 2025 Update:**

Thank you for your interest in our paper and for visiting this repository. We sincerely appreciate everyone who has read our work and tried to run it.

We've received some inquiries regarding the configuration of **Elasticsearch**, and we understand this can be a difficult step. I still remember when I was first getting into RAG two years ago, starting with Zhengbao Jiang's classic paper, FLARE. I also spent a few days troubleshooting Elasticsearch, running into all sorts of unexpected errors. A fantastic resource I discovered back then and still use today is the official Elasticsearch discussion forum:

https://discuss.elastic.co/

The forum is highly active, and you can find solutions to almost any error you encounter there. We hope this helps you get started more smoothly!

------
#### Very Important:

Another reminder for reproducing our paper: In order to reproduce the full results for our paper, please ensure that the entire corpus is completely indexed in Elasticsearch. This can take a significant amount of time, often up to half a day, and there is no progress bar.

We've observed in practice that this lengthy indexing process is highly susceptible to interruptions. If the process is disrupted for any reason, the resulting Elasticsearch index will be incomplete.

If the full corpus is not indexed, your reproduced results will be slightly lower. **This subtle drop in performance can be difficult to detect**, as it may only be a few percentage points and is not substantial. This is because the performance of the model **without RAG** is already strong, which can mask the effect of an incomplete index.

------

Additionally, we selected 1000 questions from each dataset. We have received some inquiries regarding this selection process. **Specifically, we chose the top 1000 questions**. Taking the top 1000 is as reasonable as randomly sampling 1000, and it offers more stability by avoiding the influence of random factors.

Furthermore, our method has been independently reproduced by others. 
Several published papers have also independently reproduced our work, often applying DRAGIN to new datasets using the default hyperparameters without any adjustments. These include:

- Parekh et. al. "Dynamic Strategy Planning for Efficient Question Answering with Large Language Models"
- Guan et. al. "DeepRAG: Thinking to Retrieve Step by Step for Large Language Models"
- Jiang et. al. "TC–RAG: Turing–Complete RAG’s Case study on Medical LLM Systems"
- Zhao et. al. "MedRAG: Enhancing RAG with Knowledge Graph-Elicited Reasoning for Healthcare Copilot"
- ......

This work has been reproduced by many subsequent studies and has proven effective in most cases, generally outperforming state-of-the-art methods from before 2024 (our paper was published in March 2024). **However, there are specific instances where certain datasets and corpora are not well-suited for DRAGIN, which we believe is acceptable. Essentially, DRAGIN models the information needs of LLMs based on keywords, making it more effective for retrieval tasks that rely on lexical matching. In other scenarios, particularly those where dense retrieval excels, DRAGIN's performance is not as strong.**


------

**📢 January 18, 2025 Update, Important:**

We have observed significant performance differences in recently released LLMs (e.g., Qwen 2.5 series) **when using their official chat templates versus without them.** If you wish to reproduce results using the code in this repository for new LLMs, **make sure to apply the official chat template provided on the LLM’s Hugging Face page rather than the default chat template from this repository.** Otherwise, the experimental results may be inaccurate.








-----

**📢 News: this work has been accepted at the ACL 2024 main conference!**

**If you find our project interesting or helpful, we would appreciate it if you could give us a star! Your support is a tremendous encouragement to us!**



**Welcome to the official GitHub repository for our ACL 2024 Main Conference Full paper: DRAGIN** (Dynamic Retrieval Augmented Generation based on the real-time Information Needs of LLMs), a dynamic RAG framework designed to enhance the text generation capabilities of Large Language Models (LLMs) by intelligently deciding when and what to retrieve during the generation process.






## Overview

DRAGIN addresses the limitations of current dynamic RAG (Retrieval Augmented Generation) methods by introducing a novel approach for real-time decision-making on retrieval timing and content. Our framework is built upon two core components:

- **RIND (Real-time Information Needs Detection):** A mechanism that determines the optimal moment for activation of the retrieval module by assessing the LLM's uncertainty, the importance of each token, and the semantic significance within the generated content.
- **QFS (Query Formulation based on Self-attention):** An innovative method for crafting retrieval queries that leverages the LLM's self-attention mechanism, allowing for a comprehensive understanding of the context.

## Key Features

- **Dynamic Retrieval:** DRAGIN actively decides when and what to retrieve, based on the LLM's real-time information needs, significantly improving the relevance and accuracy of generated text.
- **Lightweight Integration:** Designed as a lightweight framework, DRAGIN can be seamlessly incorporated into any Transformer-based LLM without the need for additional training, fine-tuning, or prompt engineering.
- **Enhanced Text Generation:** By addressing the when and what of retrieval more effectively, DRAGIN elevates the quality of LLM-generated text, making it more informative, contextually relevant, and coherent.

## Install environment

```bash
conda create -n dragin python=3.9
conda activate dragin
pip install torch==2.1.1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run DRAGIN

### Build Wikipedia index

Download the Wikipedia dump from the [DPR repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) using the following command:

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

Use Elasticsearch to index the Wikipedia dump:

```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz 
cd elasticsearch-7.17.9
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ../..
python prep_elastic_with_tqdm.py --data_path data/dpr/psgs_w100.tsv --index_name wiki  # build index
```


-----
After building the Wikipedia Index, and **before** you start the inference step, you **must** perform the following check to ensure the index is fully and correctly built.

 #### Check the Index Status then Verify the Output

 Run the following command in your terminal:

 ```bash
curl -X GET "localhost:9200/_cat/indices?v"
 ```



A **correctly and completely built** index will look *exactly* like this. Pay close attention to the `docs.count` (\~21 million) and `store.size` (\~11.2GB):

 ```
health status index uuid                  pri rep docs.count docs.deleted store.size pri.store.size dataset.size
yellow open   wiki  w3L9eKIMT4m8XZN39JGn5w   1   1   21015324            0     11.2gb         11.2gb         11.2gb
 ```

If your `docs.count` is in the thousands (e.g., 10,000) instead of 21 million, your index is **incomplete**, and your reproduction will fail.



 ### The "Silent Failure" Warning

This is the most critical part, and the source of most indexing failures:

   * **ES Must Be Running:** Elasticsearch *must* remain running in the background at all times.
   * **Disk Space is Key:** You must have **at least 10%** of your total disk space free *after* accounting for the 11.2GB index.
   * **THE SILENT KILLER:** If your disk's free space drops too low (often below 10% or 5%), **Elasticsearch will automatically stop indexing**. It will do this **silently**, with **no errors and no logs** to indicate it has stopped.

This silent failure is why many reproductions go wrong. You may *think* the index is built, but it only processed a tiny fraction of the 21 million articles.

### Download Dataset

For 2WikiMultihopQA:

Download the [2WikiMultihop](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1) dataset from its repository <https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1>. Unzip it and move the folder to `data/2wikimultihopqa`.

For StrategyQA:

```bash
wget -O data/strategyqa_dataset.zip https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip
mkdir -p data/strategyqa
unzip data/strategyqa_dataset.zip -d data/strategyqa
rm data/strategyqa_dataset.zip 
```

For HotpotQA:

```bash
mkdir -p data/hotpotqa
wget -O data/hotpotqa/hotpotqa-dev.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

For IIRC:

```bash
wget -O data/iirc.tgz https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz
tar -xzvf data/iirc.tgz
mv iirc_train_dev/ data/iirc
rm data/iirc.tgz
```

### Run

The parameters that can be selected in the config file `config.json` are as follows:

| parameter                 | meaning                                                      | example/options                                              |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `model_name_or_path`      | Hugging Face model.                                          | `meta-llama/Llama-2-13b-chat`                             |
| `method`                  | way to generate answers             | `non-retrieval`, `single-retrieval`, `token`, `fix-sentence-retrieval`, `fix-length-retrieval`, `dragin` |
| `dataset`                 | Dataset                                                      | `2wikimultihopqa`, `hotpotqa`, `iirc`, `strategyqa`          |
| `data_path`               | the folder where the data is located. If you use the above code to download the data, the folder will be `../data/dataset`. | `../data/2wikimultihopqa`                                    |
| `fewshot`                 | Few shot.                                                    | 6                                                            |
| `sample`                  | number of questions sampled from the dataset.<br />`-1` means use the entire data set. | 1000                                                         |
| `shuffle`                 | Whether to disrupt the data set.<br />Without this parameter, the data set will not be shuffled. | `true`, `false`(without)                                     |
| `generate_max_length`     | maximum generated length of a question                       | 64                                                           |
| `query_formulation`       | way to generate retrieval question.                          | main: `direct`, `real_words`<br />another options: `current_wo_wrong`, `current`, `forward_all`, `last_n_tokens`, `last_sentence` |
| `retrieve_keep_top_k`     | number of reserved tokens when generating a search question  | 35                                                           |
| `output_dir`              | The generated results will be stored in a folder with a numeric name at the output folder you gave. If the folder you give does not exist, one will be created. | `../result/2wikimultihopqa_llama2_13b`                       |
| `retriever`               | type of retriever.                                           | `BM25`, `SGPT`                                               |
| `retrieve_topk`           | number of related documents retained.                        | 3                                                            |
| `hallucination_threshold` | threshold at which a word is judged to be incorrect.         | 1.2                                                          |
| `check_real_words`        | Whether only content words participate in threshold judgment.<br />Without this parameter, all words will be considered. | `true`, `false`(without)                                     |
| `use_counter`             | Whether to use counters to count the number of generation, retrieval, number of problems, number of tokens generated, and number of sentences generated.<br />Without this parameter, the number will not be counted. | `true`, `false`(without)                                     |

If you are using BM25 as the retriever, you should also include the following parameters

| Parameter       | Meaning                                    | example |
| --------------- | ------------------------------------------ | ------- |
| `es_index_name` | The name of the index in the Elasticsearch | `wiki`  |

If you are using SGPT as the retriever, you should also include the following parameters.

| Parameter                 | Meaning                               | example                                                   |
| ------------------------- | ------------------------------------- | --------------------------------------------------------- |
| `sgpt_model_name_or_path` | SGPT model                            | `Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit` |
| `sgpt_encode_file_path`   | Folders to save SGPT encoding results | `../sgpt/encode_result`                                   |
| `passage_file`            | Path to the Wikipedia dump            | `../data/dpr/psgs_w100.tsv`                               |

Here is the config file for using our approach to generate answers to the top 1000 questions of 2WikiMultihopQA using the model Llama-2-13b-chat.

```json
{
    "model_name_or_path": "meta-llama/Llama-2-13b-chat",
    "method": "attn_entropy",
    "dataset": "2wikimultihopqa",
    "data_path": "../data/2wikimultihopqa",
    "generate_max_length": 64,
    "query_formulation": "real_words",
    "retrieve_keep_top_k": 40,
    "output_dir": "../result/2wikimultihopqa_llama2_13b",
    "retriever": "BM25",
    "retrieve_topk": 3,
    "hallucination_threshold": 1.2,
    "fewshot": 6,
    "sample": 1000,
    "shuffle": false,
    "check_real_words": true,
    "es_index_name": "34051_wiki",
    "use_counter": true
}
```

The config files of the main experiments in the paper are all in the `config/`.

When you have prepared the configuration file, run the following command in the `src` directory:

```bash
python main.py -c path_to_config_file
```

### Evaluate

Upon completion of the program, you will find a folder named with a numerical identifier within your designated output directory. This identifier corresponds to the sequential order of runs within that folder, allowing for easy organization of multiple executions. Additionally, during the runtime, you will receive prompts indicating the specific folder where the current run's results will be saved.

Assume that the results of your run are saved in the `result/2wikimultihopqa_llama2_13b/1`，run the following command in the `src` directory to evaluate:

```bash
python evaluate.py --dir path_to_folder(result/2wikimultihopqa_llama2_13b/1)
```

After the evaluation program has finished running, the results folder will contain the following files:

```plain
result/
└── 2wikimultihopqa_llama2_13b/
    └── 1/
        ├── config.json # the configuration file you use when running
        ├── details.txt # Evaluation details
        ├── output.txt # Original output file, which will contain statistical results if use_counter is set to true
        └── result.tsv # Evaluation results
```

The elements in `output.txt` are as follows:

```json
{
    "qid": "question id", 
    "prediction": "origin outputs", 
    "retrieve_count": 0, 
    "generate_count": 1, 
    "hallucinated_count": 0, 
    "token_count": 64, 
    "sentence_count": 5
}
```

The elements in `details.txt` are as follows:

```json
{
    "qid": "question id", 
    "final_pred": "the output used for evaluation after extraction", 
    "EM": "EM result", 
    "F1": "F1 result"
}
```

# DRAGIN





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

Additionally, we selected 1000 questions from each dataset. We have received some inquiries regarding this selection process. Specifically, we chose the top 1000 questions. Taking the top 1000 is as reasonable as randomly sampling 1000, and it offers more stability by avoiding the influence of random factors.

Furthermore, our method has been independently reproduced by others. For instance, the author of https://github.com/Bocchi7/DRAGIN_simplified independently implemented our work based solely on the content provided in the paper. We have contacted them and obtained their results. While there are minor differences in the specific numbers, a t-test for statistical significance revealed that these differences are not significant.

Several other published papers have also independently reproduced our work, often applying DRAGIN to new datasets using the default hyperparameters without any adjustments. These include:

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



**📢 News:**

We would like to thank @Bocchi7 for reproducing our work with a simplified version of the code. You can check it out here:  
[https://github.com/Bocchi7/DRAGIN_simplified](https://github.com/Bocchi7/DRAGIN_simplified)



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
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki  # build index
```

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

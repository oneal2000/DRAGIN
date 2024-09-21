# DRAGIN

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
| `method`                  | way to generate answers             | `non-retrieval`, `single-retrieval`, `token`, `fix-sentence-retrieval`, `fix-length-retrieval`, `attn_entropy` |
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

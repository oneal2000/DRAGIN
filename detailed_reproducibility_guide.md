## DRAGIN



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
wget -O elasticsearch-7.9.1.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-7.9.1.tar.gz
rm elasticsearch-7.9.1.tar.gz 
cd elasticsearch-7.9.1
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ../..
python prep_elastic_with_tqdm.py --data_path data/dpr/psgs_w100.tsv --index_name wiki  # build index
```

When you run Elasticsearch successfully, you should see log messages like:

```
#files 1
create index wiki
124001docs [00:26, 5261.58docs/s]
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
All the configuration files of our experiments are in the `config` folder.  
You can run DRAGIN with them directly. For example, if you want to run DRAGIN on 2WikiMultihopQA with LLaMA2-7B, fist, you need to get into `src` folder, then run:

```bash
python main.py -c ../config/Llama2-7b-chat/HotpotQA/DRAGIN.json
```

If you don't have llama2-7b-chat model locally, the above command will download it from Hugging Face Hub first.  

```bash
INFO:faiss.loader:Loading faiss.
INFO:faiss.loader:Successfully loaded faiss.
INFO:__main__:Namespace(model_name_or_path='meta-llama/Llama-2-7b-chat-hf', method='dragin', dataset='hotpotqa', data_path='../data/hotpotqa', fewshot=8, sample=1000, shuffle=False, generate_max_length=100, query_formulation='real_words', retrieve_keep_top_k=35, output_dir='../result/llama2_7b_chat_hotpotqa', retriever='BM25', es_index_name='wiki', retrieve_topk=3, hallucination_threshold=0.8, check_real_words=True, use_counter=True, config_path='../config/Llama2-7b-chat/HotpotQA/DRAGIN.json')
INFO:__main__:output dir: ../result/llama2_7b_chat_hotpotqa/9
INFO:data:Loading HotpotQA from ../data/hotpotqa
...
INFO:generate:Loading model from meta-llama/Llama-2-7b-chat-hf
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
...
INFO:root:Activating Elasticsearch....
INFO:root:Elastic Search Credentials: {'hostname': 'localhost', 'index_name': 'wiki', 'keys': {'title': 'title', 'body': 'txt'}, 'timeout': 100, 'retry_on_timeout': True, 'maxsize': 24, 'number_of_shards': 1, 'language': 'english'}
INFO:__main__:start inference

  0%|          | 0/1000 [00:00<?, ?it/s]
  0%|          | 1/1000 [00:07<2:08:37,  7.73s/it]
...
```

If you see log messages like above, it means DRAGIN is running successfully. The output will be saved in the folder `result/llama2_7b_chat_hotpotqa`. You can change the output folder by modifying the `output_dir` parameter in the configuration file.  


### Evaluate
Upon completion of the program, you will find a folder named with a numerical identifier within your designated output directory. This identifier corresponds to the sequential order of runs within that folder, allowing for easy organization of multiple executions. Additionally, during the runtime, you will receive prompts indicating the specific folder where the current run's results will be saved.  

To evaluate the results, you can use the `evaluate.py` script in the `src` folder. Assume the output folder is `result/llama2_7b_chat_hotpotqa/1`, you can run:

```bash
python eval.py --dir ../result/llama2_7b_chat_hotpotqa/1
```

If you see log messages like:
```bash
INFO:__main__:Namespace(model_name_or_path='/home/wangchangyue/LLM/Llama-2-7b-chat-hf', method='dragin', dataset='hotpotqa', data_path='../data/hotpotqa', fewshot=8, sample=1000, shuffle=False, generate_max_length=100, query_formulation='real_words', retrieve_keep_top_k=35, output_dir='../result/llama2_7b_chat_hotpotqa/8', retriever='BM25', es_index_name='wiki', retrieve_topk=3, hallucination_threshold=0.8, check_real_words=True, use_counter=True, config_path='../config/Llama2-7b-chat/HotpotQA/DRAGIN.json')
INFO:data:Loading HotpotQA from ../data/hotpotqa
```
It means the evaluation is running successfully. After the evaluation, you will see the evaluation results in: 
```bash
result/
└── llama2_7b_chat_hotpotqa/
    └── 1/
        ├── config.json # the configuration file you use when running
        ├── details.txt # Evaluation details
        ├── output.txt # Original output file, which will contain statistical results if use_counter is set to true
        └── result.tsv # Evaluation results
```

------
### Very Important:

**All experiments reported in our paper were conducted on multiple GPUs. Therefore, to reproduce the reported results, the program should be run with exactly two GPUs.**

In any case, under any circumstances, **do not make any changes to the code!!!** For example, if you think you need to modify the code to run it on a Single GPU, absolutely do not change it. If you change it, the results will not be reproducible. You must use two GPUs and run the code *without* making any modifications.

------



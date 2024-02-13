import os
import json
import argparse
from tqdm import tqdm
from copy import copy
import logging 
from data import StrategyQA, WikiMultiHopQA, NaturalQuestion, HotpotQA, TriviaQA, IIRC
from generate import *

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)


def get_args():
    # parser = ArgumentParser()
    # parser.add_argument("--model_name_or_path", type=str, required=True)
    # parser.add_argument("--method", type=str, required=True, choices=["non-retrieval", "single-retrieval", "fix-length-retrieval", "fix-sentence-retrieval", "token", "entity"])
    # parser.add_argument("--hallucination_threshold", type=float)
    # parser.add_argument("--entity_solver", type=str, choices=["avg", "max", "min", "first"], help="when using entity, how to calculate entity score")
    # parser.add_argument("--sentence_solver", type=str, choices=["avg", "max", "min"], help="when using entity, how to calculate sentence score")
    # parser.add_argument("--dataset", type=str, required=True, choices=["nq", "2wikimultihopqa", "strategyqa", "truthfulqa", "hotpotqa"])
    # parser.add_argument("--data_path", type=str, required=True)
    # parser.add_argument("--generate_max_length", type=int, required=True, help="max length for language model to generate")
    # parser.add_argument("--output_dir", type=str, required=True)
    # parser.add_argument("--retriever", type=str, choices=["BM25", "sgpt"])
    # parser.add_argument("--retrieve_topk", type=int, default=1, help="topk for retrieval")
    # parser.add_argument("--fix_length", type=int)
    # parser.add_argument("--query_formulation", type=str, choices=["direct", "attention"])
    # parser.add_argument("--fewshot", type=int)
    # parser.add_argument("--sample", type=int, default=-1, help="if none, use all dataset")
    # args = parser.parse_args()
    # return args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path, "r") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.config_path = config_path
    if "shuffle" not in args:
        args.shuffle = False 
    if "use_counter" not in args:
        args.use_counter = True
    if args.method.startswith("attn") and "try_use_max_weight" not in args:
        args.try_use_max_weight = False
    return args


def main():
    args = get_args()
    logger.info(f"{args}")

    # output dir
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    dir_name = os.listdir(args.output_dir)
    for i in range(10000):
        if str(i) not in dir_name:
            args.output_dir = os.path.join(args.output_dir, str(i))
            os.makedirs(args.output_dir)
            break
    logger.info(f"output dir: {args.output_dir}")
    # save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    # create output file
    output_file = open(os.path.join(args.output_dir, "output.txt"), "w")

    # load data
    if args.dataset == "strategyqa":
        data = StrategyQA(args.data_path)
    elif args.dataset == "2wikimultihopqa":
        data = WikiMultiHopQA(args.data_path)
    elif args.dataset == "nq":
        data = NaturalQuestion(args.data_path)
    elif args.dataset == "hotpotqa":
        data = HotpotQA(args.data_path)
    elif args.dataset == "triviaqa": 
        data = TriviaQA(args.data_path)
    elif args.dataset == "iirc":
        data = IIRC(args.data_path)
    else:
        raise NotImplementedError
    data.format(fewshot=args.fewshot)
    data = data.dataset
    if args.shuffle:
        data = data.shuffle()
    if args.sample != -1:
        samples = min(len(data), args.sample)
        data = data.select(range(samples))
   
    # 根据 method 选择不同的生成策略
    if args.method == "non-retrieval":
        model = BasicRAG(args)
    elif args.method == "single-retrieval":
        model = SingleRAG(args)
    elif args.method == "fix-length-retrieval" or args.method == "fix-sentence-retrieval":
        model = FixLengthRAG(args)
    elif args.method == "token":
        model = TokenRAG(args)
    elif args.method == "entity":
        model = EntityRAG(args)
    elif args.method == "attn_prob" or args.method == "attn_entropy":
        model = AttnWeightRAG(args)
    else:
        raise NotImplementedError

    logger.info("start inference")
    for i in tqdm(range(len(data))):
        last_counter = copy(model.counter)
        batch = data[i]
        pred = model.inference(batch["question"], batch["demo"], batch["case"])
        pred = pred.strip()
        ret = {
            "qid": batch["qid"], 
            "prediction": pred,
        }
        if args.use_counter:
            ret.update(model.counter.calc(last_counter))
        output_file.write(json.dumps(ret)+"\n")
    

if __name__ == "__main__":
    main()
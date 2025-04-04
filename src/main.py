import os
import json
import argparse
from tqdm import tqdm
from copy import copy
import logging
from data import *
from algorithm import *
from utils import *

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)
CONFIG_PATH = "../config/config.json"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_logging", action='store_true', help="Disable logging mode")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--note", type=str)
    args = parser.parse_args()
    with open(CONFIG_PATH, "r") as f:
        for k, v in json.load(f).items():
            setattr(args, k, v)
    if "shuffle" not in args:
        args.shuffle = False 
    if "use_counter" not in args:
        args.use_counter = True
    model_name = args.model[args.model.rfind("/") + 1:]
    args.output_dir = f"../results/{model_name}/{args.dataset}/{args.method}"
    return args

def main():
    args = get_args()
    logger.info(f"{args}")

    if not args.disable_logging:
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
        output_file_name = os.path.join(args.output_dir, "output.jsonl")
        output_file = open(output_file_name, "w")

    # load data
    if args.dataset == "strategyqa":
        data = StrategyQA(args.data_path)
    elif args.dataset == "2wikimultihopqa":
        data = WikiMultiHopQA(args.data_path)
    elif args.dataset == "hotpotqa":
        data = HotpotQA(args.data_path)
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
   
    if args.method == "non-retrieval":
        model = BasicRAG(args)
    elif args.method == "single-retrieval":
        model = SingleRAG(args)
    elif args.method == "fix-length-retrieval" or args.method == "fix-sentence-retrieval":
        model = FixLengthRAG(args)
    elif args.method == "flare":
        model = FlareRAG(args)
    elif args.method == "attn_prob" or args.method == "dragin":
        model = AttnWeightRAG(args)
    elif args.method == "flare":
        model = FlareRAG(args)
    elif args.method == "entity-flare":
        model = EntityFlareRAG(args)
    elif args.method == "semantic_entropy":
        model = SemanticEntropyRAG(args)
    elif args.method == "dynamic_thresholding":
        model = DynamicThresholdingRAG(args)

    logger.info("start inference")
    for batch in tqdm(batchify(data, args.batch_size)):
        # last_counter = copy(model.generator.counter)
        inference_results = model.inference(batch["question"], batch["demo"], batch["case"])
        preds = inference_results['text']
        preds = [pred.strip() for pred in preds]
        qids = batch['qid']
        token_uncertainty = inference_results.get('token_uncertainty', [None] * len(preds))

        rets = [
            {
                "qid": qid,
                "prediction": pred,
                **({"token_uncertainty": uncertainty} if uncertainty is not None else {})
            }
            for qid, pred, uncertainty in zip(qids, preds, token_uncertainty)
        ]
        # if args.use_counter:
        #     rets.update(model.generator.counter.calc(last_counter))
        if not args.disable_logging:
            ret_lines = '\n'.join(json.dumps(ret) for ret in rets)
            output_file.write(ret_lines+"\n")
            
    if not args.disable_logging:
        output_file.close()
        # Reopen the jsonl file, read and convert to a list of dicts
        json_path = output_file_name.rsplit(".", 1)[0] + ".json"

        with open(output_file_name, "r", encoding="utf-8") as infile:
            json_list = [json.loads(line) for line in infile]

        # Write the list of dicts into a new .json file with indentation
        with open(json_path, "w", encoding="utf-8") as outfile:
            json.dump(json_list, outfile, ensure_ascii=False, indent=2)
    

if __name__ == "__main__":
    main()

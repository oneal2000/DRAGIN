import os
import json
import argparse
from tqdm import tqdm
from copy import copy
import logging
from data import *
from algorithm import *

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    parser.add_argument("--no_log", action='store_true', help="Disable logging mode")
    args = parser.parse_args()
    if args.debug:
        import debugpy
        debugpy.listen(5679)
        print("wait for debugger")
        debugpy.wait_for_client()
        print("attached")
    enable_logging = not args.no_log
    config_path = args.config_path
    with open(config_path, "r") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.config_path = config_path
    if "shuffle" not in args:
        args.shuffle = False 
    if "use_counter" not in args:
        args.use_counter = True
    args.enable_logging = enable_logging
    return args

def main():
    args = get_args()
    logger.info(f"{args}")

    if args.enable_logging:
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
    elif args.method == "attn_prob" or args.method == "dragin":
        model = AttnWeightRAG(args)
    elif args.method == "flare":
        model = FlareRAG(args)
    elif args.method == "entity-flare":
        model = EntityFlareRAG(args)
    else:
        raise NotImplementedError

    logger.info("start inference")
    # DEBUG: for analysis token's confidence
    token_prob_file = open(os.path.join(args.output_dir, "token_scores.txt"), "w")
    # End of DEBUGGGG
    for i in tqdm(range(len(data))):
        last_counter = copy(model.counter)
        batch = data[i]
        pred = model.inference(batch["question"], batch["demo"], batch["case"])
        # DEBUG: for analysis token's confidence
        # text, confidence_list = pred
        # ret = {
        #     "qid": batch["qid"], 
        #     "question": batch["question"],
        #     "ground_truth": batch["answer"],
        #     "prediction": text,
        #     "token_scores": confidence_list
        # }
        # if args.enable_logging:
        #     token_prob_file.write(json.dumps(ret)+"\n")
        # continue
        # End of DEBUGGGG
        pred = pred.strip()
        ret = {
            "qid": batch["qid"], 
            "prediction": pred,
        }
        if args.use_counter:
            ret.update(model.counter.calc(last_counter))
        if args.enable_logging:
            output_file.write(json.dumps(ret)+"\n")
    

if __name__ == "__main__":
    main()

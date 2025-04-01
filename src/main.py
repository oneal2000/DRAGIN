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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    parser.add_argument("--disable_logging", action='store_true', help="Disable logging mode")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    if args.debug:
        import debugpy
        debugpy.listen(5679)
        print("wait for debugger")
        debugpy.wait_for_client()
        print("attached")
    with open(args.config_path, "r") as f:
        for k, v in json.load(f).items():
            setattr(args, k, v)
    if "shuffle" not in args:
        args.shuffle = False 
    if "use_counter" not in args:
        args.use_counter = True
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
        preds = model.inference(batch["question"], batch["demo"], batch["case"])
        preds = [pred.strip() for pred in preds]
        qids = batch['qid']
        rets = [{
            "qid": qid, 
            "prediction": pred,
        } for qid, pred in zip(qids, preds)]
        # if args.use_counter:
        #     rets.update(model.generator.counter.calc(last_counter))
        if not args.disable_logging:
            ret_lines = '\n'.join(json.dumps(ret) for ret in rets)
            output_file.write(ret_lines+"\n")
    

if __name__ == "__main__":
    main()

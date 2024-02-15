import os
import json
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import StrategyQA, WikiMultiHopQA, HotpotQA, IIRC
from transformers import AutoTokenizer, AutoModelForCausalLM 

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    tmp = parser.parse_args()
    with open(os.path.join(tmp.dir, "config.json"), "r") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.output_dir = tmp.dir
    return args


def regenerate_answer(cot, tokenizer, model, case, demo):
    # print("##### origin #####")
    # print(cot)
    split_words = ["Question:", "#10000000", "Note:"]
    # split_words = ["Question:", "#10000000", "\n"]
    for word in split_words:
        pos = cot.find(word)
        if pos != -1 and pos > 0:
            cot = cot[:pos]
    if "the answer is" in cot:
        return cot 

    cot += " So the answer is "
    prompt = "".join([d["case"]+"\n" for d in demo])
    prompt += case + " " + cot
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    input_length = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    outputs = model.generate(
        input_ids = input_ids, 
        attention_mask = attention_mask, 
        max_new_tokens = 20)
    generated_tokens = outputs[:, input_length:]
    text = tokenizer.decode(generated_tokens[0])
    text = cot + text.strip()
    for word in split_words:
        pos = text.find(word)
        if pos != -1:
            text = text[:pos] 
    # print("##### prompt #####")
    # print(prompt)
    # print("##### output #####")
    # print(text)
    # print("##### pred #####")
    return text


def main():
    args = get_args()
    logger.info(f"{args}")
    
    if args.dataset == 'strategyqa':
        data = StrategyQA(args.data_path)
    elif args.dataset == '2wikimultihopqa':
        data = WikiMultiHopQA(args.data_path)
    elif args.dataset == 'hotpotqa':
        data = HotpotQA(args.data_path)
    elif args.dataset == 'iirc':
        data = IIRC(args.data_path)
    else:
        raise NotImplementedError
    data.format(fewshot=args.fewshot)

    dataset = {}
    for i in range(len(data.dataset)):
        t = data.dataset[i]
        dataset[t["qid"]] = [
            t["answer"], 
            t["answer_id"] if "answer_id" in t else None,
            t["case"] if "case" in t else None
        ]

    metrics = ["EM", "F1", "Precision", "Recall"]
    if "use_counter" not in args or args.use_counter:
        count_list = ["retrieve_count", "generate_count", "hallucinated_count", "token_count", "sentence_count"]
        metrics += count_list
    value = [[] for _ in range(len(metrics))]
    with open(os.path.join(args.output_dir, "output.txt"), "r") as fin:
        lines = fin.readlines()
    
    need_generate = args.dataset in ['2wikimultihopqa', "hotpotqa", "iirc", "strategyqa"] 
    if need_generate:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto",
                                                     trust_remote_code = "falcon" in args.model_name_or_path)
        demo = data.dataset[0]["demo"]

    pred_out = open(f"{args.output_dir}/details.txt", "w")
    
    for line in tqdm(lines):
        rd = json.loads(line)
        qid = rd["qid"]
        pred = rd["prediction"]
        ground_truth, ground_truth_id, case = dataset[qid]
        if need_generate:
            pred = regenerate_answer(pred, tokenizer, model, case, demo) 
        pred = data.get_real_prediction(pred)
        # print("*****", pred)

        em_ret = data.exact_match_score(
            pred, 
            ground_truth, 
            ground_truth_id
        )
        f1_ret = data.f1_score(
            pred, 
            ground_truth, 
            ground_truth_id
        )
        value[0].append(em_ret["correct"])
        for i, k in enumerate(f1_ret.keys()):
            value[i+1].append(f1_ret[k])
        if "use_counter" not in args or args.use_counter:
            for i, k in enumerate(count_list):
                value[i+4].append(rd[k])
        detail = {
            "qid": qid, 
            "final_pred": pred,
            "EM": str(em_ret["correct"]), 
            "F1": str(f1_ret["f1"]) 
        }
        pred_out.write(json.dumps(detail)+"\n")

    ret = []
    for i, metric in enumerate(metrics):
        val = np.array(value[i])
        ret.append([metric, val.mean()])
    df = pd.DataFrame(ret)
    df.to_csv(f"{args.output_dir}/result.tsv", index=False, header=False)


if __name__ == "__main__":
    main()
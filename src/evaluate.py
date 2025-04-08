import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import IIRC, HotpotQA, StrategyQA, WikiMultiHopQA
from utils import batchify

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


def clean_generated_text(text, split_words):
    """
    Cleans and truncates the generated text based on split words.
    """
    for word in split_words:
        pos = text.find(word)
        if pos != -1:
            text = text[:pos]
    return text


def regenerate_answer(cots, tokenizer, model, cases, demos):
    split_words = ["\nQuestion:", "#10000000", "Note:", ". Question:"] + [tokenizer.eos_token]
    if tokenizer.pad_token:
        split_words.append(tokenizer.pad_token)
    
    results = [] 
    processed_cots = []
    processed_indices = []  # Track indices of processed CoTs

    for idx, cot in enumerate(cots):
        if "the answer is" in cot:
            results.append(cot)  # Directly append to results
        else:
            clean_generated_text(cot, split_words)
            processed_cots.append(cot + " So the answer is ")
            processed_indices.append(idx)  # Track the index of this CoT

    # Generate predictions only for processed CoTs
    if processed_cots:
        prompts = []
        for case, cot, demo in zip(cases, processed_cots, demos):
            prompt = "".join([d["case"] + "\n" for d in demo])
            prompt += case + " " + cot
            prompts.append(prompt)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(model.device)
        input_lengths = inputs["input_ids"].shape[1]
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=20
        )
        generated_tokens = outputs[:, input_lengths:]
        texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for cot, text in zip(processed_cots, texts):
            text = cot + text.strip()
            results.append(clean_generated_text(text, split_words))

    # Reorder results to match the original order of `cots`
    final_results = [None] * len(cots)
    result_idx = 0
    for idx in range(len(cots)):
        if idx in processed_indices:
            final_results[idx] = results[result_idx]
            result_idx += 1
        else:
            final_results[idx] = results.pop(0)

    return final_results


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
            t['question'],
            t["answer"], 
            t["answer_id"] if "answer_id" in t else None,
            t["case"] if "case" in t else None
        ]

    metrics = ["EM", "F1", "Precision", "Recall"]
    # if "use_counter" not in args or args.use_counter:
    #     count_list = ["retrieve_count", "generate_count", "hallucinated_count", "token_count", "sentence_count"]
    #     metrics += count_list
    value = [[] for _ in range(len(metrics))]
    with open(os.path.join(args.output_dir, "output.jsonl"), "r") as fin:
        lines = fin.readlines()
    
    need_generate = args.dataset in ['2wikimultihopqa', "hotpotqa", "iirc", "strategyqa"] 
    if need_generate:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto",
                                                     trust_remote_code = "falcon" in args.model)
        demo = data.dataset[0]["demo"]

    pred_out = open(f"{args.output_dir}/details.jsonl", "w")

    for batch in tqdm(batchify(lines, args.batch_size)):
        batch_data = [json.loads(line) for line in batch]
        qids = [rd["qid"] for rd in batch_data]
        preds = [rd["prediction"] for rd in batch_data]
        questions, ground_truths, ground_truth_ids, cases = zip(
            *[dataset[qid] for qid in qids]
        )

        if need_generate:
            preds = regenerate_answer(preds, tokenizer, model, cases, [demo] * len(batch_data))

        preds = [data.get_real_prediction(pred) for pred in preds]

        for qid, pred, question, ground_truth, ground_truth_id in zip(
            qids, preds, questions, ground_truths, ground_truth_ids
        ):
            em_ret = data.exact_match_score(pred, ground_truth, ground_truth_id)
            f1_ret = data.f1_score(pred, ground_truth, ground_truth_id)
            value[0].append(em_ret["correct"])
            for i, k in enumerate(f1_ret.keys()):
                value[i+1].append(f1_ret[k])
            # if "use_counter" not in args or args.use_counter:
            #     for i, k in enumerate(count_list):
            #         value[i+4].append(rd[k])
            detail = {
                "ground_truth": ground_truth,
                "final_pred": pred,
                "EM": str(em_ret["correct"]),
                "F1": str(f1_ret["f1"]),
                "question": question,
                "qid": qid
            }
            pred_out.write(json.dumps(detail, ensure_ascii=False)+"\n")

    ret = []
    for i, metric in enumerate(metrics):
        val = np.array(value[i])
        ret.append([metric, val.mean()])
    df = pd.DataFrame(ret)
    df.to_csv(f"{args.output_dir}/result.tsv", index=False, header=False)


if __name__ == "__main__":
    main()

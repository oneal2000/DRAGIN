from typing import Dict, List
import logging
import os
import json
from tqdm import tqdm
from datasets import Dataset
from .base_dataset import BaseDataset

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

class StrategyQA(BaseDataset):
    examplars: List[Dict] = [
        {
            'question': 'Do hamsters provide food for any animals?',
            'ctxs': [(None, "Hamsters are prey animals."),
                (None, "Prey animals provide food for predators.")],
            'cot': ('Hamsters are prey animals. ',
                'Prey are food for predators. ',
                'Thus, hamsters provide food for some animals.'),
            'answer': 'yes',
        },
        {
            'question': 'Could Brooke Shields succeed at University of Pennsylvania?',
            'ctxs': [(None, "Brooke Shields graduated from Princeton University."),
                (None, "Princeton is ranked as the number 1 national college by US news."),
                (None, "University of Pennsylvania is ranked as number 6 national college by US news."),
                (None, "Princeton only admits around 6 percent of applicants as of 2018."),
                (None, "University of Pennsylvania accepts around 9% of applicants as of 2018.")],
            'cot': ('Brooke Shields went to Princeton University. ',
                'Princeton University is about as academically rigorous as the University of Pennsylvania. ',
                'Thus, Brooke Shields could also succeed at the University of Pennsylvania.'),
            'answer': 'yes',
        },
        {
            'question': "Hydrogen's atomic number squared exceeds number of Spice Girls?",
            'ctxs': [(None, "Hydrogen is the first element and has an atomic number of one."),
                (None, "The Spice Girls has five members."),
                (None, "To square a number, you multiply it by itself.")],
            'cot': ("Hydrogen has an atomic number of 1. ",
                "1 squared is 1. ",
                "There are 5 Spice Girls. ",
                "Thus, Hydrogen's atomic number squared is less than 5."),
            'answer': 'no',
        },
        {
            'question': "Is it common to see frost during some college commencements?",
            'ctxs': [(None, "Frost isn't uncommon to see during the month of December, as it is the winter."),
                (None, "College commencement ceremonies often happen during the months of December, May, and sometimes June.")],
            'cot': ("College commencement ceremonies can happen in December, May, and June. ",
                "December is in the winter, so there can be frost. ",
                "Thus, there could be frost at some commencements."),
            'answer': 'yes',
        },
        {
            'question': "Could a llama birth twice during War in Vietnam (1945-46)?",
            'ctxs': [(None, "The War in Vietnam (1945-46) lasted around 6 months."),
                (None, "The gestation period for a llama is 11 months.")],
            'cot': ("The War in Vietnam was 6 months. ",
                "The gestation period for a llama is 11 months, which is more than 6 months. ",
                "Thus, a llama could not give birth twice during the War in Vietnam."),
            'answer': 'no',
        },
        {
            'question': "Would a pear sink in water?",
            'ctxs': [(None, "The density of a raw pear is about 0.59 g/cm^3."),
                (None, "The density of water is about 1 g/cm^3."),
                (None, "Objects only sink if they are denser than the surrounding fluid.")],
            'cot': ("The density of a pear is about 0.6g/cm^3, which is less than water. ",
                "Objects less dense than water float. ",
                "Thus, a pear would float."),
            'answer': 'no',
        }
    ]

    demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    test_input_template = lambda self, ques: f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {ques}\nAnswer:'
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, data_path: str):
        logger.info(f"Loading StrategyQA from {data_path}")
        dataset = []
        with open(os.path.join(data_path, "strategyqa_train.json"), "r") as fin:
            dataset_1 = json.load(fin)
        with open(os.path.join(data_path, "strategyqa_train_paragraphs.json"), "r") as fin:
            dataset_2 = json.load(fin)
        for data in tqdm(dataset_1):
            example = {
                "qid": data["qid"], 
                "question": data["question"], 
                "cot": " ".join(data["facts"]), 
                "answer": "yes" if data["answer"] == True else "no", 
            }
            title = []
            ctxs = []
            for evi in data["evidence"][0]:
                if type(evi) == list:
                    for t in evi:
                        if type(t) == list:
                            title.extend(t)
                        else:
                            title.append(t)
                else:
                    title.append(evi)
            for tl in title:
                if tl == "operation" or tl == "no_evidence":
                    continue
                if tl in dataset_2:
                    ctxs.append(dataset_2[tl]["content"])
            example["ctxs"] = " ".join(ctxs)
            dataset.append(example)
        self.dataset = Dataset.from_list(dataset)
    
    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:]
                if pred[0:3].lower() == 'yes':
                    return "yes"
                else:
                    return "no"
        else:
            return ""


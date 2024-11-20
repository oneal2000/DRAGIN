from typing import Dict, List
import logging
import os
import json
from tqdm import tqdm
from datasets import Dataset
from .base_dataset import BaseDataset

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

class IIRC(BaseDataset):
    examplars: List[Dict] = [
        {
            "question": "What is the age difference between the kicker and the quarterback for the Chargers?",
            "cot": "The kicker for the Chargers is Nate Kaeding. The quarterback (QB) for the Chargers is Philip Rivers. Nate Kaeding was born in the year 1982. Philip Rivers was born in the year 1981. Thus, the age difference between them is of 1 year.",
            "answer": "1"
        },
        {
            "question": "How many years was the ship that took the battalion from New South Wales to Ceylon in service?",
            "cot": "The ship that took the battalion from New South Wales to Ceylon is General Hewitt. General Hewitt was launched in Calcutta in 1811. General Hewitt was sold for a hulk or to be broken up in 1864. So she served for a total of 1864 - 1811 = 53 years.",
            "answer": "53"
        },
        {
            "question": "What year was the theatre that held the 2016 NFL Draft built?",
            "cot": "The theatre that held the 2016 NFL Draft is Auditorium Theatre. The Auditorium Theatre was built in 1889.",
            "answer": "1889"
        },
        {
            "question": "How long had Milan been established by the year that Nava returned there as a reserve in the first team's defense?",
            "cot": "Nava returned to Milan as a reserve in the first team's defense in the year 1990. Milan had been established in the year 1899. Thus, Milan had been established for 1990 - 1899 = 91 years when Milan returned to Milan as a reserve in the first team's defense.",
            "answer": "91"
        },
        {
            "question": "When was the town Scott was born in founded?",
            "cot": "Scott was born in the town of Cooksville, Illinois. Cooksville was founded in the year 1882.",
            "answer": "1882"
        },
        {
            "question": "In what country did Wright leave the French privateers?",
            "cot": "Wright left the French privateers in Bluefield's river. Bluefields is the capital of the South Caribbean Autonomous Region (RAAS) in the country of Nicaragua.",
            "answer": "Nicaragua"
        },
        {
            "question": "Who plays the A-Team character that Dr. Hibbert fashioned his hair after?",
            "cot": "Dr. Hibbert fashioned his hair after Mr. T from The A-Team. Mr T.'s birthname is Lawrence Tureaud.",
            "answer": "Lawrence Tureaud"
        },
        {
            "question": "How many people attended the conference held near Berlin in January 1942?",
            "cot": "The conference held near Berlin in January 1942 is Wannsee Conference. Wannsee Conference was attended by 15 people.",
            "answer": "15"
        },
        {
            "question": "When did the country Ottwalt went into exile in founded?",
            "cot": "Ottwalt went into exile in the country of Denmark. Denmark has been inhabited since around 12,500 BC.",
            "answer": "12,500 BC"
        },
        {
            "question": "When was the J2 club Uki played for in 2001 founded?",
            "cot": "The J2 club that Uki played for is Montedio Yamagata. Montedio Yamagata was founded in 1984.",
            "answer": "1984"
        },
        {
            "question": "When was the person who produced A Little Ain't Enough born?",
            "cot": "A Little Ain't Enough was produced by Bob Rock. Bob Rock was born on April 19, 1954.",
            "answer": "April 19, 1954"
        },
        {
            "question": "Which of the schools Fiser is affiliated with was founded first?",
            "cot": "The schools that Fiser is affiliated with (1) Academy of Music, University of Zagreb (2) Mozarteum University of Salzburg (3) Croatian Music Institute orchestra. Academy of Music, University of Zagreb was founded in the year 1829. Mozarteum University of Salzburg was founded in the year 1841. Croatian Music Institute was founded in the year 1827. Thus, the school founded earliest of these is Croatian Music Institute.",
            "answer": "Croatian Music Institute"
        },
        {
            "question": "How many casualties were there at the battle that Dearing fought at under Jubal Early?",
            "cot": "Under Jubal Early, Dearing fought the First Battle of Bull Run. First Battle of Bull Run has 460 union casualties and 387 confederate casualties. Thus, in total the First Battle of Bull Run had 460 + 387 = 847 casualties.",
            "answer": "847"
        },
        {
            "question": "Which of the two congregations which provided leadership to the Pilgrims was founded first?",
            "cot": "The congregations which provided leadership to the Pilgrims are Brownists and Separatist Puritans. Brownist was founded in 1581. The Separatist Puritans was founded in 1640. Thus, Brownist was founded first.",
            "answer": "Brownist"
        },
        {
            "question": "How long had the Rock and Roll Hall of Fame been open when the band was inducted into it?",
            "cot": "The band was inducted into Rock and Roll Hall of Fame in the year 2017. Rock and Roll Hall of Fame was established in the year of 1983. Thus, Rock and Roll Hall of Fame been open for 2018 - 1983 = 34 years when the band was inducted into it.",
            "answer": "34"
        },
        {
            "question": "Did the Lord Sewer who was appointed at the 1509 coronation live longer than his king?",
            "cot": "Lord Sewer who was appointed at the 1509 coronation was Robert Radcliffe, 1st Earl of Sussex. Lord Sever's king in 1509 was Henry VIII of England. Robert Radcliffe, 1st Earl of Sussex was born in the year 1483, and died in the year 1542. So Robert lived for 1542 - 1483 = 59 years. Henry VIII of England was born in the year 1491 and died in the year 1547. So Henry VIII lived for 1547 - 1491 = 56 years. Thus, Robert Radcliffe lived longer than Henry VIII.",
            "answer": "yes"
        },
        {
            "question": "When was the place near where Manuchar was defeated by Qvarqvare established?",
            "cot": "Manuchar was defeated by Qvarqvare near Erzurum. Erzurum was founded during the Urartian period.",
            "answer": "Urartian period"
        },
        {
            "question": "What year was the man who implemented the 46 calendar reform born?",
            "cot": "The man who implemented the 46 calendar reform is Julius Caesar. Julius Caesar was born in the year 100 BC.",
            "answer": "100 BC"
        },
        {
            "question": "How many years after the first recorded Tommy John surgery did Scott Baker undergo his?",
            "cot": "The first recorded Tommy John surgery happened when it was invented in the year 1974. Scott Baker underwent Tommy John surgery in the year 2012. Thus, Scott Baker underwent Tommy John surgery 2012 - 1974 = 38 years after it was first recorded.",
            "answer": "38"
        },
        {
            "question": "Which was the older of the two players who found the net in the Double-Headed Eagle of the North in the sixth final for PAOK?",
            "cot": "The two players who found the net in the Double-Headed Eagle of the North in the sixth final for PAOK are Koudas and Matzourakis. Koudas was born on 23 November 1946. Matzourakis was born on 6 June 1949. Thus, the older person among the two is Koudas.",
            "answer": "Koudas"
        }
    ]

    demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    test_input_template = lambda self, ques: f'Question: {ques}\nAnswer:' 
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, data_path: str):
        logger.info(f"Loading IIRC dev from {data_path}")
        dataset = []
        with open(os.path.join(data_path, 'dev.json'), "r") as fin:
            js = json.load(fin)
            for tmp in tqdm(js):
                for example in tmp['questions']:
                    qid = example["qid"]
                    question = example['question']

                    ans = example['answer']

                    if ans['type'] == 'none':
                        continue
                    elif ans['type'] == 'value' or ans['type'] == 'binary':
                        answer = [ans['answer_value']]
                    elif ans['type'] == 'span':
                        answer = [v['text'].strip() for v in ans['answer_spans']]
                    
                    # context = example['context']
                    dataset.append({
                        'qid': qid,
                        'question': question,
                        'answer': answer,
                        # 'ctxs': context,
                    })
        self.dataset = Dataset.from_list(dataset)

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:] # delete final "."
                for stop_word in ["</s>", "<|endoftext|>", "\n", "."]:
                    if pred.endswith(stop_word):
                        pred = pred[:len(pred) - len(stop_word)]
                return pred
        else:
            return ""

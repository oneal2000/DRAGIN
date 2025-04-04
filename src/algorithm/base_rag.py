from retriever import BM25, SGPT
from .generator import BasicGenerator
from .utils import nlp, Counter

class BasicRAG:
    def __init__(self, args):
        args = args.__dict__ 
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                # gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer, 
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name, 
                    engine = "elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model = self.sgpt_model, 
                    sgpt_encode_file_path = self.sgpt_encode_file_path,
                    passage_file = self.passage_file
                )
            else:
                raise NotImplementedError
        
        self.counter = Counter()

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk, 
                max_query_length = max_query_length,
            )
            return docs[0]
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk,
            )
            return docs[0] 
        else:
            raise NotImplementedError
    
    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" 
    
    def inference(self, questions, demos, cases, view_uncertainty=True):
        # non-retrieval
        assert self.query_formulation == "direct"
        prompts = ["".join([d["case"]+"\n" for d in demo]) + case for demo, case in zip(demos, cases)]
        return_dict = self.generator.generate(prompts, self.generate_max_length)
        text = return_dict['text']
        # token_uncertainty = list(zip(return_dict['tokens'], return_dict['entropies'], return_dict['logprobs']))
        inference_results = dict(text=text)
        if view_uncertainty:
            token_uncertainty = list(zip(return_dict['tokens'], return_dict['entropies'], return_dict['logprobs']))
            inference_results['token_uncertainty'] = token_uncertainty
        return inference_results

    def extract_sentence_token_positions(self, text, tokens):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        
        position_list = []
        tid = 0
        
        for sent in sentences:
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr].strip())
                if apr == -1:
                    break
                pos += apr + len(tokens[tr].strip())
                tr += 1
            position_list.append((tid, tr))
            tid = tr
            
        return sentences, position_list

from .base_rag import BasicRAG
from .utils import nlp

class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        text = ""
        retrieve_question = question
        while True:
            old_len = len(text)
            docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += "Context:\n"
            for i, doc in enumerate(docs):
                prompt += f"[{i+1}] {doc}\n"
            prompt += "Answer in t he same format as before.\n"
            prompt += case + " " + text
            
            # fix sentence
            new_text, tokens, _ = self.generator.generate(prompt, self.generate_max_length)
            if self.use_counter == True:
                self.counter.add_generate(new_text, tokens)
            new_text = new_text.strip()
            sentences = list(nlp(new_text).sents)
            sentences = [str(sent).strip() for sent in sentences]
            if len(sentences) == 0:
                break
            text = text.strip() + " " + str(sentences[0])
            retrieve_question = sentences[0]
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text

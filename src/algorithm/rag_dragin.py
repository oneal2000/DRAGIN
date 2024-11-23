from .base_rag import BasicRAG
from .utils import nlp
import numpy as np
import torch

class AttnWeightRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, attentions, weight):
        # DEBUG: for analysis token's confidence
        # scores_list = []
        # End of DEBUGGGG
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0
        prev = ""
        for sid, sent in enumerate(sentences):
            if 'the answer is' in sent.lower():
                prev = "" if sid == 0 else " ".join(sentences[:sid + 1])
                break
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr
            # value = attenion * (-log prob)
            attns = attentions[tl:tr]
            attns = np.array(attns) / sum(attns)
            value = [attns[i-tl] * weight[i] * (tr-tl) for i in range(tl, tr)] 
            # DEBUG: for analysis token's confidence
            # scores_list += zip(tokens[tl:tr], value, weight)
            # End of DEBUGGGG
            thres = [1 if v > self.hallucination_threshold else 0 for v in value]
            if 1 in thres:
                # hallucinated
                if "check_real_words" in self.__dict__ and self.check_real_words:
                    doc = nlp(sent)
                    real_words = set(token.text for token in doc if token.pos_ in 
                        ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                    def match(tok):
                        for word in real_words:
                            if word in tok:
                                return True
                        return False
                    for i in range(len(thres)):
                        if not match(tokens[tl+i]):
                            thres[i] = 0                
                
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # curr = " ".join(
                #     [tokens[i] if thres[i] == 0 else "[xxx]" for i in range(len(thres))]
                # )
                
                # DEBUG: for analysis token's confidence
                # return True, prev, tokens[tl:tr], thres, scores_list
                # End of DEBUGGGG
                return True, prev, tokens[tl:tr], thres
        # DEBUG: for analysis token's confidence
        # return False, text, None, None,scores_list
        # End of DEBUGGGG
        
        return False, prev, None, None

    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt")
        input_ids = input_ids.to(self.generator.model.device)
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])

        atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[-1][0]

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] in self.generator.sep_ids:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        # 获取幻觉词对应的 attention
        curr_st = len(tokens) - len(curr_tokens)
        atten_tmp = torch.mean(atten_tmp, dim=0)
        attns = []
        for r in range_:
            # att = torch.zeros(atten_tmp.shape[0], input_length)
            att = torch.zeros(input_length)
            for i in range(r[0], r[1] + 1):
                if i == 0:
                    continue
                v = atten_tmp[i-1][:r[0]] # 上一位的
                v = v / v.sum()
                t = torch.zeros(input_length)
                t[:r[0]] = v
                att += t
            att /= (r[1] - r[0] + 1)
            # merge token for att
            att = torch.tensor([att[rr[0]:rr[1]+1].sum() for rr in range_])
            attns.append(att)
            
        # 计算每个超过阈值的 token 在前文的 attentions
        forward_attns = torch.zeros(len(tokens))
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        # 分析词性，保留实词对应的 attns
        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in 
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
        
        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False
        
        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if i >= curr_st and curr_hit[i - curr_st]:
                continue
            if match(tok):
                real_pairs.append((att, tok, i))
        
        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)
        
        real_pairs = sorted(real_pairs, key = lambda x:x[0], reverse=True)
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        return " ".join([x[1] for x in real_pairs])
        
    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        # print(question)
        # print("#" * 20)
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)
            # print('####', prompt)
            # prompt += case + " " + text
            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt, 
                self.generate_max_length, 
                # self.attention_solver, 
                use_entropy = self.method == "dragin", 
                use_logprob = self.method == "attn_prob"
            )
            weight = entropies if self.method == "dragin" else [-v for v in logprobs]

            if self.use_counter == True:
                self.counter.add_generate(new_text, tokens)
            hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                forward_all = [question, text, ptext]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)

                def fetch_last_n_tokens(text, num, tokenizer = self.generator.tokenizer):
                    tokens = tokenizer.tokenize(text)
                    if num >= len(tokens):
                        return text
                    last_n_tokens = tokens[-num:]
                    last_n_sentence = ' '.join(last_n_tokens)
                    return last_n_sentence

                if self.query_formulation == "current":
                    retrieve_question = " ".join(curr_tokens)

                elif self.query_formulation == "current_wo_wrong":
                    retrieve_question = " ".join(
                        list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
                    )

                elif self.query_formulation == "forward_all":
                    retrieve_question = forward_all
                
                elif self.query_formulation == "last_sentence":
                    retrieve_question = self.get_last_sentence(forward_all)
                
                elif self.query_formulation == "last_n_tokens":
                    assert "retrieve_keep_top_k" in self.__dict__
                    retrieve_question = fetch_last_n_tokens(
                        forward_all, self.retrieve_keep_top_k)
                
                elif self.query_formulation == "real_words": 
                    retrieve_question = self.keep_real_words(
                        prev_text = question + " " + text + " " + ptext, 
                        curr_tokens = curr_tokens, 
                        curr_hit = curr_hit,
                    ) 
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                tmp_li = [case, text, ptext.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                # print('#####', prompt)
                # prompt += case + " " + text + " " + ptext.strip()
                new_text, tokens, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, tokens)
                    self.counter.hallucinated += 1
                new_text = self.get_top_sentence(new_text)
                tmp_li = [text.strip(), ptext.strip(), new_text.strip()]
                text = " ".join(s for s in tmp_li if len(s) > 0)
                # text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        # print("#" * 20)
        return text

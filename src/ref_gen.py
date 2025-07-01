# from utils import get_model, load_data, get_model_path, predict
from prompt_template import *
from imports import np,os,gc,torch,AutoTokenizer,AutoModelForCausalLM,argparse,ROOT_DIR,PeftModel,softmax,exp,bm25_retrieve,json, nlp, AutoConfig
import copy
import prompt_template
from augment import one_psg_aug
from encode import one_psg_encode
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils import get_model, predict
from prompt_template import get_prompt

class LlamaGenerator:
    def __init__(self, model_name_or_path):
        self.model, self.tokenizer, self.generation_config = get_model(model_name_or_path)
             
    def generate(self, question, passages=None, answer=None, return_entropies=False):
        self.model.eval()
        input_ids = get_prompt(self.tokenizer, question, passages = passages, answer=answer, with_cot=True)
        input_len = len(input_ids)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.model.device)
        attention_mask = torch.ones(input_ids.shape).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids, 
                attention_mask=attention_mask, 
                **self.generation_config,
            )
        generated_tokens = output.sequences[:, input_len:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        tokens = [token.replace('Ġ', ' ') for token in tokens]
        text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        # print(tokens)
        
        if return_entropies:
            range_ = []
            for i, t in enumerate(tokens):
                if i == 0 or t.startswith(' ') or generated_tokens[0][i] == 13 or tokens[i - 1] == '</s>':
                    range_.append([i, i])
                else:
                    range_[-1][-1] += 1

            tmp = [v.cpu() for v in output.scores]
            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]

            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1] + 1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq)
            
            return text, tokens, seqentropies, range_
        else:
            return text, None, None


class BasicRAG:
    def __init__(self, args):
        args = args.__dict__ 
        for k, v in args.items(): # 使用 setattr 方法将其存储到 self 对象中，使得 self 拥有所有的初始化参数
            setattr(self, k, v)
        self.generator = LlamaGenerator(self.model_name_or_path)

    def inference(self, case): # 直接召回
        docs_ids, docs = bm25_retrieve(case)
        text, _, _ = self.generator.generate(question=case, passages=docs) # 未返回entropies
        print(text, '\n', '='*50)
        return text, {doc_id: doc for doc_id, doc in zip(docs_ids, docs)}
    

class WoRAG:
    def __init__(self, args):
        args = args.__dict__ 
        for k, v in args.items(): # 使用 setattr 方法将其存储到 self 对象中，使得 self 拥有所有的初始化参数
            setattr(self, k, v)
        self.generator = LlamaGenerator(self.model_name_or_path)
    
    def inference(self, case): # 不召回
        text, _, _ = self.generator.generate(question=case)
        print(text, '\n', '='*50)
        return text, {}
    

class SFT_LORA(WoRAG):
    def __init__(self, args):
        super().__init__(args)
        adapter_path = f"{ROOT_DIR}/baselines/{args.model_name}/sft_lora/model/{args.dataset}"
        self.generator.model = PeftModel.from_pretrained(
            self.generator.model, 
            adapter_path,
            adapter_name = "default", 
            is_trainable = False
        )
    
    def inference(self, case):
        text, _, _ = self.generator.generate(question=case)
        print(text, '\n', '='*50)
        return text, {}
        


class TokenRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, entropies, ranges, start_sid):
        # 分句并获取原始 SpaCy Span 对象
        doc = nlp(text)
        sentences = [sent for sent in doc.sents if sent.text.strip()]
        ents = [ent for ent in doc.ents if ent.text.strip()]
        # print(ents)
        # for sent in sentences:
        #     print(sent, end='\n====\n')
        if not ents:
            return text, None, False

        # 定位每个 token range 在文本中的起止位置
        span_positions = []
        pos = 0
        # print(ranges)
        for r in ranges:
            t_str = ''.join(tokens[r[0]:r[1]+1]).strip()
            idx = text[pos:].find(t_str)
            if idx == -1:
                span_positions.append((None, None))  # fallback
            else:
                start = pos + idx
                end = start + len(t_str)
                span_positions.append((start, end))
                pos = end
        # print(span_positions)
        answer_pos = text.find("the answer is")
        # cutoff_char = len(text)
        # if answer_pos != -1: # 找到 the answer is 后第一个句号
        #     after_answer = text[answer_pos:]
        #     dot_idx = after_answer.find(".")
        #     if dot_idx != -1:
        #         cutoff_char = answer_pos + dot_idx + 1 
        # print(start_sid)
        for sid, sent in enumerate(sentences[start_sid:]):
            # print(sid, sent)
            if sent.end_char > answer_pos: # 改成了the answer is一句及之后的句子都不进行modify了
                break
            # 找出当前句子的所有实体，记录顺序
            sentence_ents = [ent for ent in ents if sent.start_char <= ent.start_char < sent.end_char]
            # print(sentence_ents)
            # print('-'*50)
            for eid, ent in enumerate(sentence_ents):
                if ent.start_char - sent.start_char < 2:
                    continue  # 跳过每句的第一个词
                
                # 跳过“，”之后的第一个词
                prev_text = sent.text[:ent.start_char - sent.start_char]
                if prev_text.rstrip().endswith((',', 'both')):
                    continue
                
                matched_ranges = []
                for i, (s, e) in enumerate(span_positions):
                    if s is None or e is None:
                        continue
                    if s >= ent.start_char and e <= ent.end_char:
                        matched_ranges.append(i)
                # print(matched_ranges)
                if matched_ranges:
                    ent_slice = np.array([entropies[i] for i in matched_ranges])
                    e_val = {
                        "avg": np.mean,
                        "max": np.max,
                        "min": np.min,
                        "first": lambda x: x[0] if len(x) > 0 else 0
                    }.get(self.entity_solver, lambda x: 0)(ent_slice)
                    # print(e_val)
                    if e_val > self.hallucination_threshold:
                        # 构造 prev 和 curr
                        prev = text[:ent.start_char].strip()
                        # print(f"prev={prev}")
                        before, after = text[sent.start_char:ent.start_char], text[ent.end_char:sent.end_char]
                        curr = before + "[xxx]" + after
                        return prev, curr, True

        return text, None, False

    def inference(self, case):
        rel_psgs = {}
        final_text = ""         # 从最开头开始到最终输出的完整文本
        halluc_free_text = ""   # 当前为止没有幻觉的完整文本
        sid = 0                 # 步数限制
        w_psgs = False
        docs_ids, docs = [], []  # 当前召回文档
        start_sid = 0
        while True:
            old_len = len(final_text)
            if self.model_name == "llama3_8b" and self.dataset == "2wiki":
                docs_ids, docs = bm25_retrieve(case)
                w_psgs = True
            if w_psgs:
                for docid, doc in zip(docs_ids, docs):
                    rel_psgs[docid] = doc

            new_text, tokens, entropies, ranges = self.generator.generate(
                question=case,
                passages=docs,
                answer=halluc_free_text,
                return_entropies=True
            )
            print('new_text:', new_text)
            print('#'*100)

            # 检查幻觉：返回 halluc_free_text, curr, hallucination
            halluc_free_part, halluc_sent, hallucination = self.modifier(
                new_text, tokens, entropies, ranges, start_sid
            ) # halluc_free_part是从这一轮新生成的地方开始

            if not hallucination:
                w_psgs = False
                # print("no hallucination")
                final_text += new_text.strip()
                break
            else:
                w_psgs = True
                # curr 是当前幻觉句，替换为 [] 以用于 query 检索
                curr = halluc_sent.replace("[xxx]", "[]")
                halluc_free_text = halluc_free_text + " " + halluc_free_part + " " # 从最开头到幻觉处

                # 构造 query
                
                if self.query_formulation == "forward_all" or curr.strip().startswith(("He", "She", "His", "Her")):
                    tmp_all = [case, halluc_free_text] # 因为halluc_free_text包含了curr的幻觉前的部分
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                elif self.query_formulation == "direct":
                    retrieve_question = curr
                else:
                    raise NotImplementedError

                print("="*100)
                print(f"retrieve_question: {retrieve_question}")
                print("="*100)

                # 检索
                docs_ids, docs = bm25_retrieve(retrieve_question)
                if not len(docs):
                    return final_text, rel_psgs
                
                for docid, doc in zip(docs_ids, docs):
                    rel_psgs[docid] = doc

                # 更新当前生成的 text（保留之前未出现幻觉的部分），从最头开始的
                final_text = halluc_free_text.strip()

            sid += 1
            start_sid = 1 # 除了第一次生成以外，后面都忽略掉新生的第一句（也就是上一轮出现幻觉的句子。）
            tokens_count = len(self.generator.tokenizer.encode(final_text))
            if tokens_count > self.generate_max_length or "the answer is" in final_text.lower():
                break

        return final_text, rel_psgs

       

class Moe(TokenRAG):
    def __init__(self, args):
        # self.inference_method = "prag"
        super().__init__(args)
        self.inference_method = getattr(args, "inference_method", "prag")
        self.insert_first = getattr(args, "insert_first", True)
        self.load_adapter_path = os.path.join(
            ROOT_DIR,
            "experts",
            self.model_name,
            "model",
            f"rank={self.lora_rank}_alpha={self.lora_alpha}",
            self.lora_layers,
            f"lr={self.learning_rate}_epoch={self.num_train_epochs}_{'cot' if self.with_cot else 'direct'}",
        )
    
    def load_experts(self, docs_ids, docs):
        adapter_names = []
        for pid, (doc_id, doc) in enumerate(zip(docs_ids, docs)):
            adapter_path = os.path.join(self.load_adapter_path, str(doc_id))
            
            # 检查是否存在
            if not os.path.exists(adapter_path):
                one_psg_aug(doc_id, doc, "llama3.2_1b", f'../other/{self.model_name}_{self.dataset}_tmp_aug.json')
                from argparse import Namespace
                ARGS = Namespace(
                    model_name=self.model_name,
                    dataset=self.dataset,
                    one_aug_path=f'../other/{self.model_name}_{self.dataset}_tmp_aug.json',
                    lora_rank=2,
                    lora_alpha=32,
                    lora_layers=self.lora_layers,
                    learning_rate=3e-4,
                    num_train_epochs=1,
                    with_cot=True,
                    per_device_train_batch_size=1
                )
                
                one_psg_encode(ARGS)


            # 加载 adapter
            try:
                if pid == 0:
                    self.generator.model = PeftModel.from_pretrained(
                        self.generator.model,
                        adapter_path,
                        adapter_name=str(pid),
                        is_trainable=False
                    )
                else:
                    self.generator.model.load_adapter(
                        adapter_path,
                        adapter_name=str(pid)
                    )
            except Exception as e:
                print(f"[Load Error] Failed to load adapter for doc_id={doc_id}: {e}")
                continue

            # # 验证是否为有效 adapter（检查 lora_B）
            # bad_adapter = False
            # for name, param in self.generator.model.named_parameters():
            #     if f"adapters.{pid}" in name and "lora_B" in name:
            #         mean_abs = param.data.abs().mean().item()
            #         if mean_abs < 1e-8:
            #             print(f"[ERROR] Invalid adapter detected for doc_id={doc_id} (mean={mean_abs}) → will retrain.")
            #             bad_adapter = True
            #             break

            # if bad_adapter:
            #     # 重训此 adapter
            #     one_psg_aug(doc_id, doc, self.model_name, f'../other/{self.model_name}_{self.dataset}_tmp_aug.json')
            #     ARGS = Namespace(
            #         model_name=self.model_name,
            #         dataset=self.dataset,
            #         one_aug_path=f'../other/{self.model_name}_{self.dataset}_tmp_aug.json',
            #         lora_rank=2,
            #         lora_alpha=32,
            #         lora_layers=self.lora_layers,
            #         learning_rate=3e-4,
            #         num_train_epochs=1,
            #         with_cot=True,
            #         per_device_train_batch_size=1
            #     )
            #     one_psg_encode(ARGS)

            #     # 重新加载 adapter
            #     self.generator.model.load_adapter(
            #         adapter_path,
            #         adapter_name=str(pid)
            #     )

            adapter_names.append(str(pid))

        # 合并 adapter
        if adapter_names:
            self.generator.model.add_weighted_adapter(
                adapters=adapter_names,
                weights=[1] * len(adapter_names),
                adapter_name="merge",
                combination_type="cat",
            )
            self.generator.model.set_adapter("merge")
        else:
            print("[ERROR] No valid adapters loaded. Merge skipped.")

    
    def unload_experts(self):
        self.generator.model.delete_adapter("merge")
        self.generator.model = self.generator.model.unload()
    
    def inference(self, case):
        rel_psgs = {}
        final_text = ""         # 从最开头开始到最终输出的完整文本
        halluc_free_text = ""   # 当前为止没有幻觉的完整文本
        sid = 0                 # 步数限制
        w_psgs = False
        docs_ids, docs = [], []  # 当前召回文档
        start_sid = 0
        while sid < 6:
            old_len = len(final_text)

            # if sid == 0 and (self.model_name == "llama3_8b" or self.model_name == "qwen2.5_1.5b") and self.dataset == "2wiki":
            if self.insert_first == True and sid == 0:
                docs_ids, docs = bm25_retrieve(case)
                w_psgs = True
                for docid, doc in zip(docs_ids, docs):
                    rel_psgs[docid] = doc
            
            if w_psgs:
                self.load_experts(docs_ids, docs)

            new_text, tokens, entropies, ranges = self.generator.generate(
                question=case,
                passages=None if self.inference_method=="prag" else docs, # always None
                answer=halluc_free_text,
                return_entropies=True
            )
            if w_psgs:
                self.unload_experts()
            # print('new_text:', new_text)
            # print('#'*100)

            # 检查幻觉：返回 halluc_free_text, curr, hallucination
            halluc_free_part, halluc_sent, hallucination = self.modifier(
                new_text, tokens, entropies, ranges, start_sid
            ) # halluc_free_part是从这一轮新生成的地方开始

            if not hallucination:
                w_psgs = False
                # print("no hallucination")
                final_text += " " + new_text.strip()
                break
            else:
                w_psgs = True
                # curr 是当前幻觉句，替换为 [] 以用于 query 检索
                curr = halluc_sent.replace("[xxx]", "[]")
                halluc_free_text = halluc_free_text + " " + halluc_free_part + " " # 从最开头到幻觉处

                # 构造 query
                
                if self.query_formulation == "forward_all" or curr.strip().startswith(("He", "She", "His", "Her")):
                    tmp_all = [case, halluc_free_text] # 因为halluc_free_text包含了curr的幻觉前的部分
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                elif self.query_formulation == "direct":
                    retrieve_question = curr
                else:
                    raise NotImplementedError

                # print("="*100)
                # print(f"retrieve_question: {retrieve_question}")
                # print("="*100)

                # 检索
                docs_ids, docs = bm25_retrieve(retrieve_question)
                if not len(docs):
                    return final_text, rel_psgs
                
                for docid, doc in zip(docs_ids, docs):
                    rel_psgs[docid] = doc

                # 更新当前生成的 text（保留之前未出现幻觉的部分），从最头开始的
                final_text = halluc_free_text.strip()

            sid += 1
            start_sid = 1 # 除了第一次生成以外，后面都忽略掉新生的第一句（也就是上一轮出现幻觉的句子。）
            tokens_count = len(self.generator.tokenizer.encode(final_text))
            if tokens_count > self.generate_max_length or "the answer is" in final_text.lower():
                break

        return final_text, rel_psgs


class PRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        self.load_adapter_path = os.path.join(
            ROOT_DIR,
            "experts",
            self.model_name,
            "model",
            f"rank={self.lora_rank}_alpha={self.lora_alpha}",
            self.lora_layers,
            f"lr={self.learning_rate}_epoch={self.num_train_epochs}_{'cot' if self.with_cot else 'direct'}",
        )
    
    def load_experts(self, docs_ids, docs):
        for pid, (doc_id, doc) in enumerate(zip(docs_ids, docs)):
            adapter_path = os.path.join(self.load_adapter_path, str(doc_id))
            if not os.path.exists(adapter_path):
                continue

            if pid == 0:
                self.generator.model = PeftModel.from_pretrained(
                    self.generator.model,
                    adapter_path,
                    adapter_name="0",
                    is_trainable=False
                )
            else:
                self.generator.model.load_adapter(adapter_path, adapter_name=str(pid))
        self.generator.model.add_weighted_adapter(
            adapters=[str(i) for i in range(len(docs_ids))],
            weights=[1] * len(docs_ids),
            adapter_name="merge",
            combination_type="cat",
        )
        self.generator.model.set_adapter("merge")
        for name, param in self.generator.model.named_parameters():
            if "lora_B" in name and param.data.abs().mean().item() < 0.00000001:
                print(f"[ERROR] invalid expert in {' '.join(doc_ids)}")
    
    def unload_experts(self):
        self.generator.model.delete_adapter("merge")
        self.generator.model = self.generator.model.unload()
    
    def inference(self, case):
        rel_psgs = {}
        docs_ids, docs = bm25_retrieve(case)
        for docid, doc in zip(docs_ids, docs):
            rel_psgs[docid] = doc
        self.load_experts(docs_ids, docs)
        new_text, _, _ = self.generator.generate(
            question=case,
            passages=None, # always None
            answer=None,
            return_entropies=False
        )
        self.unload_experts()

        return new_text, rel_psgs

  
class OldMoe2(TokenRAG):
    def __init__(self, args): # 完全继承了TokenRAG的init
        super().__init__(args)
        self.load_adapter_path = os.path.join(
            ROOT_DIR,
            "experts",
            self.model_name,
            "model",
            f"rank={self.lora_rank}_alpha={self.lora_alpha}",
            f"lr={self.learning_rate}_epoch={self.num_train_epochs}_{'cot' if self.with_cot else 'direct'}",
        )
    
    def load_experts(self, docs_ids, docs):
        for pid, (doc_id, doc) in enumerate(zip(docs_ids, docs)):
            adapter_path = os.path.join(self.load_adapter_path, str(doc_id))
            if not os.path.exists(adapter_path):
                one_psg_aug(doc_id, doc, self.model_name, f'../other/{self.model_name}_{self.dataset}_tmp_aug.json')
                from argparse import Namespace
                ARGS = Namespace(
                    model_name=self.model_name,
                    dataset=self.dataset,
                    one_aug_path=f'../other/{self.model_name}_{self.dataset}_tmp_aug.json',
                    lora_rank=2,
                    lora_alpha=32,
                    learning_rate=3e-4,
                    num_train_epochs=1,
                    with_cot=True,
                    per_device_train_batch_size=1
                )
                one_psg_encode(ARGS)

            if pid == 0:
                self.generator.model = PeftModel.from_pretrained(
                    self.generator.model,
                    adapter_path,
                    adapter_name="0",
                    is_trainable=False
                )
            else:
                self.generator.model.load_adapter(adapter_path, adapter_name=str(pid))
        self.generator.model.add_weighted_adapter(
            adapters=[str(i) for i in range(len(docs_ids))],
            weights=[1] * len(docs_ids),
            adapter_name="merge",
            combination_type="cat",
        )
        self.generator.model.set_adapter("merge")
        for name, param in self.generator.model.named_parameters():
            if "lora_B" in name and param.data.abs().mean().item() < 0.00000001:
                print('error!')
                # os._exit(-1)
    
    def unload_experts(self):
        self.generator.model.delete_adapter("merge")
        self.generator.model = self.generator.model.unload()
    
    def inference(self, examplars, case):
        rel_psgs = {}
        text = ""
        sid, check_sid = 0, 0
        w_psgs, need_experts = False, False ### try
        # w_psgs, need_experts = True, False
        docs_ids, docs = [], []
        while sid < 3:
            old_len = len(text) # 旧的len是text的len
            if not w_psgs:
                prompt = token_rag_wo_psgs_template(examplars, case, text)
            else:
                prompt = token_rag_w_psgs_template(examplars, case, text, docs)
            # print('^'*100)
            # print('initial_prompt:', prompt)
            # print('^'*100)
            if need_experts:
                self.load_experts(docs_ids, docs)
            new_text, tokens, entropies, ranges = self.generator.generate( # 新生成的文本，tokens和logprobs
                prompt, 
                self.generate_max_length, 
                return_entropies=True
            )
            if need_experts:
                self.unload_experts()
                
            print('new_text:', new_text)
            print('#'*100)

            ptext, curr, hallucination, check_sid = self.modifier(
                new_text, tokens, entropies, ranges, check_sid # temperarily set to 0
            ) # 使用 modifier 方法检测并修正幻觉
            if not hallucination:
                w_psgs = False
                print ("no hallu")
                text = new_text.strip() # 如果没有检测到幻觉，则将新的文本加入 text 中
                break # 既然当前位置之后都没有幻觉了那不妨直接break？
            else: # 如果检测到幻觉，则需要基于修正后的文本重新进行检索，生成新的查询并获取更多文档。
                w_psgs = True
                # if sid >= 2 and hallucination:
                #     need_experts = True
                need_experts = True
                
                if self.query_formulation == "direct": # 直接替换法进行检索，只考虑modifier之后的curr，不考虑之前其它生成的内容
                    curr = curr.replace("[xxx]", "[]")
                    retrieve_question = curr # 我把这块改成中括号了，要不然prompt里面不知道哪个地方是空的
                elif self.query_formulation == "forward_all":
                    curr = curr.replace("[xxx]", "[]")
                    # tmp_all = [question, text, ptext] # 用空格连接起来问题、text和ptext
                    tmp_all = [case, text, curr] # I change this 
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented
                # print('retrieve question:', retrieve_question)
                # print('='*100)
                docs_ids, docs = bm25_retrieve(retrieve_question) # 召回回来的docs
                if not len(docs):
                    return text, rel_psgs
                for docid, doc in zip(docs_ids, docs):
                    rel_psgs[docid] = doc
                text = ptext.strip()
                # docs.append(ptext) ###
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            sid += 1
            if tokens_count > self.generate_max_length or "the answer is" in text:
                break
        return text, rel_psgs


def test_retriever(args):
    # 创建 BasicRAG 实例
    rag = BasicRAG(args)

    # 测试 retrieve 方法
    query = "What is George Rankin's occupation?"
    retrieved_doc = rag.bm25_retrieve(query, topk=3)
    for doc in retrieved_doc:
        print('='*80)
        print(doc)

def test_basic_gen(args):
    brag = BasicRAG(args)
    wiki = WikiMultiHopQA()
    examplars = wiki.examplars
    for item in wiki.dataset:
        case = item["question"]
        # print('case', case)
        # 执行推理
        output = brag.inference(examplars, case)
        print(f"{output}\n", '='*100, '\n')

def test_worag_gen(args):
    worag = WoRAG(args)
    wiki = WikiMultiHopQA()
    examplars = wiki.examplars
    for item in wiki.dataset:
        case = item["question"]
        # print('case', case)
        # 执行推理
        output = worag.inference(examplars, case)
        print(f"{output}\n", '='*100, '\n')
             
def test_moe_gen(args):
    moe = Moe(args)     
    moe.moe_generate(args.dataset, 0, 0)
    
def test_modifier(args):
    # 模拟输入数据
    BG = LlamaGenerator(args.model_name_or_path)
    input_text = "Which film came out first, Blind Shaft or The Mask Of Fu Manchu? Let's think step by step."
    text, tokens, entropies, ranges = BG.generate(input_text, 256, True)
    # print('text: ', text, '\n', 'tokens: ', tokens, '\n', 'logprobs: ', logprobs, '\n')
    print(entropies)
    model = TokenRAG(args)

    prev, curr, hallucinated = model.modifier(text, tokens, entropies, ranges)

    print("\nOriginal Text:")
    print(text)
    if hallucinated:
        print("\nModified Text:")
        print(prev + " ---- " + curr)
    else:
        print("\nNo hallucination detected.")

def get_method(method):
    methods = {
        "worag": WoRAG,
        "basicrag": BasicRAG,
        "prag": PRAG,
        "sft_lora": SFT_LORA,
        "token": TokenRAG,
        "moe": Moe,
        "dragin": AttnWeightRAG
    }
    
    if method not in methods:
        raise NotImplementedError(f"Method '{method}' is not implemented.")

    return methods[method]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", type=str, default="BM25")  # 指定使用 BM25
    parser.add_argument("--es_index_name", type=str, default="wiki")  # 可选，Elasticsearch 索引名
    parser.add_argument("--sentence_solver", type=str, default="avg")
    parser.add_argument("--entity_solver", type=str, default="avg")
    parser.add_argument("--dataset", type=str, default="2wiki")
    parser.add_argument("--model_name", type=str, default="llama3.2_1b")
    parser.add_argument("--model_name_or_path", type=str, default="/yeesuanAI08/guest/yanjunxi/LLM/Llama-3.2-1B-Instruct")
    parser.add_argument("--query_formulation", type=str, default="direct")
    parser.add_argument("--hallucination_threshold", type=int, default=0.5)
    parser.add_argument("--generate_max_length", type=int, default=256)
    args = parser.parse_args()  # 适用于 Jupyter 或直接运行
    test_modifier(args)
    
    # test_retriever(args)
    # test_moe_gen(args)
    
    # test_basic_gen(args)
    # test_worag_gen(args)


class oldMoe(TokenRAG):
    def __init__(self, args): # 完全继承了TokenRAG的init
        super().__init__(args)

    def moe_generate(self, input_text, max_length, test_id, return_logprobs=False):
        prompt_template.get_fewshot(self.dataset)
        model, tokenizer, generation_config = get_model(
            get_model_path(self.model_name),
            max_new_tokens = 200,
        )
        # print('Params num Before:', sum(p.numel() for p in model.parameters()))
        load_adapter_path = os.path.join(
            ROOT_DIR, 
            "offline", 
            self.model_name, 
            f"rank=2_alpha=32",
            self.dataset,
            f"lr=0.0003_epoch=1_cot",
            f"aug_model={self.model_name}",
            "total",
            f"data_{test_id}"
        )
        num_psgs = sum(1 for entry in os.scandir(load_adapter_path) if entry.is_dir())
        for pid in range(num_psgs):
            adapter_path = os.path.join(load_adapter_path, f"passage_{pid}")
            if pid == 0:
                model = PeftModel.from_pretrained(
                    model, 
                    adapter_path,
                    adapter_name = "0", 
                    is_trainable = False
                )
            else:
                model.load_adapter(adapter_path, adapter_name = str(pid)) 
        # merge
        if num_psgs > 0:
            model.add_weighted_adapter(
                adapters = [str(i) for i in range(num_psgs)], 
                weights = [1] * num_psgs,
                adapter_name = "merge", 
                combination_type = "cat", # 拼接多个适配器的权重
            )
            model.set_adapter("merge")
        
        def get_pred(model):
            text = predict(model, tokenizer, generation_config, input_text, with_cot=True)
            pred = {
                "test_id": test_id, 
                "question": input_text, 
                "answer": "", 
                "text": text,
            }
            pred.update(evaluate(text, "", True))
            # print(pred)
            return pred['eval_predict']
        
        # print('Params num After:', sum(p.numel() for p in model.parameters()))
        old_model, old_tokenizer = self.generator.model, self.generator.tokenizer
        self.generator.model, self.generator.tokenizer = model, tokenizer
        
        # generated_text, _, _ = self.generator.generate(input_text, max_length)
        generated_text = get_pred(model)
        
        self.generator.model, self.generator.tokenizer = old_model, old_tokenizer # backtrace
        if num_psgs > 0:
            model.delete_adapter("merge") # 删除该adapter
            model = model.unload()
        
        torch.cuda.empty_cache() # 清理缓存
        gc.collect()
        return generated_text, None, None
        
        
    def inference(self, question, demo, case, test_id):
        rel_psgs = set()
        text = ""
        while True:
            old_len = len(text) # 旧的len是text的len
            prompt = f"You should answer my question through reasoning step-by-step. Here are {len(demo)} examples:\n"
            prompt += "".join([f"Example[{idx+1}]: {d['case']}\n" for idx, d in enumerate(demo)]) # 组合所有 demo 作为前缀
            prompt += '\n' + case + " " + text # 加上 case(问题) 和已有的生成文本
            new_text, tokens, logprobs = self.generator.generate( # 新生成的文本，tokens和logprobs
                prompt, 
                self.generate_max_length, 
                return_logprobs=True
            )
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs) # 使用 modifier 方法检测并修正幻觉部分。
            if not hallucination:
                print ("no hallu")
                text = text.strip() + " " + new_text.strip() # 如果没有检测到幻觉，则将新的文本加入 text 中
                print('@'*100)
                print('no_hallu_text: ', text)
                print('@'*100)
                break
                
            else: # 如果检测到幻觉，则需要基于修正后的文本重新进行检索，生成新的查询并获取更多文档。
                if self.query_formulation == "direct": # 直接替换法进行检索，只考虑modifier之后的curr，不考虑之前其它生成的内容
                    retrieve_question = curr.replace("[xxx]", "[]") # 我把这块改成中括号了，要不然prompt里面不知道哪个地方是空的
                elif self.query_formulation == "forward_all":
                    curr = curr.replace("[xxx]", "[]")
                    tmp_all = [question, text, curr] # I change this 
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented
                print('='*100)
                print(f"retrieve_question: {retrieve_question}")
                print('='*100)
                docs = bm25_retrieve(retrieve_question, topk=self.retrieve_topk) # 召回回来的docs

                # =================== #
                for doc in docs:
                    rel_psgs.add(doc)
                # =================== #
                prompt = f"You should answer my question through reasoning step-by-step. Here are {len(demo)} examples:\n"
                prompt += "".join([f"Example[{idx+1}]: {d['case']}\n" for idx, d in enumerate(demo)])
                prompt += "\nGiven the following information about this question:\n"
                for i, doc in enumerate(docs): # 加上召回回来的docs作为“context”
                    prompt += f"[{i+1}] {doc}\n"
                prompt += '\n' + case + " " + text + " " + ptext.strip() # case + text + ptext 作为已有信息
                print('*' * 100)
                print(prompt)
                print('*' * 100)
                new_text, _, _ = self.moe_generate(prompt, self.generate_max_length, test_id) # 不管tokens和logprobs了
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip() # 拼接
                
                print('@'*100)
                print('has_hallu_text: ', text)
                print('@'*100)
                break

            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text, rel_psgs

 
# class PRAG:
#     def __init__(self, args):
#         args = args.__dict__  # 转为字典形式方便设置属性
#         for k, v in args.items():
#             setattr(self, k, v)

#         self.fulldata = self.read_mapping(f"../other/{self.dataset}_mapping.jsonl")
#         self.generator = LlamaGenerator(self.model_name_or_path)

#         self.load_adapter_path = os.path.join(
#             ROOT_DIR,
#             "offline",
#             self.model_name,
#             f"rank={self.lora_rank}_alpha={self.lora_alpha}",
#             self.dataset,
#             f"lr={self.learning_rate}_epoch={self.num_train_epochs}_{'cot' if self.with_cot else 'direct'}",
#         )

#     def read_mapping(self, mapping_path):
#         mapping = []
#         with open(mapping_path, 'r') as file:
#             for line in file:
#                 tmp = json.loads(line.strip())
#                 mapping.append(tmp)
#         return mapping

#     def inference(self, examplars, question, test_id):
#         passages = self.fulldata[test_id]["doc"]
        
#         # print('before', sum(p.numel() for p in self.generator.model.parameters()))
        
#         for pid in range(len(passages)):
#             adapter_path = os.path.join(self.load_adapter_path, f"data_{test_id}", f"passage_{pid}")
#             if pid == 0:
#                 self.generator.model = PeftModel.from_pretrained(
#                     self.generator.model,
#                     adapter_path,
#                     adapter_name="0",
#                     is_trainable=False
#                 )
                
#             else:
#                 self.generator.model.load_adapter(adapter_path, adapter_name=str(pid))

#         # merge adapters
#         self.generator.model.add_weighted_adapter(
#             adapters=[str(i) for i in range(len(passages))],
#             weights=[1] * len(passages),
#             adapter_name="merge",
#             combination_type="cat",
#         )
#         self.generator.model.set_adapter("merge")
#         # for name, param in self.generator.model.named_parameters():
#         #     if "lora_A" in name or "lora_B" in name:
#         #         print(name, param.data.abs().mean().item())

#         # self.generator.model = self.generator.model.merge_and_unload()
#         # self.generator.model = model
#         # print('after', sum(p.numel() for p in self.generator.model.parameters()))
#         # print("PEFT adapter names:", self.generator.model.peft_config.keys())
#         # def lora_forward_hook(module, input, output):
#         #     print(f"[LORA FORWARD] {module.__class__.__name__} | Output norm: {output.norm().item()}")

#         # 确保只 hook 到 lora_A 和 lora_B
#         # for name, module in self.generator.model.named_modules():
#         #     if 'lora_A' in name or 'lora_B' in name:
#         #         module.register_forward_hook(lora_forward_hook)
#         # for name, param in self.generator.model.named_parameters():
#         #     if param.requires_grad:
#         #         print(f"{name} - grad: {param.grad is not None}")


#         # print(self.generator.model)
#         # print("Active adapter:", self.generator.model.active_adapter)


#         # prompt = wo_rag_template(examplars, question) if self.method == "prag" else basic_rag_template(examplars,question,passages)
#         prompt = basic_rag_template(examplars,question,passages)
#         # print('prompt:', prompt, '-'*80)
#         text, _, _ = self.generator.generate(prompt, self.generate_max_length)
#         print(text, '\n', '=' * 50)
#         # self.generator.model = model_backup
#         self.generator.model.delete_adapter("merge")
#         self.generator.model = self.generator.model.unload()

#         # print('end', sum(p.numel() for p in self.generator.model.parameters()))
#         if test_id % 50 == 0: 
#             torch.cuda.empty_cache()
#             gc.collect()

#         # return text, passages
#         return text, {}
        


class AttnWeightRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, attentions, weight):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0
        for sid, sent in enumerate(sentences):
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
                return True, prev, tokens[tl:tr], thres
        return False, text, None, None

    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt")
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])

        atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[-1][0]

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
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
        
    def inference(self, demo, question, case):
        # assert self.query_formulation == "direct"
        # print(question)
        # print("#" * 20)
        text = ""
        while True:
            old_len = len(text)
            # prompt = "".join([d["case"]+"\n" for d in demo]) ### original
            prompt = wo_rag_template(demo, case) ### new
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

                docs = bm25_retrieve(retrieve_question)
                # prompt = "".join([d["case"]+"\n" for d in demo])
                prompt = wo_rag_template(demo, case) ### new
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                tmp_li = [case, text, ptext.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                # print('#####', prompt)
                # prompt += case + " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)

                new_text = self.get_top_sentence(new_text)
                tmp_li = [text.strip(), ptext.strip(), new_text.strip()]
                text = " ".join(s for s in tmp_li if len(s) > 0)
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        # print("#" * 20)
        return text(base)
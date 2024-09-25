from DifyService import DifyLLMaaS
import os
import pandas as pd
from zeus_utility.parallel_executor import QueueExecutor
from tqdm import tqdm
import json
from rag import rag

BANK_SET=set()

document=rag.default_knowledge_graph_rag("hi", top_v=10, top_r=5)
llmaas=DifyLLMaaS(output_path="banks_entities")
df_path=os.path.join("dataset","entities_transcripts.jsonl")
df=pd.read_json(df_path,lines=True)

for index, row in df.iterrows():
    bank=row["bank"]
    BANK_SET.add(bank)
    pass

banks_entities_map={}
banks_entities_interface_map={}
def generate_bank_entities(bank_name:str):
    banks_entities_map[bank_name]={}
    banks_entities_interface_map[bank_name]={}
    def qna(key:str,query:str):
        document=rag.default_knowledge_graph_rag(query, top_v=100, top_r=20)
        res=llmaas.call({"query":query,"document":document})
        answer=res["data"]["outputs"]["text"]
        banks_entities_map[bank_name][key]=answer
        banks_entities_interface_map[bank_name][key]=""
        progress_bar.update(1)
    qna("profit",f"how much profit does {bank_name} make?")
    qna("cost",f"can you tell me something about {bank_name} cost?")
    qna("challenge",f"what are the challenges faced by {bank_name}?")
    qna("opportunity",f"what are the opportunities for {bank_name}?")
    qna("plan",f"what are the future plans of {bank_name}?")
    qna("significant_one_time_gain_or_loss",f"Any significant one time gain or loss for {bank_name}?")
    qna("dividend_policy",f"what is the dividend policy of {bank_name}?")
    return

executor=QueueExecutor(threads_count=10)
for bank_name in BANK_SET:
    executor.add_task(generate_bank_entities,bank_name=bank_name)
    pass
progress_bar=tqdm(total=len(executor)*7)
executor.execute()
df=pd.DataFrame(banks_entities_map)
df_path=os.path.join("dataset","banks_entities.json")
df.to_json(df_path,force_ascii=False)
df=pd.DataFrame(banks_entities_interface_map)
df_path=os.path.join("dataset","banks_entities_interface.json")
df.to_json(df_path,force_ascii=False)
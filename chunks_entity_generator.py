from DifyService import DifyWorkflows
import os
import pandas as pd
from zeus_utility.parallel_executor import QueueExecutor
from tqdm import tqdm
import json

BANK_SET=set()

workflow=DifyWorkflows(output_path="entities")
df_path=os.path.join("dataset","questionsed_transcripts.jsonl")
df=pd.read_json(df_path,lines=True)

for index, row in df.iterrows():
    bank=row["bank"]
    BANK_SET.add(bank)
    pass

def generate_entity_from_chunk(row:pd.Series):
    row=row.to_dict()
    chunk=row["chunk"]
    bank=row["bank"]
    chunk_hash=row["hash"]

    # region One Sentence Summary Generation
    query=f"""Try to summarize the document and include the bank {bank} information into one sentence.
This document was extract from the {bank} Bank.
Here is the document:
{chunk}"""
    inputs={"query":query,"function":"entities"}
    res=workflow.call(inputs)
    summary=res["data"]["outputs"]["text"]
    row["summary"]=summary
    # endregion

    # region Speakers Information Extraction
    inputs={"document":f"- {chunk}","function":"speaker_detection"}
    res=workflow.call(inputs)
    speakers:list=res["data"]["outputs"]["speakers_information"]
    for speaker in speakers:
        # speaker["chunk_hash"]=chunk_hash
        if "name" in speaker:
            name:str=speaker["name"]
            if name is None or name.strip()=="" or "Unknown" in name.strip() or "Operator" in name or "Speaker" in name:
                speakers.remove(speaker)
            else:
                if name not in speaker_chunks_map:
                    speaker_chunks_map[name]=[]
                speaker_chunks_map[name].append(chunk_hash)
        else:
            speakers.remove(speaker)
    row["speakers"]=speakers
    # endregion

    transformed_rows.append(row)

def analysis_entities():
    all_speakers:dict[str,list]={}
    for row in transformed_rows:
        bank=row["bank"]
        speakers=row["speakers"]
        for speaker in speakers:
            if bank not in all_speakers:
                all_speakers[bank]=[]
            all_speakers[bank].append(speaker)
            pass
        pass

def map_speakers_to_chunks():
    for row in transformed_rows:
        speakers=row["speakers"]
        for speaker in speakers:
            name=speaker["name"]
            if name in speaker_chunks_map:
                speaker["chunk_hash"]=speaker_chunks_map[name]
            pass
        pass

def insert_summaries_to_chunk():
    other_summaries={}
    for row in transformed_rows:
        summary=row["summary"]
        if row["bank"] not in other_summaries:
            other_summaries[row["bank"]]=[]
        other_summaries[row["bank"]].append(summary)
        pass
    for row in (transformed_rows):
        row["full_summary"]="\n".join(other_summaries[row["bank"]])
        pass


transformed_rows=[]
speaker_chunks_map:dict[str,list]={}
executor=QueueExecutor(threads_count=1)
for index, row in df.iterrows():
    executor.add_task(generate_entity_from_chunk,row=row)
    pass
executor.execute(progress_bar=tqdm(total=len(executor)))
analysis_entities()
map_speakers_to_chunks()
insert_summaries_to_chunk()
df=pd.DataFrame(transformed_rows)
df_path=os.path.join("dataset","entities_transcripts.jsonl")
df.to_json(df_path,orient="records", lines=True)

entities_interface=transformed_rows.copy()
for row in entities_interface:
    for key in row.keys():
        row[key]=""
    pass
df=pd.DataFrame(entities_interface)
df_path=os.path.join("dataset","entities_interface.jsonl")
df.to_json(df_path,orient="records", lines=True)
from DifyService import DifyWorkflows
import os
import pandas as pd
from zeus_utility.parallel_executor import QueueExecutor
from tqdm import tqdm

workflow=DifyWorkflows(output_path="questions")
df_path=os.path.join("dataset","chunked_transcripts.jsonl")
df=pd.read_json(df_path,lines=True)

def generate_questions_from_chunk(row:pd.Series):
    row=row.to_dict()
    chunk=row["chunk"]
    bank=row["bank"]
    questions=[]
    for _ in range(10):
        document=f"""This document was extract from the {bank} Bank, try to include the bank {bank} information into the generated question.
Here is the document:
{chunk}"""
        inputs={"document":document,"function":"Q&A","existing_questions":"\n".join(questions)}
        res=workflow.call(inputs)
        question=res["data"]["outputs"]["text"]
        questions.append(question)
    workflow.save_cache()
    row["questions"]=questions
    transformed_rows.append(row)

transformed_rows=[]
executor=QueueExecutor(threads_count=10)
for index, row in df.iterrows():
    executor.add_task(generate_questions_from_chunk,row=row)
    pass
executor.execute(progress_bar=tqdm(total=len(executor)))
df=pd.DataFrame(transformed_rows)
df_path=os.path.join("dataset","questionsed_transcripts.jsonl")
df.to_json(df_path,orient="records", lines=True)
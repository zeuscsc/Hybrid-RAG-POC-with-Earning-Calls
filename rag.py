from pymilvus import MilvusClient, DataType,db,connections, AnnSearchRequest
import os
import pandas as pd
import json
from neo4j import GraphDatabase
import numpy as np

# CLUSTER_DOMAIN="host.docker.internal"
CLUSTER_DOMAIN = "localhost"
PORT = 19530
CLUSTER_ENDPOINT = f"http://{CLUSTER_DOMAIN}:{PORT}"
DATABASE_NAME = "HSBC"
COLLECTION_NAME = "banks_earnings_calls"
VECTOR_DB_USERNAME = "developers"
VECTOR_DB_PASSWORD = "developers"

DEFAULT_EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'

KG_URI = "neo4j://localhost:7687"
KG_USER = "neo4j"
KG_PASSWORD="meCfTH39XssP92e"

CHUNKS_SEPRATOR_STRING = "\n\n"
MAX_APPROXIMATE_TOKENS = 1280000

class Neo4jConnection:
    def __init__(self, uri=KG_URI, user=KG_USER, password=KG_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()

    def query(self, query, parameters=None):
        with self.driver.session() as session:
            return list(session.run(query, parameters))
        
class OrderedSet:
    def __init__(self):
        self.dict = {}

    def add(self, value):
        self.dict[value] = None

    def remove(self, value):
        if value in self.dict:
            del self.dict[value]

    def __contains__(self, value):
        return value in self.dict

    def __iter__(self):
        return iter(self.dict.keys())

    def __len__(self):
        return len(self.dict)

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self.dict.keys())})"

    def __getitem__(self, index):
        return list(self.dict.keys())[index]
class RAG:
    model = None
    reranker = None
    def __init__(self,milvus_client=MilvusClient(db_name=DATABASE_NAME,uri=CLUSTER_ENDPOINT,user=VECTOR_DB_USERNAME,password=VECTOR_DB_PASSWORD),
                 neo4j_connection=Neo4jConnection(),
                 max_approximate_tokens=MAX_APPROXIMATE_TOKENS):
        self.milvus_client = milvus_client
        self.neo4j_connection = neo4j_connection
        self.max_approximate_tokens=max_approximate_tokens
        pass
    
    def words_size_to_approximate_tokens_size(words_size):
        return int(words_size * (4/3))
    def approximate_tokens_counter(document:str):
        words_size=len(document.split())
        tokens_size=RAG.words_size_to_approximate_tokens_size(words_size)
        return tokens_size
    def format_rag_documents(documents_map:dict):
        rag_docoment=""
        for key in documents_map:
            rag_docoment+=f"The following is document for {key}:\n\n{documents_map[key]}"
        return rag_docoment
    
    def get_embeddings(self, queries)->dict[str, np.ndarray]:
        from FlagEmbedding import BGEM3FlagModel
        if RAG.model is None:
            RAG.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
            RAG.model.model.to('cuda')
            print("Model loaded.")
        embeddings = self.model.encode(queries, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dense_vectors = embeddings['dense_vecs']
        lexical_weights = embeddings['lexical_weights']
        return {"dense_vectors": dense_vectors, "sparse_vectors": lexical_weights}
    def similarity_sort(self,sentences_1 :list[str], sentences_2 :list[str]):
        embeddings_1 =self.get_embeddings(sentences_1)['dense_vectors']
        embeddings_2 =self.get_embeddings(sentences_2)['dense_vectors']
        similarity = embeddings_1 @ embeddings_2.T
        sorted_sentences_2 = [sentences_2[i] for i in similarity.argsort()[0][::-1]]
        return sorted_sentences_2
    def rerank(self,query:str,records:list,topk):
        from FlagEmbedding import FlagReranker
        if self.reranker is None:
            self.reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
            self.reranker.model.to('cuda')
            print("Reranker loaded.")
        chunks=[]
        for record in records:
            content=record["entity"]["content"]
            summary=record["entity"]["summary"]
            full_summary:str=record["entity"]["full_summary"]
            chunk=full_summary.replace(summary,content)
            chunks.append(chunk)
            chunk=record["entity"]["content"]
            chunks.append(chunk)
        rerank_pairs=[(query,chunk) for chunk in chunks]
        reranker_scores=self.reranker.compute_score(rerank_pairs,batch_size=32)
        best_documents=self.sort_by_reranker_scores(records,reranker_scores)
        best_documents=best_documents[:topk]
        df=pd.DataFrame(best_documents)
        df.to_json("reranked.jsonl",orient="records",lines=True)
        return best_documents
    def sort_by_reranker_scores(self,documents,reranker_scores):
        # print(reranker_scores)
        # print(documents)
        # return [doc for _, doc in sorted(zip(reranker_scores, documents), reverse=True)]
        paired_list = list(zip(reranker_scores, documents))
        paired_list.sort(key=lambda x: x[0], reverse=True)
        sorted_documents = [doc for _, doc in paired_list]
        return sorted_documents
    def filter_unique_knowledge_graph_old(results):
        unique_kg_set=dict()
        for result in results[0]:
            hash_id=result["entity"]['hash']
            if hash_id not in unique_kg_set:
                if "cypher" in result["entity"]:
                    unique_kg_set[hash_id]={"cypher":result['entity']['cypher']}
                else:
                    unique_kg_set[hash_id]={"hash":hash_id}
        return unique_kg_set
    def filter_unique_unique_hash_id(results):
        unique_kg_set=OrderedSet()
        for result in results:
            hash_id=result["entity"]['hash']
            if hash_id not in unique_kg_set:
                unique_kg_set.add(hash_id)
        return unique_kg_set
    def filter_unique_vector_db(self,results):
        chunk_set=OrderedSet()
        for result in results:
            chunk=result["entity"]['content']
            if chunk not in chunk_set:
                current_document:str=CHUNKS_SEPRATOR_STRING.join(chunk_set)
                if RAG.approximate_tokens_counter(current_document)+RAG.approximate_tokens_counter(chunk) > self.max_approximate_tokens:
                    break
                chunk_set.add(chunk)
        return chunk_set
    def retrive_documents(self, query:str, top_v=10, top_r=5):
        embeddings:dict[str,np.ndarray]=self.get_embeddings([query])
        dense_vectors=embeddings['dense_vectors'].tolist()
        dense_search_params = {"metric_type": "IP"}
        res = self.milvus_client.search(COLLECTION_NAME, data=dense_vectors, search_params=dense_search_params, 
                                 output_fields=["id","hash","bank","content","summary","full_summary"], topk=top_v)
        df = pd.DataFrame(res[0])
        df.to_json("vector_search.jsonl",orient="records",lines=True)
        # return res[0]
        response=self.rerank(query,res[0],top_r)
        return response
    def vector_db_rag(self,query:str,top_v=10, top_r=5)->str:
        results=self.retrive_documents(query,top_v, top_r)
        documents=self.filter_unique_vector_db(results)
        return "\n\n".join(documents)

    def select_via_cypher_query(self,cypher_query:str):
        records = self.neo4j_connection.query(cypher_query)
        return records
    def select_via_chunks(self,hash_id):
        query = """MATCH (c:Chunk {hash: $hash})-[:HAS_CHUNK]-(b:Bank)
OPTIONAL MATCH (c)-[:SPOKE_IN]-(s:Speaker)
RETURN c as chunk, b as bank, collect(s) AS speakers"""
        records = self.neo4j_connection.query(query,{"hash":hash_id})
        return records
    def format_default_kg_chunks_rag_documents_old(self,kg_sets:dict)->str:
        documents_map:dict[str,str]={}
        for hash_id,kg_set in kg_sets.items():
            if "cypher" in kg_set:
                records=self.select_via_cypher_query(kg_set["cypher"])
                for record in records:
                    bank_name=kg_set["cypher"]
            else:
                records=self.select_via_chunks(hash_id)
                for record in records:
                    bank = record["bank"]
                    bank_name=bank["name"]
                    chunk=record["chunk"]
                    summary=chunk["summary"]
                    original_text=chunk["chunk"]
                    full_summary:str=bank["full_summary"]
                    temp_documents_map:dict[str,str]=json.loads(json.dumps(documents_map))
                    if bank_name not in temp_documents_map:
                        temp_documents_map[bank_name]=f"{bank_name}'s 2024 Q1 Earnings Call:\n{full_summary.replace(summary,original_text)}"
                    else:
                        temp_documents_map[bank_name].replace(summary,original_text)
                    if RAG.approximate_tokens_counter(RAG.format_rag_documents(temp_documents_map))>self.max_approximate_tokens:
                        return documents_map
                    documents_map=temp_documents_map
                    pass
            pass
        return documents_map
    def format_default_kg_chunks_rag_documents(self,unique_hash_ids:OrderedSet)->str:
        documents_map:dict[str,str]={}
        for hash_id in unique_hash_ids:
            records=self.select_via_chunks(hash_id)
            for record in records:
                bank = record["bank"]
                bank_name=bank["name"]
                chunk=record["chunk"]
                summary=chunk["summary"]
                original_text=chunk["chunk"]
                full_summary:str=bank["full_summary"]
                temp_documents_map:dict[str,str]=json.loads(json.dumps(documents_map))
                if bank_name not in temp_documents_map:
                    temp_documents_map[bank_name]=f"{bank_name}'s 2024 Q1 Earnings Call:\n{full_summary.replace(summary,original_text)}"
                else:
                    temp_documents_map[bank_name].replace(summary,original_text)
                if RAG.approximate_tokens_counter(RAG.format_rag_documents(temp_documents_map))>self.max_approximate_tokens:
                    return documents_map
                documents_map=temp_documents_map
                pass
            pass
        return documents_map
    def format_default_kg_nodes_rag_documents(self,banks:str,information_type:str)->str:
        banks_names=[bank.strip() for bank in banks.split(",")]
        information_types=[information_type.strip() for information_type in information_type.split(",")]
        def get_property_values(conn: Neo4jConnection, banks: list, properties: list):
            query = """
            MATCH (b:Bank)-[r:HAS_PROPERTY]->(p:Property)
            WHERE b.name IN $banks AND p.name IN $properties
            RETURN b.name AS bank, p.name AS property, r.value AS value
            """
            result = conn.query(query, {"banks": banks, "properties": properties})
            return result
        records = get_property_values(self.neo4j_connection, banks_names, information_types)
        documents_map:dict[str,str]={}
        for record in records:
            bank=record["bank"]
            property=record["property"]
            value=record["value"]
            if bank not in documents_map:
                documents_map[f"{bank} {property}"]=value
            else:
                documents_map[f"{bank} {property}"]+="\n"+value
            pass
        return documents_map
    
    def default_knowledge_graph_rag(self,query:str,banks:str,information_type:str):
        rag_documents_map:dict={}
        rag_documents_map.update(self.format_default_kg_nodes_rag_documents(banks,information_type))
        if len(rag_documents_map.keys())<2:
            results=self.retrive_documents(query,top_v=10, top_r=3)
            unique_hash_id=RAG.filter_unique_unique_hash_id(results)
            rag_documents_map.update(self.format_default_kg_chunks_rag_documents(unique_hash_id))
        rag_documents_str:str=RAG.format_rag_documents(rag_documents_map)
        df=pd.DataFrame(rag_documents_map.items(),columns=["bank","content"])
        df.to_json("default_knowledge_graph_rag.json")
        return rag_documents_str
    
    def select_speakers_via_banks(self,banks_names:list[str]|None=None):
        if banks_names is None or len(banks_names)==0:
            query = """MATCH (s:Speaker)-[:ATTENDED]-(b:Bank) RETURN s.name as speakers_names, b.name as banks_names"""
            records = self.neo4j_connection.query(query)
            return records
        else:
            query = """MATCH (s:Speaker)-[:ATTENDED]-(b:Bank{name: $bank_name}) RETURN s.name as speakers_names, b.name as banks_names"""
            all_records=[]
            for bank_name in banks_names:
                records = self.neo4j_connection.query(query,{"bank_name":bank_name})
                all_records.extend(records)
            return all_records
    def specific_cypher_query_test_knowledge_graph_rag(self):
        records=self.specific_cypher_query_test()
        speakers_in_banks_map:dict[str,list]={}
        for record in records:
            bank_name=record["bank_name"]
            speaker_name=record["speaker_name"]
            if bank_name not in speakers_in_banks_map:
                speakers_in_banks_map[bank_name]=[speaker_name]
            else:
                speakers_in_banks_map[bank_name].append(speaker_name)
            pass
        rag_documents_str=""
        for bank_name in speakers_in_banks_map:
            rag_documents_str+=f"\n\nName List of People who have spoke in Bank < {bank_name} > Earnings Call:\n"
            rag_documents_str+="\n".join(speakers_in_banks_map[bank_name])
        return rag_documents_str
    
    def specific_cypher_query_test(self):
    # MATCH (s:Speaker)-[r:ATTENDED]->(b:Bank)
        query = """
    MATCH (s:Speaker)-[:SPOKE_IN]->(:Chunk)<-[:HAS_CHUNK]-(b:Bank)
    WITH s, COUNT(DISTINCT b) AS bank_count
    WHERE bank_count > 1
    MATCH (s:Speaker)-[:SPOKE_IN]->(:Chunk)<-[:HAS_CHUNK]-(b:Bank)
    RETURN DISTINCT s.name AS speaker_name, b.name AS bank_name
    """
        records = self.neo4j_connection.query(query)
        return records
    
    def cypher_query(self,cypher_query:str,params:dict=None):
        if params is None:
            records=self.neo4j_connection.query(cypher_query)
        else:
            records=self.neo4j_connection.query(cypher_query,params)
        serialized_records=[]
        for record in records:
            serialized_record={}
            for key in record.keys():
                serialized_record[key]=record[key]
            serialized_records.append(serialized_record)
        return serialized_records
rag=RAG()
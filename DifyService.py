import requests
import os
import pandas as pd
import hashlib
import json

def generate_hash(text:str):
    return hashlib.md5(text.encode()).hexdigest()

class DifyWorkflows:
    def __init__(self,output_path:str,cache_dir:str="dify_cache",domain:str="localhost",api_key:str="app-IK0NGgiK9nKb2iWLnqHTANeQ"):
        self.domain=domain
        self.cache = {}
        self.output_path=output_path
        self.cache_dir=cache_dir
        self.api_key=api_key
        self.load_cache()
    def save_cache(self):
        cache_path=os.path.join(self.cache_dir,f"{os.path.basename(self.output_path)}.json")
        cache_df=pd.DataFrame(self.dify_response_cache)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        cache_df.to_json(cache_path,force_ascii=False)
    def load_cache(self):
        cache_path=os.path.join(self.cache_dir,f"{os.path.basename(self.output_path)}.json")
        if os.path.exists(cache_path):
            cache_df=pd.read_json(cache_path)
            self.dify_response_cache=cache_df.to_dict(orient="dict")
        else:
            self.dify_response_cache={}
    def cache_migration(self,old_inputs,new_inputs):
        if generate_hash(json.dumps(old_inputs)) in self.dify_response_cache:
            response=self.dify_response_cache[generate_hash(json.dumps(old_inputs))]
            self.dify_response_cache[generate_hash(json.dumps(new_inputs))]=response
            self.dify_response_cache.pop(generate_hash(json.dumps(old_inputs)))
            self.save_cache()
    def cache_delete(self,inputs):
        if generate_hash(json.dumps(inputs)) in self.dify_response_cache:
            self.dify_response_cache.pop(generate_hash(json.dumps(inputs)))
            self.save_cache()

    def call(self,inputs):
        if generate_hash(json.dumps(inputs)) in self.dify_response_cache:
            return self.dify_response_cache[generate_hash(json.dumps(inputs))]
        url = f'http://{self.domain}/v1/workflows/run'
        api_key = self.api_key  # replace this with your actual API key

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            "inputs": inputs,
            "response_mode": "blocking",
            "user": "abc-123"
        }
        response=None
        try:
            response = requests.post(url, headers=headers, json=data)
            response_json = response.json()
            self.dify_response_cache[generate_hash(json.dumps(inputs))]=response_json
            if response.status_code==200:
                self.save_cache()
            return response_json
        except Exception as e:
            print(e)
            if response is not None:
                print(data)
                print(response.status_code)
                print(response.json())
            return None
        
class DifyLLMaaS(DifyWorkflows):
    def __init__(self,output_path:str,cache_dir:str="dify_cache",domain:str="localhost",api_key:str="app-O1mn4q38H5SBBPOPNEqV23Dg"):
        super().__init__(output_path,cache_dir,domain,api_key)
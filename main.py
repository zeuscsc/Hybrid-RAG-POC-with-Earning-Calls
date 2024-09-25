from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import rag

app = FastAPI()

class UserRequest(BaseModel):
    query: str=""
    banks: str=""
    information_type: str=""
class CypherRequest(UserRequest):
    cypher_query:str=""
    param:dict=None

@app.post("/hybrid_rag/")
async def hybrid_rag(request: UserRequest):
    document=rag.default_knowledge_graph_rag(query=request.query, banks=request.banks, information_type=request.information_type)
    return {"document":document}

@app.post("/cypher_query/")
async def cypher_query(request: CypherRequest):
    records=rag.cypher_query(request.cypher_query,request.param)
    return {"records":records}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3240)
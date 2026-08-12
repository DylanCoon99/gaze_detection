# FastAPI app entry point and route definitions.
# Exposes endpoints: GET /models, GET /models/{name}, GET /pareto
from fastapi import FastAPI
import mlflow_client

# Initialize the application instance
app = FastAPI()

# Define a root GET endpoint
@app.get("/models")
def read_root():



    return 

# Define a parameterized GET endpoint
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}

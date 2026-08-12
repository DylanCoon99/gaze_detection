# FastAPI app entry point and route definitions.
# Exposes endpoints: GET /models, GET /models/{name}, GET /pareto
from fastapi import FastAPI
from mlflow_client import get_latest_per_model, get_all_runs, get_runs_by_model
from pareto import generate_pareto

# Initialize the application instance
app = FastAPI()

# Define a GET model endpoint — returns latest run per model
@app.get("/models")
def get_models():
	models = get_latest_per_model()
	return {"models": models}

# Define a GET model endpoint
@app.get("/models/runs")
def get_runs():
	runs = get_all_runs()
	return {"runs": runs}



@app.get("/models/runs/{model_name}")
def get_model_runs(model_name: str):

	runs = get_runs_by_model(model_name)

	return {"runs": runs}




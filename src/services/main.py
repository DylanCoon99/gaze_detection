# FastAPI app entry point and route definitions.
# Exposes endpoints: GET /models, GET /models/{name}, GET /pareto
from fastapi import FastAPI
from mlflow_client import get_latest_per_model, get_all_runs, get_runs_by_model
from pareto import pareto_frontier

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


@app.get("/pareto")
def get_pareto(
	metric_a: str = "accuracy.mae_combined",
	metric_b: str = "safety.fnr",
):
	"""Return Pareto-optimal models across two metrics (lower is better)."""
	models = get_latest_per_model()
	frontier = pareto_frontier(models, metric_a, metric_b)
	return {
		"metric_a": metric_a,
		"metric_b": metric_b,
		"frontier": frontier,
	}

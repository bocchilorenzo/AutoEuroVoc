from os import path, environ
from transformers import pipeline
from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from dotenv import load_dotenv

load_dotenv()

# Retrieve the path to the model from the environment variables
model_path = environ.get("MODEL_PATH", "./model")
device = environ.get("DEVICE", "cpu")
pred_type = environ.get("PRED_TYPE", "id")
language = environ.get("LANGUAGE", "it")
threshold = environ.get("THRESHOLD", None) # If None, the pipeline will use top_k
top_k = environ.get("TOP_K", None) # If None, all the labels will be returned. Only considered if threshold is None

if top_k is not None:
    top_k = int(top_k)
if threshold is not None:
    threshold = float(threshold)
    top_k = None

# Load the label mappings if the prediction type is label
if pred_type == "label":
    with open(f"./config/label_mappings/{language}.json", "r", encoding="utf-8") as f:
        labels = json.load(f)

# Load the model with its tokenizer and config
# To know more about the available parameters, read the documentation for more info: https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline
print("Loading model...")
classifier = pipeline(
    "text-classification",
    model=model_path,
    tokenizer=model_path,
    config=path.join(model_path, "config.json"),
    device=device,
    top_k=top_k,
)
tokenizer_kwargs = {"padding": "max_length", "truncation": True, "max_length": 512}

print("Starting API...")
# Allow CORS for all origins
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        # allow_credentials=True, # uncomment this line if you want to allow credentials, but you have to set allow_origins to a list of allowed origins
        allow_methods=['*'],
        allow_headers=['*'],
        expose_headers=['access-control-allow-origin'],
    )
]

app = FastAPI(middleware=middleware)

# Define the request body. It should contain only a text field
class TextRequest(BaseModel):
    text: str

# Dummy endpoint to check if the API is running
@app.get("/")
async def get_data():
    return {"status": "OK"}

# Endpoint to get the predictions for a text
@app.post("/api")
async def post_data(request: TextRequest):
    text = request.text
    predictions = classifier(text, **tokenizer_kwargs)
    if not threshold:
        # If no threshold is specified, return all the predictions
        if pred_type == "id":
            return {"predictions": predictions[0]}
        elif pred_type == "label":
            # Map the label ids to the actual labels
            to_return = {"predictions": []}
            for pred in predictions[0]:
                try:
                    to_return["predictions"].append({"label": pred["label"], "score": pred["score"], "term": labels[pred["label"]]})
                except KeyError:
                    # If the label is not found in the label mappings, return an empty string
                    to_return["predictions"].append({"label": pred["label"], "score": pred["score"], "term": ""})
            return to_return
    else:
        # If a threshold is specified, return only the predictions with a score higher than the threshold
        if pred_type == "id":
            return {"predictions": [pred for pred in predictions[0] if pred["score"] >= threshold]}
        elif pred_type == "label":
            # Map the label ids to the actual labels
            to_return = {"predictions": []}
            for pred in predictions[0]:
                try:
                    if pred["score"] >= threshold:
                        to_return["predictions"].append({"label": pred["label"], "score": pred["score"], "term": labels[pred["label"]]})
                except KeyError:
                    # If the label is not found in the label mappings, return an empty string
                    to_return["predictions"].append({"label": pred["label"], "score": pred["score"], "term": ""})
            return to_return
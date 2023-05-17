from os import path, environ
from transformers import pipeline
from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Retrieve the path to the model from the environment variables
model_path = environ.get("MODEL_PATH", "./model")
device = environ.get("DEVICE", "cpu")

# Load the model with its tokenizer and config
# To know more about the available parameters, read the documentation for more info: https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline
print("Loading model...")
classifier = pipeline(
    "text-classification",
    model=model_path,
    tokenizer=model_path,
    config=path.join(model_path, "config.json"),
    top_k=None, #all the labels will be returned, use a fixed number to return only the top k labels
)

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
    predictions = classifier(text)
    return {"predictions": predictions[0]}
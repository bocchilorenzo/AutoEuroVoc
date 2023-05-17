from os import path
from transformers import pipeline
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the label of a text. Loading a model takes some time, so only use this script as a test or a reference for your own code.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    args = parser.parse_args()

    print("Loading model...")
    classifier = pipeline("text-classification", model=args.model_dir, tokenizer=args.model_dir, config=path.join(args.model_dir, "config.json"), top_k=None)

    print("Predicting...")
    preds = classifier("Regolamento (UE) 2023/956 del Parlamento europeo e del Consiglio del 10 maggio 2023 che istituisce un meccanismo di adeguamento del carbonio alle frontiere (Testo rilevante ai fini del SEE)")

    print("Predictions:")
    for pred in preds[0]:
        if pred["score"] > 0.5:
            print(pred["label"], pred["score"])
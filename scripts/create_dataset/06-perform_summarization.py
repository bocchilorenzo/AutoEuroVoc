import argparse
import os
import gzip
import json
import re
import math
from pagerange import PageRange
from datetime import datetime
from copy import deepcopy

def get_years(data_path):
    args.summarized = False
    if args.years == "all":
        args.years = [year for year in os.listdir(data_path)
                      if os.path.isfile(os.path.join(data_path, year))
                      and year.endswith(".json.gz")]
    else:
        args.years = PageRange(args.years).pages
        files_in_directory = [file for file in os.listdir(data_path)
                              if file.endswith(".json.gz")]

        files_per_year = {}
        for s in files_in_directory:
            m = re.match(r"^([0-9]{4})[^0-9]", s)
            if m:
                y = m.group(1)
                if y not in files_per_year:
                    files_per_year[y] = []
                files_per_year[y].append(s)

        files_to_consider = []
        for year in args.years:
            if str(year) in files_per_year:
                files_to_consider += files_per_year[str(year)]
        args.years = files_to_consider
    
    # Test if the file is summarized or not
    if len(args.years) > 0:
        with gzip.open(os.path.join(data_path, args.years[0]), "rt", encoding="utf-8") as file:
            data = json.load(file)
            if "importance" in data[tuple(data.keys())[0]]:
                args.summarized = True
            del data

    if args.summarized:
        print(f"### Files are summarized ###")
    print(f"Files to process: {', '.join(args.years)}\n")

def summarize(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    get_years(args.data_path)
    if not args.summarized:
        print("No summary data found, nothing to do here")
        exit()

    for yearFile in args.years:
        docsToSave = {}

        outputFile = os.path.join(args.output_path, yearFile)
        if os.path.exists(outputFile):
            print(f"File {outputFile} exists, skipping")
            continue

        with gzip.open(os.path.join(args.data_path, yearFile), "rt", encoding="utf-8") as file:
            data = json.load(file)

            j = 0
            for doc in data:
                print(f"{datetime.now().replace(microsecond=0)} - {j}/{len(data)}", end="\r")
                j += 1
                # if j > 10:
                #     break

                text = ""
                full_text = data[doc]["full_text"]
                phrase_importance = []
                i = 0

                for imp in data[doc]["importance"]:
                    if math.isnan(imp):
                        imp = 0
                    phrase_importance.append((i, imp))
                    i += 1
                
                phrase_importance = sorted(phrase_importance, key=lambda x: x[1], reverse=True)

                # Second option
                new_text = []
                phrase_index = 0
                while len(" ".join([full_text[phrase[0]] for phrase in new_text]).split()) < args.max_length and phrase_index < len(phrase_importance):
                    new_text.append(phrase_importance[phrase_index])
                    phrase_index += 1

                # Then, we sort the phrases by their position in the document.
                if len(new_text) > 0:
                    new_text = sorted(new_text, key=lambda x: x[0])
                    text += " ".join([full_text[phrase[0]] for phrase in new_text])
                else:
                    text += " ".join([full_text[phrase[0]] for phrase in phrase_importance])

                text = re.sub(r'\r', '', text)
                data[doc]["summarized_text"] = text

                docsToSave[doc] = data[doc]

        print(f"Saving file {outputFile}")
        with gzip.open(outputFile, "wt", encoding="utf-8") as fw:
            json.dump(docsToSave, fw, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFIDF-based summarizer for the dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, required=True, help="Directory containing the input dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Directory where the output dataset should be saved")
    parser.add_argument("--years", type=str, default="all", help="Range of years to summarize (e.g. 2010-2022 includes 2022). Use 'all' to process all the files in the given folder.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum number of words of the text to be processed.")
    args = parser.parse_args()

    summarize(args)
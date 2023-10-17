import json
import matplotlib.pyplot as plt
from os import path, makedirs
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from math import ceil


def create_graphs(args):
    def round_up_to_nearest_1000(num):
        return ceil(num / 1000) * 1000

    if not path.isdir(args.stats_path):
        raise Exception(f"{args.stats_path} is not a directory.")

    makedirs(args.output_path, exist_ok=True)

    print("Reading data...")

    with open(path.join(args.stats_path, "byLang.json"), "r") as f:
        stats = json.load(f)

        deprecated = pd.read_csv("../../config/deprecated.csv", sep=",")
        deprecated_labels = set()
        for row in deprecated.itertuples():
            deprecated_labels.add(row[1].split("/")[-1])

        docs_with_tags_general = []
        total_docs_general = []
        samples_per_label_general = []
        labels_per_doc_general = []
        labels_per_doc_2010_general = []
        deprecated_per_year_general = []
        labels_per_doc_per_year_general = []

        for lang in tqdm(stats):
            # total documents
            docs_with_tags = stats[lang]["docsWithEuroVocByYear"]
            docs_with_tags = dict(sorted(docs_with_tags.items()))
            docs_without_tags = stats[lang]["docsWithoutEuroVocByYear"]
            docs_without_tags = dict(sorted(docs_without_tags.items()))
            total_docs = {}
            for year in zip(docs_with_tags, docs_without_tags):
                if year[0] == year[1]:
                    total_docs[year[0]] = (
                        docs_with_tags[year[0]] + docs_without_tags[year[1]]
                    )
            docs_with_tags_general.append(docs_with_tags)
            total_docs_general.append(total_docs)

            # labels per document per year
            labels_per_doc_per_year = stats[lang]["labelNumberPerYear"]
            labels_per_doc_per_year = {int(k): np.mean(v) if len(v) > 0 else np.nan for k, v in labels_per_doc_per_year.items()}
            labels_per_doc_per_year = dict(sorted(labels_per_doc_per_year.items()))
            labels_per_doc_per_year_general.append(labels_per_doc_per_year)

            # sample per label
            labels_initial = stats[lang]["frequencies"]
            samples_per_label_general.append(labels_initial)

            # labels per document
            labels_per_doc = stats[lang]["labelsPerDoc"]
            labels_per_doc = {int(k): v for k, v in labels_per_doc.items()}
            labels_per_doc = dict(sorted(labels_per_doc.items()))
            labels_per_doc_max15 = {15: 0}

            for k, v in labels_per_doc.items():
                if int(k) <= 15:
                    labels_per_doc_max15[k] = v
                else:
                    labels_per_doc_max15[15] += v

            labels_per_doc_general.append(labels_per_doc_max15)

            labels_per_doc_2010 = stats[lang]["labelsPerDoc2010"]
            labels_per_doc_2010 = {int(k): v for k, v in labels_per_doc_2010.items()}
            labels_per_doc_2010 = dict(sorted(labels_per_doc_2010.items()))
            labels_per_doc_max15_2010 = {15: 0}

            for k, v in labels_per_doc_2010.items():
                if int(k) <= 15:
                    labels_per_doc_max15_2010[k] = v
                else:
                    labels_per_doc_max15_2010[15] += v

            labels_per_doc_2010_general.append(labels_per_doc_max15_2010)

            # deprecated labels
            labels_per_year = stats[lang]["labelsPerYear"]
            labels_per_year = dict(sorted(labels_per_year.items()))

            deprecated_per_year = {}
            for year in labels_per_year:
                deprecated_per_year[year] = 0
                for label in labels_per_year[year]:
                    if label in deprecated_labels:
                        deprecated_per_year[year] += 1

            deprecated_per_year_general.append(deprecated_per_year)

        ###########PLOTTING############
        print("Plotting graphs...")
        
        # Total documents
        docs_with_tags_final = {}
        total_docs_final = {}
        for lang in docs_with_tags_general:
            for year in lang:
                if year not in docs_with_tags_final:
                    docs_with_tags_final[year] = [lang[year]]
                else:
                    docs_with_tags_final[year].append(lang[year])
        for year in docs_with_tags_final:
            docs_with_tags_final[year] = np.mean(docs_with_tags_final[year])
        for lang in total_docs_general:
            for year in lang:
                if year not in total_docs_final:
                    total_docs_final[year] = [lang[year]]
                else:
                    total_docs_final[year].append(lang[year])
        for year in total_docs_final:
            total_docs_final[year] = np.mean(total_docs_final[year])

        docs_with_tags_final = dict(sorted(docs_with_tags_final.items()))
        total_docs_final = dict(sorted(total_docs_final.items()))

        plt.figure(figsize=(18, 10))
        plt.bar(
            total_docs_final.keys(), total_docs_final.values(), label="Total documents"
        )
        plt.bar(
            docs_with_tags_final.keys(),
            docs_with_tags_final.values(),
            label="Documents with eurovoc classifiers",
        )
        plt.title("Number of documents per year", fontsize=24)
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        """ for i, v in enumerate(stats.values()):
            plt.text(i - 0.35, v + 150, str(v), fontsize=12, rotation=90) """
        plt.savefig(path.join(args.output_path, "num_docs.png"), bbox_inches="tight")

        # Labels per document per year
        labels_per_doc_per_year_final = {}
        for lang in labels_per_doc_per_year_general:
            for year in lang:
                if year not in labels_per_doc_per_year_final:
                    labels_per_doc_per_year_final[year] = [lang[year]]
                else:
                    labels_per_doc_per_year_final[year].append(lang[year])
        for year in labels_per_doc_per_year_final:
            labels_per_doc_per_year_final[year] = np.nanmean(
                np.array(labels_per_doc_per_year_final[year], dtype=np.float32)
            )
        labels_per_doc_per_year_final = dict(
            sorted(labels_per_doc_per_year_final.items())
        )
        for key in list(labels_per_doc_per_year_final.keys()):
            if np.isnan(labels_per_doc_per_year_final[key]):
                del labels_per_doc_per_year_final[key]
        
        labels_per_doc_per_year_final = {str(k): v for k, v in labels_per_doc_per_year_final.items()}
        del labels_per_doc_per_year_final["1001"]
        
        plt.figure(figsize=(18, 10))
        plt.plot(labels_per_doc_per_year_final.keys(), labels_per_doc_per_year_final.values(), label="_avg", color="tab:green")
        plt.ylim(bottom=0)
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        plt.title("Average number of labels per document per year", fontsize=24)
        plt.savefig(path.join(args.output_path, "labs_per_doc_per_year_0.png"), bbox_inches="tight")

        # Sample per label
        samples_per_label_final = {}
        for lang in samples_per_label_general:
            for label_num in lang:
                if label_num not in samples_per_label_final:
                    samples_per_label_final[label_num] = [lang[label_num]]
                else:
                    samples_per_label_final[label_num].append(lang[label_num])
        for label in samples_per_label_final:
            samples_per_label_final[label] = np.mean(samples_per_label_final[label])
        samples_per_label_final = dict(
            sorted(
                samples_per_label_final.items(), key=lambda item: item[1], reverse=True
            )
        )
        maximum = 0
        for label in samples_per_label_final:
            maximum = samples_per_label_final[label]
            break
        maximum = round_up_to_nearest_1000(maximum)
        step = 1000 if maximum <= 15000 else 2000
        plt.figure(figsize=(15, 8))
        plt.bar(
            samples_per_label_final.keys(),
            samples_per_label_final.values(),
            label="Samples",
            snap=False,
        )
        plt.title("Number of samples per label", fontsize=24)
        plt.xticks([])
        plt.yticks(fontsize=12, ticks=range(0, maximum, step))
        plt.ylim(top=maximum)
        plt.savefig(
            path.join(args.output_path, "labels_samples.png"),
            bbox_inches="tight",
        )

        # Labels per document
        labels_per_doc_final = {}
        for lang in labels_per_doc_general:
            for label_num in lang:
                if label_num not in labels_per_doc_final:
                    labels_per_doc_final[label_num] = [lang[label_num]]
                else:
                    labels_per_doc_final[label_num].append(lang[label_num])
        for label_num in labels_per_doc_final:
            labels_per_doc_final[label_num] = np.mean(labels_per_doc_final[label_num])
        labels_per_doc_final = dict(sorted(labels_per_doc_final.items()))
        plt.figure(figsize=(10, 7))
        plt.plot(
            labels_per_doc_final.keys(),
            labels_per_doc_final.values(),
            label="_noshow",
            linewidth=2,
            marker="o",
            markersize=8,
        )
        plt.tick_params(axis="x", rotation=90, labelsize=12)
        plt.tick_params(axis="y", labelsize=12)
        plt.title("Number of labels per document", fontsize=24)
        plt.xticks(
            rotation=0,
            fontsize=12,
            ticks=range(0, 16, 1),
            labels=[n if n != 15 else "15+" for n in range(0, 16, 1)],
        )
        plt.yticks(fontsize=12)
        plt.xlabel("Labels", fontsize=16)
        plt.ylabel("Documents", fontsize=16)
        plt.savefig(
            path.join(args.output_path, "labels_per_doc.png"),
            bbox_inches="tight",
        )

        # Labels per document 2010
        labels_per_doc_2010_final = {}
        for lang in labels_per_doc_2010_general:
            for label_num in lang:
                if label_num not in labels_per_doc_2010_final:
                    labels_per_doc_2010_final[label_num] = [lang[label_num]]
                else:
                    labels_per_doc_2010_final[label_num].append(lang[label_num])
        for label_num in labels_per_doc_2010_final:
            labels_per_doc_2010_final[label_num] = np.mean(
                labels_per_doc_2010_final[label_num]
            )
        labels_per_doc_2010_final = dict(
            sorted(
                labels_per_doc_2010_final.items(),
            )
        )
        plt.figure(figsize=(10, 7))
        plt.plot(
            labels_per_doc_2010_final.keys(),
            labels_per_doc_2010_final.values(),
            label="_noshow",
            linewidth=2,
            marker="o",
            markersize=8,
        )
        plt.tick_params(axis="x", rotation=90, labelsize=12)
        plt.tick_params(axis="y", labelsize=12)
        plt.suptitle("Number of labels per document", fontsize=24)
        plt.title("Year range 2010-2022", fontsize=16)
        plt.xticks(
            rotation=0,
            fontsize=12,
            ticks=range(0, 16, 1),
            labels=[n if n != 15 else "15+" for n in range(0, 16, 1)],
        )
        plt.yticks(fontsize=12)
        plt.xlabel("Labels", fontsize=16)
        plt.ylabel("Documents", fontsize=16)
        plt.savefig(
            path.join(args.output_path, "labels_per_doc_2010.png"),
            bbox_inches="tight",
        )

        # Deprecated labels
        deprecated_per_year_final = {}
        for lang in deprecated_per_year_general:
            for year in lang:
                if year not in deprecated_per_year_final:
                    deprecated_per_year_final[year] = [lang[year]]
                else:
                    deprecated_per_year_final[year].append(lang[year])
        for year in deprecated_per_year_final:
            deprecated_per_year_final[year] = np.mean(deprecated_per_year_final[year])
        deprecated_per_year_final = dict(sorted(deprecated_per_year_final.items()))
        maximum = 0
        for year in deprecated_per_year_final:
            if deprecated_per_year_final[year] > maximum:
                maximum = deprecated_per_year_final[year]
        maximum = ceil(maximum / 100) * 100

        plt.figure(figsize=(18, 10))
        plt.plot(
            deprecated_per_year_final.keys(),
            deprecated_per_year_final.values(),
            label="_noshow",
            linewidth=2,
        )
        plt.tick_params(axis="x", rotation=90, labelsize=12)
        plt.tick_params(axis="y", labelsize=12)
        plt.suptitle("Deprecated labels usage", fontsize=24, y=0.95, x=0.513)
        plt.title("Total number of occurrences per year", fontsize=18, x=0.5)
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        plt.vlines("2010", 0, maximum, colors="r", linestyles="dashed", label="_2010")
        plt.savefig(path.join(args.output_path, "deprecated.png"), bbox_inches="tight")


if __name__ == "__main__":
    # fmt:off
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create graphs starting from the stats for each language.",
    )
    parser.add_argument("--stats_path", type=str, default=None, required=True, help="Path to the stat files.")
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to the output directory for the images.")

    args = parser.parse_args()

    create_graphs(args)

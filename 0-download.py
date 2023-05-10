import urllib.request as req
from os import makedirs, path, remove
import zipfile
import argparse

def show_progress(block_num, block_size, total_size):
    """
    Show the progress of the download.

    :param block_num: The number of blocks transferred so far
    :param block_size: The size of each block
    :param total_size: The total size
    """
    print(round(block_num * block_size / total_size *100,2), end="\r")

def download_data():
    """
    Download the data for the given languages.
    """
    # Create the directory for the data
    makedirs(args.data_path, exist_ok=True)

    if args.langs == "all":
        args.langs = "en,it"
    
    # Download the data for all the languages
    for lang in args.langs.split(","):
        file_name = ""
        # Download the full text data
        if not args.summarized:
            file_name = "full_text"
        elif args.summarized and args.bigrams:
            file_name = "summarized_bigram"
        else:
            file_name = "summarized"
        
        print(f"Dowloading {file_name} data for {lang}...")
        url = f"link_to_the_data/{lang}/{file_name}.zip"
        req.urlretrieve(url, path.join(args.data_path, lang), show_progress)

        # Extract the data
        with zipfile.ZipFile(path.join(args.data_path, lang, f"{file_name}.zip"), 'r') as zip_ref:
            zip_ref.extractall(path.join(args.data_path, lang))
        
        # Remove the zip file
        remove(path.join(args.data_path, lang, f"{file_name}.zip"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="data/", help="Path to download the data in.")
    parser.add_argument("--langs", type=str, default="it", help="Languages to be downloaded, separated by a comme (e.g. en,it). Write 'all' to download all the languages.")
    parser.add_argument("--summarized", action="store_true", default=False, help="Download the summarized data instead of the full text one.")
    parser.add_argument("--bigrams", action="store_true", default=False, help="Download datasets summarized with bigrams instead of single words. Only used if --summarized is also used.")
    args = parser.parse_args()

    download_data()
import urllib.request
import zipfile
from os import path, remove, rename
import subprocess
import sys

print("Downloading summarizer...")
url = 'https://github.com/bocchilorenzo/text-summarizer/archive/refs/heads/main.zip'
urllib.request.urlretrieve(url, path.join("./", "text_summarizer.zip"))
with zipfile.ZipFile(path.join("./", "text_summarizer.zip"), 'r') as zip_ref:
    zip_ref.extractall("./")
rename(path.join("./", "text-summarizer-main"), path.join("./", "text_summarizer"))
remove(path.join("./", "text_summarizer.zip"))

with open(path.join("./", "text_summarizer", "__init__.py"), "wb") as f:
    f.write(b"from .summarizer import *")

print("Downloading udpipe models...")
url = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/udpipe-ud-2.5-191206.zip?sequence=1&isAllowed=y"
urllib.request.urlretrieve(url, path.join("./", "udpipe.zip"))
with zipfile.ZipFile(path.join("./", "udpipe.zip"), 'r') as zip_ref:
    zip_ref.extractall(path.join("./", "text_summarizer"))
rename(path.join("./", "text_summarizer", "udpipe-ud-2.5-191206"), path.join("./", "text_summarizer", "models"))

remove(path.join("./", "udpipe.zip"))

print("Installing requirements...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", path.join("./", "text_summarizer", "requirements.txt")])
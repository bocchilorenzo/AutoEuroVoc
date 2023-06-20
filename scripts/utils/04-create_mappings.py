import json
from os import makedirs, remove, path, listdir
from zipfile import ZipFile
from tqdm import tqdm
from urllib.request import urlopen
from shutil import copyfileobj, rmtree
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings('ignore')

def create_mappings():
    path_tc = path.realpath(path.join(__file__, "../../../config/label_mappings_tc"))
    path_do = path.realpath(path.join(__file__, "../../../config/label_mappings_do"))
    path_mt = path.realpath(path.join(__file__, "../../../config/label_mappings_mt"))

    makedirs(path_tc, exist_ok=True)
    makedirs(path_do, exist_ok=True)
    makedirs(path_mt, exist_ok=True)
    
    # Download XML archive
    with urlopen("https://op.europa.eu/o/opportal-service/euvoc-download-handler?cellarURI=http%3A%2F%2Fpublications.europa.eu%2Fresource%2Fcellar%2Ffb5bc6f8-a143-11ed-b508-01aa75ed71a1.0001.07%2FDOC_1&fileName=eurovoc_xml.zip") as in_stream, open('eurovoc_xml.zip', 'wb') as out_file:
        copyfileobj(in_stream, out_file)

    # Extract XML archive
    archive = ZipFile("./eurovoc_xml.zip")

    makedirs("./eurovoc_xml", exist_ok=True)
    archive.extractall("./eurovoc_xml")
    archive.close()

    for filename in tqdm(listdir("./eurovoc_xml")):
        to_export = {}
        if "desc_" in filename or "dom_" in filename or "thes_" in filename:
            tree = ET.parse(path.join("./eurovoc_xml/", filename))
            root = tree.getroot()
            to_add = []
            id_label, save_path = ("DOMAINE_ID", path_do) if "dom_" in filename else ("DESCRIPTEUR_ID", path_tc) if "desc_" in filename else ("THESAURUS_ID", path_mt)
            for child in root.iter():
                if child.tag == id_label:
                    to_add.append(child.text)
                elif child.tag == "LIBELLE":
                    to_add.append(child.text)
                    to_export[to_add[0]] = to_add[1]
                    to_add = []
            with open(path.join(save_path, f"{filename.split('_')[-1].split('.')[0]}.json"), "w", encoding="utf-8") as outfile:
                json.dump(to_export, outfile, ensure_ascii=False)
    
    remove("./eurovoc_xml.zip")
    rmtree("./eurovoc_xml")

if __name__ == "__main__":
    create_mappings()
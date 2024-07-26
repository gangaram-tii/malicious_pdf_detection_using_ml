#from pdfid import PDFiD, PDFiD2JSON
import sys

#WA for PDFID module path
#sys.path.insert(0, "/workspace/malware_detection/.venv/lib/python3.11/site-packages/pdfid/")

from pdfid import PDFiD, PDFID2Dict
import pandas as pd
import os
import re
import numpy as np

class PDFFeatures:
    def __init__(self, file_path):
        print(file_path)
        pdfinfo = PDFiD(file_path)
        self.features = [];
        PDFID2Dict(pdfinfo, False, False, self.features)
        if len(self.features):
            del self.features[0]["version"]

    @property
    def isFormatPDF(self):
        return len(self.features) > 0

    def new_feature(self, name, value):
        self.features[0][name] = value

    @property
    def data_frame(self):
        return pd.DataFrame.from_dict(self.features)

def convert_header_to_int(header):
    match = re.search(r'%PDF-(\d+)\.(\d+)', header)
    if match:
        major, minor = match.groups()
        return int(major) * 10 + int(minor)
    else:
        return 0  # Use 0 for invalid strings


def process_pdf_dataset(dir_path, isMalicious = False):
    df = pd.DataFrame()
    for root, dirs, files in os.walk(dir_path):
        #i = 0
        for file in files:
            #i = i + 1
            #if i == 10:
            #    return df
            file_path = os.path.join(root, file)
            pdff = PDFFeatures(file_path)
            if pdff.isFormatPDF:
                pdff.new_feature("isMalicious", 1 if isMalicious else 0)
                df = pd.concat([df, pdff.data_frame], ignore_index=True)
    return df
            

#List of pair [("dataset_path", "Malicious" or "Clean")] 

def PDFFeaturesCSV(datasets, outputfile):
    dfs = []
    valid_labels = ["Clean", "Malicious"]
    for d in datasets:
        label = d[1]
        if label in valid_labels:
            if os.path.exists(d[0]):
                dfs.append(process_pdf_dataset(d[0], True if label == "Malicious" else False))
            else:
                raise ValueError("Invalid Dataset: {d[0]} path!")
        else:
            raise ValueError("Invalid lable for dataset: {d[0]}, accepted labels are 'Clean' 'Malicious'")

    df = pd.concat(dfs, ignore_index=True)
    #df = df.fillna(0)

    # Convert columns to integers if possible
    for col in df.columns:
        # Check if the column can be converted to integers
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].fillna(0).astype(int)

    # Apply the conversion function to the 'header' column
    df['header'] = df['header'].apply(convert_header_to_int)
    df.to_csv(outputfile, index_label='index')

#features = process_pdf_dataset("./test")
#features.to_csv("./features.csv", index = False)

PDFFeaturesCSV(
    [
        ("./clean", "Clean"),
        ("./malicious", "Malicious")
    ],
    "./features.csv")


x = pd.read_csv("./features.csv")
print(x.head(5))

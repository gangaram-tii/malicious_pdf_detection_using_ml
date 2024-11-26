#from pdfid import PDFiD, PDFiD2JSON
import sys

#WA for PDFID module path
#sys.path.insert(0, "/workspace/malware_detection/.venv/lib/python3.11/site-packages/pdfid/")

from .thirdparty.pdfid.pdfid import PDFiD, PDFID2Dict
import pandas as pd
import os
import re
import numpy as np

def convert_header_to_int(header):
    match = re.search(r'%PDF-(\d+)\.(\d+)', header)
    if match:
        major, minor = match.groups()
        return int(major) * 10 + int(minor)
    else:
        return 0  # Use 0 for invalid strings


class PDFFeatures:
    def __init__(self, file_path):
        print(file_path)
        pdfinfo = PDFiD(file_path)
        self.features = [];
        PDFID2Dict(pdfinfo, False, False, self.features)
        if len(self.features):
            del self.features[0]["version"]

    def as_encoded_data_frame(self):
        df = self.data_frame
        # Convert columns to integers if possible
        for col in df.columns:
            # Check if the column can be converted to integers
            if df[col].dtype in [np.float64, np.float32]:
                df[col] = df[col].fillna(0).astype(int)

        # Apply the conversion function to the 'header' column
        df['header'] = df['header'].apply(convert_header_to_int)
        return df

    @property
    def isFormatPDF(self):
        return len(self.features) > 0

    def new_feature(self, name, value):
        self.features[0][name] = value

    @property
    def data_frame(self):
        return pd.DataFrame.from_dict(self.features)

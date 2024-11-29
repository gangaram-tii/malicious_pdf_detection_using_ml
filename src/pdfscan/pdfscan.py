import pandas as pd
import numpy as np
import xgboost as xgb
from .features import PDFFeatures

import socket
import os


__all__ = ["PDFClassifier", "main"]
__version__ = "0.1.0"


TRACE_ENABLED = False

def trace(message):
    if TRACE_ENABLED:
        print(message)


# Define host and port for the server
HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port to listen on


# Define a function to merge two columns using bitwise operations
def merge_values(val1, val2):
    return (val1 << 10) | val2

def merge_columns(df, column1, column2, newcolumn, drop=False):
    # Apply the function to create a new column 'merged'
    if column1 not in df:
        trace(f"Feature {column1} missing")
        df[column1] = 0
    if column2 not in df:
        trace(f"Feature {column2} missing")
        df[column2] = 0

    df[newcolumn] = df.apply(lambda row: merge_values(row[column1], row[column2]), axis=1)
    if drop:
        df = df.drop(labels=[column1, column2], axis=1)
    return df

class PDFClassifier:
    def __init__(self):
        # To load the model from the file later:
        self.booster = xgb.Booster()
        model = os.path.join(os.path.dirname(__file__), "xgboost-classifier.json")
        self.booster.load_model(model)

    def classify(self, filename):
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            features = PDFFeatures(filename)
            df = features.as_encoded_data_frame()
            final_features = self.clean_and_transform(df)
            pred_label_prob = self.booster.predict(final_features)
            if (pred_label_prob > 0.5):
                return "NOK"
            else:
                return "OK"
        else:
            return "INV"

    def start_server(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((HOST, port))  # Bind to address
            server_socket.listen()

            print(f"Server listening on {HOST}:{port}...")

            while True:
                # Wait for a connection from a client
                conn, addr = server_socket.accept()
                with conn:
                    trace(f"Connected by {addr}")

                    # Receive the file name from the client
                    filename = conn.recv(1024).decode('utf-8')
                    if not filename:
                        continue

                    trace(f"Received file name: {filename}")

                    # Process the file and get the result
                    label = self.classify(filename)

                    # Send the result back to the client
                    response = f"{label}"
                    conn.sendall(response.encode('utf-8'))
    
    def clean_and_transform(self, dataframe):
        dataframe = dataframe.drop(labels=['filename'], axis=1)
        dataframe = merge_columns(dataframe, "/OpenAction", "/AcroForm", "ActionForm", True)
        dataframe = merge_columns(dataframe, "/JBIG2Decode", "/RichMedia", "Decode-Media", True)
        dataframe = merge_columns(dataframe, "/Launch", "/EmbeddedFile", "Launch-EmbeddedFile", True)
        dataframe = merge_columns(dataframe, "/XFA", "/Colors > 2^24", "XFA-Colors", True)
        dataframe = merge_columns(dataframe, "/JS_hexcode_count", "/JavaScript_hexcode_count", "hexcode1", True)
        dataframe = merge_columns(dataframe, "/Page_hexcode_count", "/OpenAction_hexcode_count", "hexcode2", True)
        dataframe = merge_columns(dataframe, "/AA_hexcode_count", "/AcroForm_hexcode_count", "hexcode3", True)
        dataframe = merge_columns(dataframe, "/XFA_hexcode_count", "/JBIG2Decode_hexcode_count", "hexcode4", True)
        dataframe = merge_columns(dataframe, "/EmbeddedFile_hexcode_count", "/RichMedia_hexcode_count", "hexcode5", True)

        selected_features = ['header', 'obj', 'endobj', 'stream', 'endstream',
                            'xref', 'trailer', 'startxref', '/Page', '/Encrypt',
                            '/ObjStm', '/JS', '/JavaScript', '/AA', 'ActionForm',
                            'Decode-Media', 'Launch-EmbeddedFile', 'XFA-Colors']
        for f in selected_features:
            if f not in dataframe:
                trace(f"Feature {f} missing")
                dataframe[f] = 0
        features = dataframe[selected_features]
        # Create DMatrix, the data structure that XGBoost uses
        test_dmatrix = xgb.DMatrix(features)
        return test_dmatrix

def main():
    classifier = PDFClassifier()
    classifier.start_server(PORT)
    #print(classifier.classify("/home/gangaram/Downloads/shilpa_eid.pdf"))

# Main
if __name__ == '__main__':
    main()

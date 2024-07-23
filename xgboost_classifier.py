import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


# Define a function to merge two columns using bitwise operations
def merge_values(val1, val2):
    return (val1 << 10) | val2

def merge_columns(df, column1, column2, newcolumn, drop=False):
    # Apply the function to create a new column 'merged'
    df[newcolumn] = df.apply(lambda row: merge_values(row[column1], row[column2]), axis=1)
    if drop:
        df = df.drop(labels=[column1, column2], axis=1)
    return df

def preprocess_data(csvfile):
    dataframe = pd.read_csv(csvfile)
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
    labels = dataframe['isMalicious']
    features = dataframe[selected_features]
    # Splitting the data into a training and test set with test_size 0.25 and random_state 2019
    f_train, f_test, training_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=2019)

    # Create DMatrix, the data structure that XGBoost uses
    training_dmatrix = xgb.DMatrix(f_train, label=training_labels)
    test_dmatrix = xgb.DMatrix(f_test)
    return training_dmatrix, test_dmatrix, test_labels

def train_model(training_dmatrix, params, num_rounds, filename):
    # Set up the parameters for the XGBoost model
    # Train the model
    model = xgb.train(params, training_dmatrix, num_rounds)

    model.save_model(filename)
    print(f'Model saved to {filename}')

def evaluate_model(filename, test_dmatrix, test_labels):
    # To load the model from the file later:
    booster = xgb.Booster()
    booster.load_model(filename)

    # Verify the loaded model by making predictions again
    pred_label_prob = booster.predict(test_dmatrix)
    pred_label = (pred_label_prob > 0.5).astype(int)

    # Evaluate the model
    accuracy = accuracy_score(test_labels, pred_label)
    conf_matrix = confusion_matrix(test_labels, pred_label)
    TN, FP, FN, TP = conf_matrix.ravel()
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    print(f'Accuracy: {accuracy:.4f}\n')
    print('Confusion Matrix:')
    print(conf_matrix)

    print('\nKey Metrics:')
    print(f"[Precision, Recall, Specificity] \n[{precision:.4f},    {recall:.4f}, {specificity:.4f}]")

# Main 
if __name__ == '__main__':
    params = {
        'booster': 'gbtree',             # Use the tree-based model
        'objective': 'binary:logistic',  # Binary classification
        'eta': 0.1,                      # Learning rate
        'max_depth': 8,                  # Maximum depth of the trees
        'subsample': 0.75,               # Subsample ratio of the training instances
        'colsample_bytree': 0.75,        # Subsample ratio of columns when constructing each tree
        'gamma': 0.1,                    # Minimum loss reduction required to make a further partition
        'lambda': 1.0,                   # L2 regularization term on weights
        'alpha': 0.0,                    # L1 regularization term on weights
        'eval_metric': 'logloss'         # Evaluation metric
    }
    model_path = "xgboost-classifier.json"
    training_dmatrix, test_dmatrix, test_labels = preprocess_data('./dataset/features.csv')
    train_model(training_dmatrix, params, 100, model_path)
    evaluate_model(model_path, test_dmatrix, test_labels)




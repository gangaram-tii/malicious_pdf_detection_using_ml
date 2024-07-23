import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

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
    training_data, test_data, training_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=2019)

    return training_data, test_data, training_labels, test_labels

 
 
 
def train_model(training_data, training_labels, hyperparams, filename):
    print("training started")
    knn = KNeighborsClassifier()
    # tune hyperparameters
    models = GridSearchCV(estimator=knn, param_grid=hyperparams, cv=10, scoring='accuracy', verbose=1)

    models.fit(training_data, training_labels)

    # best model
    best_model = models.best_estimator_
    best_score = models.best_score_

    # evaluate on test data
    training_accu = best_score*100   

    #print result
    print(f"Training Accuracy:   {training_accu:.2f}%")
    print("training finished")
    print("\n")  

    fp = open(filename, 'wb') 
    pickle.dump(best_model, fp)  
    fp.close()
                
      
def evaluate_model(filename, test_data, test_labels):
    # load the model from disk
    fp = open(filename, 'rb')
    model = pickle.load(fp)
    pred_labels = model.predict(test_data)
    # Evaluate the model
    accuracy = accuracy_score(test_labels, pred_labels)
    conf_matrix = confusion_matrix(test_labels, pred_labels)
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
    model_path = "knn-classifier.model"
    hyperparams = {
        'n_neighbors':[5,10,20,30,40,50],
        'metric':["euclidean","manhattan", "minkowski"],
        'weights':["uniform","distance"],
    }
    training_data, test_data, training_labels, test_labels = preprocess_data('./dataset/features.csv')
    train_model(training_data, training_labels, hyperparams, model_path)
    evaluate_model(model_path, test_data, test_labels)


import csv
import os
from config.config import (
    EXPERIMENT_RESULTS_PATH )
from core.experiments.ContextClasses import (DataContext, TrainingContext)
import datetime

def evaluate_model(model, data):
    #### Function to evaluate a model and return precision, recall and F1
    model.eval()
    pred = model(data).argmax(dim=1)
    correct_predictions_mask = pred[data.test_mask] == data.y[data.test_mask]
    incorrect_predictions_mask = pred[data.test_mask] != data.y[data.test_mask]
    TP = sum(pred[data.test_mask][correct_predictions_mask] == 1)
    FP = sum(pred[data.test_mask][incorrect_predictions_mask] == 1)
    FN = sum(pred[data.test_mask][incorrect_predictions_mask] == 0)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    return precision, recall, F1

def save_experiment_results_to_file(file_name, experiment_context:str,
                                    data_context:DataContext,
                                    training_context:TrainingContext,
                                    experiment, #TO DO set object type
                                    evaluation_results #TO DO pass class objects. for now passing dict object
                                    ):

    # Define the CSV field names for training context and evaluation results
    fieldnames = ["time_stamp","info","Epochs", "Learning Rate", "hidden_layer_dimension","num_layers", "Model", 
                  "accuracy", "precision", "recall","F1","model_directory","triples_path","entities_labels_path",
                  "properties_labels_path","graph_embeddings_path"]
    file_path = os.path.join(EXPERIMENT_RESULTS_PATH["eval_results"], file_name)
    # Create a dictionary with the data to be saved
    data = {

        "time_stamp":datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
        "info":experiment_context,
        #data context
        "triples_path" : data_context.triples_path,
        "entities_labels_path" : data_context.entities_labels_path,
        "properties_labels_path" : data_context.properties_labels_path,
        "graph_embeddings_path" : data_context.graph_embeddings_path,

        # training context
        "Epochs": training_context.num_epochs,
        "Learning Rate": training_context.learning_rate,
        "hidden_layer_dimension": training_context.dim_hidden_layer,
        "num_layers":training_context.num_layers,

        "Model": experiment.model,
        "model_directory":experiment.experiment_results_folder_path,

        # evaluation results
        "accuracy": evaluation_results["accuracy"],
        "precision": evaluation_results["precision"],
        "recall": evaluation_results["recall"],
        "F1":evaluation_results["F1"],

    
    }

    # Check if the file exists to decide whether to write headers or not
    file_exists = False
    try:
        with open(file_path, 'r') as file:
            file_exists = True
    except FileNotFoundError:
        pass

    # Open the file in append mode to append new data
    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the headers only if the file is newly created
        if not file_exists:
            writer.writeheader()

        # Write the data to the file
        writer.writerow(data)

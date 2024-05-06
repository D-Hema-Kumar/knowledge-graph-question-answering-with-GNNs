# psmg-imkg-gnn-qa
This is a repository developed for implementing the knowledge graph Question and Answering system (KGQA) using Graph Neural Networks (GNNs). The thesis is based on the below research questions:

* RQ1: How do the embeddings obtained by a pre-trained LMs (RoBERTa and GPT2) from entity descriptions in a KG help in classifying an entity as being the answer for a given question Q?
* RQ2: How do the embeddings obtained from TransE, ComplEx, and DistMult models impact the reasoning abilities of various GNN architecture types?
* RQ3: How do the combined embeddings (one of RoBERTa and GPT2 and one of TransE, ComplEx, and DistMult) embeddings affect the performance of the QA system?


![Image Alt Text]([image_url](https://github.com/D-Hema-Kumar/knowledge-graph-question-answering-with-GNNs/blob/main/images/training_workflow.png?raw=true))

The below figure shows how the KGQA system is structured on a high level:

![Alt text](images\QA_system.png "KGQA Pipeline")

## Thesis Summary

* KGs : Two knowledge graphs namely VAD and MetaQA KGs were used.
* GNN Models : Three GNN models were tested : GCN, R-GCN and R-GAT.
* Embeddings : The above GNN models were initialized with 11 types of embeddings obtained using:
    * KGE : TransE, DistMult and ComplEx embeddings
    * LM : RoBERTa and GPT2 embeddings from textual descriptions of nodes.
    * Composite : KGE X LM (e.g. DistMult+RoBERTa, DistMult+GPT2, .. etc.)

## Data
To train the KGQA system, the user needs data as below:

* KG in the form of triples in a csv file (data\source_data_old\VAD_triples.csv), entities and properties with their corresponding labels (data\source_data_old\VAD_entities_labels.csv)
* QA training and testing dataset as structured like in the file (data\source_data_old\qa_training_data.csv)
* Subgraph information : This file is generated using the core\Q_subgraphs\QuestionSubgraphs.py file. It generates the subgraph for a given question and stores the results in the user provided location. For example, 1-Hop MetaQA dev subgraphs are stored at:  data\MetaQA\source_data\one-hop\dev_subgraphs.npz

## Graph And Question Embeddings
The embeddings of above KG entities needs to be generated first. I used 5 methods ( language models LLMs : RoBERTa and GPT2 and KGE models : DistMult, TransE and ComplEx) 
* KGE : relevant files in core\KGE
* LLM : relevant files in core\LLM

Entity and relation embeddings for MetaQA are stored at : data\MetaQA\graph_embeddings

MetaQA Question embeddings are stored at : data\MetaQA\question_embeddings

## Reproducability

Once the above steps are completed, the KGQA pipeline can be trained using the .\main.py file where all the experiments are being called.
```
pipenv run python main.py
```

## Results

The corresponding data for the experiments for the QA tasks are stored :

* Trained Models at : core\experiments\qa\results
* Experiment results for VAD-QA at : core\experiments\qa\qa_experiments_masterdata.csv
* Experiment results for MetaQA at : core\experiments\qa\MetaQA_experiments_masterdata.csv

# Running the Code

The below details are for evaluation of the implemented KGQA system by the examination committee (Professor, etc.)

### Dependencies
Install python and pipenv and change the directory and create a virtual environment and install the required libraries specified in the Pipfile and Pipfile.lock. 
```
cd your-repository
pipenv install
```
### Running the experiment
1. Activate the environment
```
pipenv shell
```
2. Run the experiments
```
pipenv run python main.py
```
This runs the experiment relevant for research question RQ1 with the below settings:
* KG used : MetaQA
* QA Training Dataset : 1-Hop MetaQA Dev
* QA Testing Dataset : 1-Hop MetaQA Dev
* LM Embedding used : RoBERTa
* KGE Embedding used : DistMult
* GNN Model Trained : GCN

The code is set to run for 20 Minutes completing 2 epochs on 9992 questions in MetaQA.

Uncomment the necessary lines in the main.py file for running the experiments relevant to other research questions (RQ2 and RQ3). For evaluation of the thesis by the professor, I have added
'RUN BELOW FOR EVALUATION OF THESIS' comment in the main.py file. Please find the 'Uncomment below' comment and uncomment the relevant lines as per the need.

### Results
The above 1-Hope metaQA experimental results can be found at :

* Model data is stored at location by creating a new folder based on the timestamp : core\experiments\qa\results\<--exp_folder-->
* Metadata with evaluation metrics are appended to csv file : core\experiments\qa\MetaQA_experiments_masterdata.csv

## Additional Info
Because of space restrictions in the USB stick, I have only prepared the code to run on a subset of data i.e. on the dev dataset for 1-hop MetaQA. If you want to train the models on the other datasets, all you need to do is
* Generate the subgraphs for the respective dataset by using core\Q_subgraphs\QuestionSubgraphs.py
* Modify the main.py file's input parameters accordingly and run the file






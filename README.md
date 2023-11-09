# psmg-imkg-gnn-qa
This is a repository developed for implementing the knowledge graph Question and Answering system (KGQA) using Graph Neural Networks (GNNs). The thesis is based on the below research questions:

* RQ1: How do the embeddings obtained by a pre-trained LMs (RoBERTa and GPT2) from entity descriptions in a KG help in classifying an entity as being the answer for a given question Q?
* RQ2: How do the embeddings obtained from TransE, ComplEx, and DistMult models impact the reasoning abilities of various GNN architecture types?
* RQ3: How do the combined embeddings (one of RoBERTa and GPT2 and one of TransE, ComplEx, and DistMult) embeddings affect the performance of the QA system?

![Alt text](images\training_workflow.png "KGQA Pipeline")

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
* Subgraph information 

## Graph And Question Embeddings
The embeddings of above KG entities needs to be generated first. I used 5 methods ( language models LLMs : RoBERTa and GPT2 and KGE models : DistMult, TransE and ComplEx) 
* KGE : relevant files in core\KGE
* LLM : relevant files in core\LLM

Entity and relation embeddings for VAD-QA are stored at : data\graph_embeddings

VAD Question embeddings are stored at : data\question_embeddings

## Training

Once the above steps are completed, the KGQA pipeline can be trained using the .\main.py file where all the experiments are being called.

## Results

The corresponding data for the experiments for the QA tasks are stored :

* Trained Models at : core\experiments\qa\results
* Experiment results at : core\experiments\qa\qa_experiments_masterdata.csv

## Running the Code

1. Install the necessary libraries using pipenv piplock file
2. Ruuning the main.py file with command :
    pipenv run python main.py


runs the experiment relevant for research question RQ1 with the below settings:
* KG used : MetaQA
* QA Training Dataset : 1-Hop MetaQA Dev
* QA Testing Dataset : 1-Hop MetaQA Dev
* LM Embedding from : RoBERTa
* KGE Embedding : DistMult
* GNN Model : GCN

Uncomment the necessary lines in the main.py file for running the experiments relevant to other research questions (RQ2 and RQ3). For evaluation of the thesis by the professor, I have added
'RUN BELOW FOR EVALUATION OF THESIS' comment in the main.py file. Please find the 'Uncomment below' comment and uncomment the relevant lines as per the need.




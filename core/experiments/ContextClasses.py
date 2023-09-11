class DataContext(object):

    '''The data context class creates a data object that contains all the attributes that represent
    training data'''

    def __init__(self, triples_path:str,
                 entities_labels_path:str,
                 properties_labels_path:str,
                 graph_embeddings_path:str):
        
        self.triples_path = triples_path
        self.entities_labels_path = entities_labels_path
        self.properties_labels_path = properties_labels_path
        self.graph_embeddings_path = graph_embeddings_path


                 
class TrainingContext(object):
    '''The training context class creates an object that contains all the attributes that represent
    training_context of the experiment'''

    def __init__(
            self,  
            info:str, 
            num_epochs:int, 
            learning_rate:float,
            dim_hidden_layer:int,
            num_layers=3,
            train_ratio=0.8,
            num_bases=None):

        self.info = info
        self.train_ratio = train_ratio
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dim_hidden_layer = dim_hidden_layer
        self.num_layers=num_layers
        self.num_bases = num_bases

class QADataContext(DataContext):
    def __init__(
        self,
        triples_path,
        entities_labels_path,
        properties_labels_path,
        graph_embeddings_path,
        training_questions_concepts_answers_file_path,
        testing_questions_concepts_answers_file_path,
        training_questions_embeddings_path,
        testing_questions_embeddings_path,
        training_subgraphs_file_path= None,
        testing_subgraphs_file_path=None,
        is_vad_kb=True
    ):
        
        super().__init__( 
                         triples_path,
                         entities_labels_path,
                         properties_labels_path,
                         graph_embeddings_path
                )
        self.training_questions_concepts_answers_file_path = training_questions_concepts_answers_file_path
        self.testing_questions_concepts_answers_file_path = testing_questions_concepts_answers_file_path
        self.training_questions_embeddings_path = training_questions_embeddings_path
        self.testing_questions_embeddings_path = testing_questions_embeddings_path
        self.is_vad_kb = is_vad_kb
        self.training_subgraphs_file_path= training_subgraphs_file_path
        self.testing_subgraphs_file_path=testing_subgraphs_file_path




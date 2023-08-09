class DataContext(object):

    '''The data context class creates a data object that contains all the attributes that represent
    training data'''

    def __init__(self, triples_path:str,entities_labels_path:str,properties_labels_path:str,graph_embeddings_path:str):
        
        self.triples_path = triples_path
        self.entities_labels_path = entities_labels_path
        self.properties_labels_path = properties_labels_path
        self.graph_embeddings_path = graph_embeddings_path


                 
class TrainingContext(object):
    '''The training context class creates an object that contains all the attributes that represent
    training_context of the experiment'''

    def __init__(self, num_epochs:int,learning_rate:float,dim_hidden_layer:int,num_layers=3):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dim_hidden_layer = dim_hidden_layer
        self.num_layers=num_layers


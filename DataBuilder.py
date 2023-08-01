class DataBuilder:
    def __init__(self, turtle_path, NLP_client, path=None):
        def _read_turtle_file(self, turtle_path):
            return graph

        def _get_feature_label(self, label):
            return self.NLP_client(label)

        def _create_x(self):
            """
            Return the  Node feature matrix
            """
            pass

        def _create_edge_attr(self):
            """
            Return Graph connectivity in COO format with
            """
            pass

        def _create_y(self):
            """
            Return Graph-level or node-level ground-truth labels with arbitrary shape.
            For us, initially all will be zero
            """
            pass

    self.graph = "TODO: use library to load turtle into memory"
    self.NLP_client = NLP_client
    self.x = self._create_x()
    self.edge_attr = self._create_edge_attr()
    self.y = self._create_y()

    def persist_data():
        """
        Write Data to disk
        """

import csv
import torch
import tqdm


class DataBuilder:
    def __init__(
        self, triples_path, entities_path, relations_path, NLP_client, graph_path=None
    ):
        def _read_triples_file(self, triples_path, entities_path, relations_path):
            entity = []
            entity_label = []
            with open(entities_path, "r", encoding="utf8") as fin:
                reader = csv.reader(fin)

                # skip the header
                header = next(reader)

                for row in reader:
                    entity.append(row[0])
                    entity_label.append(row[1])
            entity2id = {w: i for i, w in enumerate(entity)}

            relation = []
            relation_label = []
            with open(relations_path, "r", encoding="utf8") as fin:
                reader = csv.reader(fin)

                # skip the header
                header = next(reader)

                for row in reader:
                    relation.append(row[0])
                    relation_label.append(row[1])
            relation2id = {w: i for i, w in enumerate(relation)}

            subject = []
            relation = []
            object = []

            nrow = sum(1 for _ in open(triples_path, "r", encoding="utf-8"))
            with open(triples_path, "r", encoding="utf8") as fin:
                for line in tqdm(fin, total=nrow):
                    ls = line.strip().split(",")

                    subject.append(entity2id[ls[0]])
                    relation.append(relation2id[ls[1]])
                    object.append(entity2id[ls[2]])
            edge_index = torch.stack((torch.tensor(subject), torch.tensor(object)))
            edge_type = torch.tensor(relation)

            return edge_index, edge_type

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

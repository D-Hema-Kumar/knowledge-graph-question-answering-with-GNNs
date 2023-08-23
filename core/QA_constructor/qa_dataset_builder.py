from config.config import  (
    ENTITIES_LABELS_PATH_OLD,
    PROPERTIES_LABELS_PATH_OLD,
    VAD_KGQA_GoldStandard,
    QUESTIONS_CONCEPTS_ANSWERS_PATH)

import json
from tqdm import tqdm
import pandas as pd

class QADatasetBuilder():
    '''The class is for creating a QA dataset which will contain pattern_id, question, concepts, answers for each question.
     The input data is contained in the VAD_KGQA_GoldStandard json file.'''
    def __init__(
        self,
        entities_labels_path,
        properties_labels_path=None,
    ):

        self.entities_data = pd.read_csv(entities_labels_path)

        self.entity_to_index = {
            entity: index
            for index, entity in enumerate(self.entities_data.iloc[:, 0].to_list())
        }
        self.index_to_entity = {
            index: entity
            for index, entity in enumerate(self.entities_data.iloc[:, 0].to_list())
        }

    def get_concept_ids(self,concepts:list):
        conceptIds = []
        for concept in concepts:
            if concept in self.entity_to_index.keys():
                conceptIds.append(self.entity_to_index[concept])
        return conceptIds
    
    def create_qa_dataset(self):
    
        with open(VAD_KGQA_GoldStandard) as gs:
            print(gs)
            data = json.load(gs)

        res = []
        n_samples = len(data['questions'])

        for idx, q in tqdm(enumerate(data['questions']),total=n_samples, desc='loading question, concepts, answers'):
            
            for item in q['question']:
                annotations  = item['annotations']  
                candidates = [annotation['uri'] for annotation in annotations]
                answer = []
                for ans in q['answers']:
                    answer.append(ans['value'])

            
                res.append((q['pattern_id'],item['string'], candidates, answer,q['answertype'],q['aggregation']))
        questions_data = pd.DataFrame.from_records(res,columns=['pattern_id',"question","concepts","answers",'answertype','isAggregation'])

        for col, count_col in [('concepts','count_concepts'),('answers','count_answers')]:
            questions_data[count_col] = questions_data[col].apply( lambda x : len(x))

        for col, id_col in [('concepts','concept_ids'),('answers','answer_ids')]:
            questions_data[id_col] = questions_data[col].apply(self.get_concept_ids)

        # mask if concepts and answers for each question actually present in the main graph.
        concept_ids_mask = questions_data['concept_ids'].str.len()>0
        answer_ids_mask = questions_data['answer_ids'].str.len()>0

        #save resource type questions
        resource_type_QA = questions_data[questions_data['answertype'].isin(['resources','resource'])& concept_ids_mask & answer_ids_mask ].copy()
        resource_type_QA.drop_duplicates(subset='question', keep='first', inplace=True)
        print('Total questions: ',len(resource_type_QA))
        print('Total questions with more than 1 answer:',len(resource_type_QA[resource_type_QA['count_answers']>1]))
        print('Total questions with more than 2 concepts:',len(resource_type_QA[resource_type_QA['count_concepts']>2]))
        resource_type_QA[['pattern_id', 'question', 'concepts', 'answers']].to_csv(QUESTIONS_CONCEPTS_ANSWERS_PATH,index=False)
        print(f'QA data is saved to file:  {QUESTIONS_CONCEPTS_ANSWERS_PATH}')

if __name__ == "__main__":

    qa_dataset_builder = QADatasetBuilder(entities_labels_path=ENTITIES_LABELS_PATH_OLD)
    qa_dataset_builder.create_qa_dataset()
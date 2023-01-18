from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer, DefaultDataCollator
import pandas as pd
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from collections import defaultdict

type2hypothesis = {
0: ['it is related with society or culture', 'this text  describes something about an extended social group having a distinctive cultural and economic organization or a particular society at a particular time and place'],
1: ['it is related with science or mathematics', 'this text  describes something about a particular branch of scientific knowledge or a science (or group of related sciences) dealing with the logic of quantity and shape and arrangement'],
2: ['it is related with health', 'this text  describes something about a healthy state of wellbeing free from disease'],
3: ['it is related with education or reference', 'this text  describes something about the activities of educating or instructing or activities that impart knowledge or skill or an indicator that orients you generally'],
4: ['it is related with computers or Internet', 'this text  describes something about a machine for performing calculations automatically or a computer network consisting of a worldwide network of computer networks that use the TCP/IP network protocols to facilitate data transmission and exchange'],
5: ['it is related with sports', 'this text  describes something about an active diversion requiring physical exertion and competition'],
6: ['it is related with business or finance', 'this text  describes something about a commercial or industrial enterprise and the people who constitute it or the commercial activity of providing funds and capital'],
7: ['it is related with entertainment or music', 'this text  describes something about an activity that is diverting and that holds the attention or an artistic form of auditory communication incorporating instrumental or vocal tones in a structured and continuous manner'],
8: ['it is related with family or relationships', 'this text  describes something about a social unit living together, primary social group; parents and children or a relation between people'],
9: ['it is related with politics or government', 'this text  describes something about social relations involving intrigue to gain authority or power or the organization that is the governing authority of a political unit']}

def read_data_file(path, stage):
    with open(path) as f:
        data = f.readlines()

    seen_types = set()
    for row in data:
        line = row.strip().split('\t')
        if len(line) == 2: # label_id, text
            type_index = int(line[0])
            seen_types.add(type_index)

    exam_co = 0
    line_co = 0
    examples = defaultdict(list)
    for row in data:
        line = row.strip().split('\t')
        if len(line) == 2: # label_id, text
            type_index = int(line[0])
            for i in range(10):
                hypo_list = type2hypothesis.get(i)
                if i == type_index:
                    '''pos pair'''
                    for hypo in hypo_list:
                        examples['guid'].append(f"{stage}-"+str(exam_co))
                        examples['text_a'].append(line[1])
                        examples['text_b'].append(hypo)
                        examples['label'].append('entailment') #if line[0] == '1' else 'not_entailment'
                        examples['type_index'].append(type_index)
                        exam_co+=1
                elif i in seen_types or stage!= 'train':
                    '''neg pair'''
                    for hypo in hypo_list:
                        examples['guid'].append(f"{stage}-"+str(exam_co))
                        examples['text_a'].append(line[1])
                        examples['text_b'].append(hypo)
                        examples['label'].append('not_entailment') #if line[0] == '1' else 'not_entailment'
                        examples['type_index'].append(type_index)
                        exam_co+=1
            line_co+=1
            if line_co % 10000 == 0:
                print(f'loading {stage} data size: {line_co}')
    ds = Dataset.from_dict(examples)
    ds.save_to_disk(f"data/{stage}")


class ZeroShotYahooDataModule(LightningDataModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("YahooData")
        parser.add_argument("--data_train_path", type=str, default="")
        return parent_parser

    def __init__(
        self,
        model_checkpoint: str,
        task_name: str = 'yahoo-topic',
        max_seq_length: int = 128,
        batch_size: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_checkpoint)
        self.data_collator = DefaultDataCollator()
        self.label_map = {'entailment': 0, 'not_entailment': 1}
    
    def setup(self, stage: str):
        self.train_ds = load_from_disk('data/train').shuffle()
        self.dev_ds = load_from_disk('data/dev').shuffle()
        self.test_ds = load_from_disk('data/test').shuffle()

        if self.hparams.debugging:
            self.train_ds = self.train_ds.select(range(10000))
            self.dev_ds = self.dev_ds.select(range(1000))
            self.test_ds = self.test_ds.select(range(1000))

        self.train_ds = self.train_ds.map(
            self.convert_examples_to_features,
            batched = True,
            num_proc= 4,
            remove_columns= ['guid', 'text_a', 'text_b', 'label'],)

        self.dev_ds = self.dev_ds.map(
            self.convert_examples_to_features,
            batched = True,
            num_proc= 4,
            remove_columns= ['guid', 'text_a', 'text_b', 'label'],)

        self.test_ds = self.test_ds.map(
            self.convert_examples_to_features,
            batched = True,
            num_proc= 4,
            remove_columns= ['guid', 'text_a', 'text_b', 'label'],)
       
    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.hparams.model_checkpoint, use_fast=True)
        if self.hparams.data_train_path:
            # Read data train file
            read_data_file(self.hparams.data_train_path, 'train')
            # Read data dev file
            read_data_file('BenchmarkingZeroShot/topic/dev.txt', 'dev')
            # Read data test file 
            read_data_file('BenchmarkingZeroShot/topic/test.txt', 'test')

        train_ds = load_from_disk('data/train')
        seen_types = list(dict.fromkeys(train_ds['type_index']))
        return seen_types

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.hparams.batch_size, collate_fn= self.data_collator, shuffle= True, num_workers =10)

    def val_dataloader(self):
        return DataLoader(self.dev_ds, batch_size = self.hparams.batch_size, collate_fn = self.data_collator, num_workers =10)

    def test_dataloader(self):
        return DataLoader(self.dev_ds, batch_size = self.hparams.batch_size, collate_fn = self.data_collator, num_workers =10)

    def convert_examples_to_features(self, examples):
        inputs = self.tokenizer(
            examples['text_a'], 
            examples['text_b'],
            max_length = self.hparams.max_seq_length,
            padding = 'max_length',
            truncation= True, # 'only_first'
            )
        inputs['label'] = [self.label_map[i] for i in examples['label']]
        return inputs

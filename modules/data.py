import random

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import DataCollatorWithPadding

from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem import AllChem
from rdkit import DataStructs

class DataManager():    
    def __init__(self, data_path, tokenizer):
        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        
        self.df = self.df[['SMILES', 'Ki']].rename(columns={'SMILES':'smiles', 'Ki':'labels'})
        self.df = self.df[self.df['labels'] < float(100000)] #filter Ki to less than 100uM
        self.df = self.df[self.df['smiles'].str.len() < 128]
        
        #for smiles with count > 1 - take average (median)
        self.dup_smiles = self.df.groupby('smiles').filter(lambda x: len(x) > 1).groupby('smiles')['labels'].median().reset_index()
        self.df = self.df.groupby('smiles').filter(lambda x: len(x) == 1)
        self.df = pd.concat([self.df, self.dup_smiles]).reset_index()
        print(f'-- Training on a total dataset of {len(self.df)} labels')
        
        def can(x):
            try:
                return Chem.MolToSmiles(Chem.MolFromSmiles(x), True)            
            except:
                return None
        
        self.df['smiles'] = self.df['smiles'].copy().apply(lambda x: can(x))
        self.df = self.df[self.df['smiles'].notna()]
        self.df['labels'] = -np.log(self.df['labels'])
        #scaler = MinMaxScaler((-1,1))
        #scaler.fit(np.array(self.df['labels']).reshape(-1,1))
        #self.df['labels'] = scaler.transform(np.array(self.df['labels']).reshape(-1,1))
        
    def process_comparativeSMILES(self):
        low_ki_smiles = self.df.sort_values(by='labels', ascending=True).head(n=5)
        low_ki_smiles = low_ki_smiles['smiles'].to_list()
        
        ideal_ligand = low_ki_smiles[0]
                        
        concat_smiles = []
        for smile in self.df['smiles']:
            if smile == ideal_ligand:
                concat_smiles.append(smile)    
            else:
                concat_smiles.append(ideal_ligand + smile)
        self.df['smiles'] = concat_smiles
    
    def generate_scaffold(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=True)
        return scaffold    
    
    def generate_scaffolds(self, dataset: pd.DataFrame):
        dataset['smiles'].apply(lambda x: self.generate_scaffold(x))
    
        scaffolds = {}
        for ind, smiles in enumerate(dataset['smiles']):
            scaffold = self.generate_scaffold(smiles)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)

        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set
            for (scaffold,
                 scaffold_set) in sorted(scaffolds.items(),
                                         key=lambda x: (len(x[1]), x[1][0]),
                                         reverse=True)
        ]
        return scaffold_sets
    
    def split_fingerprints(self, dataset: pd.DataFrame, train_split):
        smiles = dataset['smiles']
        #generate mols and fingerprints
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]
        
        fp_in_group = [[fps[0]], []]
        indices_in_group = ([0], [])
        remaining_fp = fps[1:]
        remaining_indices = list(range(1, len(fps)))
        max_similarity_to_group = [DataStructs.BulkTanimotoSimilarity(fps[0], remaining_fp), [0] * len(remaining_fp)]
        
        train_size = len(dataset) * train_split
        test_size = len(dataset) - train_size

        while len(remaining_fp) > 0:
            # Decide which group to assign a molecule to.
            group = 0 if len(fp_in_group[0]) / train_size <= len(fp_in_group[1]) / test_size else 1
            # Identify the unassigned molecule that is least similar to everything in
            # the other group.
            i = np.argmin(max_similarity_to_group[1 - group])
            # Add it to the group.
            fp = remaining_fp[i]
            fp_in_group[group].append(fp)
            indices_in_group[group].append(remaining_indices[i])
            # Update the data on unassigned molecules.
            similarity = DataStructs.BulkTanimotoSimilarity(fp, remaining_fp)
            max_similarity_to_group[group] = np.delete(
                np.maximum(similarity, max_similarity_to_group[group]), i)
            max_similarity_to_group[1 - group] = np.delete(
                max_similarity_to_group[1 - group], i)
            del remaining_fp[i]
            del remaining_indices[i]
        return indices_in_group     
    
    def getAlternateProteinSplit(self, train_target_path, test_target_path):
        train_target = DataManager(train_target_path, self.tokenizer).df
        test_target = DataManager(test_target_path, self.tokenizer).df
        train_target = train_target.merge(test_target.drop(['labels'], axis=1), on=['smiles'], how='left', indicator=True).query('_merge == "left_only"').drop(['_merge'], axis=1)
        test_target = test_target.merge(train_target.drop(['labels'], axis=1), on=['smiles'], how='left', indicator=True).query('_merge == "left_only"').drop(['_merge'], axis=1)       
        return train_target, test_target
    
    def getTrainTestSplit(self, train_split, method):
        if method == 'random':
            self.train_df, self.test_df = train_test_split(self.df, train_size=train_split, shuffle=True, random_state=11)
        if method == 'scaffold':
            train_inds = []
            test_inds = []
            train_cutoff = train_split * len(self.df)
            scaffold_sets = self.generate_scaffolds(self.df)
            for scaffold_set in scaffold_sets:
                if len(train_inds) + len(scaffold_set) > train_cutoff:
                        test_inds += scaffold_set
                else:
                    train_inds += scaffold_set
            self.train_df = self.df.iloc[train_inds]
            self.test_df = self.df.iloc[test_inds]
        if method == 'fingerprintSimilarity':
            train_inds, test_inds = self.split_fingerprints(self.df, train_split)
            self.train_df = self.df.iloc[train_inds]
            self.test_df = self.df.iloc[test_inds]
        if method == 'alternateProteinSplit':
            train_target_path, test_target_path = [#'/data/ben/bindingDB/processed_data/bindingDB_DRD2_HUMAN_P14416.csv',
                                                   '/data/ben/bindingDB/processed_data/bindingDB_OPRM_HUMAN_P35372.csv',
                                                   '/data/ben/bindingDB/processed_data/bindingDB_OPRK_HUMAN_P41145.csv'] #'/data/ben/bindingDB/processed_data/bindingDB_CAH2_HUMAN_P00918.csv']     
            self.train_df, self.test_df = self.getAlternateProteinSplit(train_target_path, test_target_path)
                   
    def buildDataLoader(self, batch_size, ddp):
        dataset = DatasetDict({
            'train': Dataset.from_pandas(self.train_df),
            'test': Dataset.from_pandas(self.test_df)
        }) 
        
        def tokenize_function(dataset):
            return self.tokenizer(dataset['smiles'], padding='max_length', truncation=True, max_length=128)
        def reshape_labels(dataset):
            dataset['labels'] = np.array([dataset['labels']]).reshape(-1,1)
            return dataset
        self.dataset = dataset.map(tokenize_function, batched = True).map(reshape_labels, batched=True).remove_columns(['smiles', '__index_level_0__'])
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer) 

        if ddp:
            train_dataloader = DataLoader(
                self.dataset["train"], batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(self.dataset['train']), collate_fn=data_collator
            )
            test_dataloader = DataLoader(
                self.dataset["test"], batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(self.dataset['test']), collate_fn=data_collator
            )
        else:
            train_dataloader = DataLoader(
                self.dataset["train"], batch_size=batch_size, pin_memory=True, shuffle=False, collate_fn=data_collator
            )
            test_dataloader = DataLoader(
                self.dataset["test"], batch_size=batch_size, pin_memory=True, shuffle=False, collate_fn=data_collator
            )
        
        return train_dataloader, test_dataloader
    
if __name__ == '__main__':
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
    d = DataManager('/data/ben/bindingDB/processed_data/bindingDB_OPRK_HUMAN_P41145.csv', tok)    
    
    import glob
    data_path = '/data/ben/bindingDB/processed_data/'
    lst = []
    for l in glob.glob(data_path + '*.csv'):
        d = DataManager(l, tok)
        lst.append(len(d.df))
    for n, l in zip(glob.glob(data_path + '*.csv'), lst):
        print(n,l)
    
    train_target_path = '/data/ben/bindingDB/processed_data/bindingDB_OPRK_HUMAN_P41145.csv'
    test_target_path = '/data/ben/bindingDB/processed_data/bindingDB_DRD2_HUMAN_P14416.csv'
    
    t1 = DataManager(train_target_path, tok).df
    t2 = DataManager(test_target_path, tok).df
    
    t1.merge(t2, on=['smiles'], how='left', indicator=True).query('_merge == "left_only"')
    
    train_target_path, test_target_path = ['/data/ben/bindingDB/processed_data/bindingDB_DRD2_HUMAN_P14416.csv',
                                        '/data/ben/bindingDB/processed_data/bindingDB_JAK2_HUMAN_O60674.csv',] #'/data/ben/bindingDB/processed_data/bindingDB_CAH2_HUMAN_P00918.csv']     
    train_df, test_df = d.getAlternateProteinSplit(train_target_path, test_target_path)
import os
import random

import torch.multiprocessing as mp
import glob

from model import *
from modules import optimisers, loss

import traceback

if __name__ == '__main__':

    model_link = 'seyonec/ChemBERTa-zinc-base-v1'
    project = 'sheepy-getting-sleepy'
    
    #set together
    world_size = 2
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    
    model_list = [#'seyonec/ChemBERTa-zinc-base-v1',
                  #'seyonec/ChemBERTa-zinc250k-v1',
                  #'seyonec/ChemBERTA_PubChem1M_shard00',
                  #'DeepChem/ChemBERTa-10M-MLM',
                  'DeepChem/ChemBERTa-10M-MTR',]
                  #'DeepChem/ChemBERTa-77M-MLM',
                #   'DeepChem/ChemBERTa-77M-MTR']
    
    data_path = [#'/data/ben/bindingDB/processed_data/bindingDB_OPRK_HUMAN_P41145.csv',
                 '/data/ben/bindingDB/processed_data/bindingDB_DRD2_HUMAN_P14416.csv',]
                 #'/data/ben/bindingDB/processed_data/bindingDB_OPRM_HUMAN_P35372.csv']
    # data_list = glob.glob(data_path + '*.csv')
    prot_names = [dp.split('/')[-1].split('.')[0] for dp in data_path]

    parameters = {
        'num_epochs': [100],
        'batch_size': [128],
        'train_split': [0.85],
        'lr': [0.0000690452546751227],
        'loss_fn': [loss.RMSE_loss],
        'optimiser': [optimisers.Optim_BaseAdamW],
        'scheduler': [torch.optim.lr_scheduler.LinearLR],
        'num_warmup_steps': [10],
        #'reinit_n': [0,5,10,20],
        'comp_smiles': [False],
        'split_method': ['random',] # 'scaffold', 'fingerprintSimilarity',] #['alternateProteinSplit'] 
    }
    
    ddp = True
    log = True
    
    for model in model_list:
        for prot, prot_name in zip(data_path, prot_names):
            tags = [model, prot_name, 'hidden_states', ] #oprm_oprk
            for split in parameters['split_method']:
                for expt in range(1):
                    print(f'''
                        Model: {model}
                        Protein: {prot}
                        Expt No.: {expt}
                        ''')
                    config = {}
                    config['protein'] = prot_name
                    config['model'] = model
                    for k,v in parameters.items():
                        if isinstance(v, list):
                            # if k == 'comp_smiles':
                            #     v = np.random.choice(v, p = [0.25, 0.75])
                            # else:
                            v = random.sample(v, 1).pop()
                        elif isinstance(v, dict):
                            _min = v['min']
                            _max = v['max']
                            v = random.uniform(_min, _max)
                        config[k] = v
                    config['split_method'] = split

                    # model is a hf link to repo
                    # prot is a reference to a df in /data
                    try:
                        if ddp:
                            mp.spawn(fn=main_ddp, #takes in a function, and 'spawns' that across all of our processes 
                                args=(world_size, model, prot, project, tags, config, ddp, log), #note, mp.spawn automatically includes rank - no need to include in args
                                nprocs=world_size)
                        else:
                            main(model, prot, project, tags, config, ddp, log)
                    except:
                        traceback.print_exc()
                        print(f'---- Error has occurred - see stack trace above for details.')
                        print(f'---- Protein: {prot_name} ---- Model: {model}')
                        input('Press Enter to Retry')
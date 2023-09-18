import os

import pandas as pd
import wandb
import sklearn
import scipy

class WandBLogger():
    def __init__(self, 
                 rank, 
                 project, 
                 tags, 
                 config, 
                 model_link) -> None:
        self.rank = rank
        wandb.login(key = '405542679c65fa8dc14eaf9c86e73b6dc1b3c70d')
        wandb.init(
            project=project,
            tags=tags,
            config=config,
            notes=model_link)
        
    def log(self,
            model_dir,
            num_epochs,
            labels,
            preds,
            epoch,
            end,
            start,
            train_loss,
            test_loss,
            lr_scheduler,
            optimiser):     
        
        self.model_dir = model_dir
        label_preds_df = pd.DataFrame({'labels': labels,
                            'preds': preds})     
        label_preds_df.to_csv(os.path.join(self.model_dir, 'label_preds.csv'))
        
        wandb.log({
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_mse': sklearn.metrics.mean_squared_error(labels, preds),
            'test_rmse': sklearn.metrics.mean_squared_error(labels, preds, squared=False),
            'test_mae': sklearn.metrics.mean_absolute_error(labels, preds),
            'test_ssr': ((labels - preds)**2).sum(),
            'test_r': scipy.stats.pearsonr(labels, preds).statistic,
            'test_r2': sklearn.metrics.r2_score(labels, preds),
            'elapsed': end - start,
            'epoch': epoch + 1,
            'percentage_complete': (epoch / num_epochs)*100
            })
        
        lr = lr_scheduler.get_last_lr() 
        
        if 'LLRD' in optimiser.__class__.__name__:
            param_names = [i['name'] for i in optimiser.optimiser.param_groups] 
            LLRD_lr_table = pd.DataFrame({'param_names': param_names, 'lr': lr}).drop_duplicates()
            wandb.log({
                'lr_llrd': LLRD_lr_table,
                })
        else:
            [lr] = lr
            wandb.log({
                'lr': lr
            })
        print('Logged to W&B.')

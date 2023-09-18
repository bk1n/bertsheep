import torch

class Optim_BaseAdamW():
    def __init__(self, model, init_lr) -> None:
        self.optimiser = torch.optim.AdamW(model.parameters(), lr=init_lr)
        
class Optim_LLRDAdamW_reduce(Optim_BaseAdamW):
    '''Builds an optimiser with layer wise learning rate decay (LLRD) on AdamW.
    Reduces LR throughout layers, so classifier fine-tuning head has highest learning rate.'''
    def __init__(self, model, init_lr):        
        opt_parameters = []
        named_parameters = list(model.named_parameters()) 
        
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        head_lr = init_lr * 1.1
        lr = init_lr
        
        #parameters for classifier
        params_0 = [p for n,p in named_parameters if "classifier" in n and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if "classifier" in n and not any(nd in n for nd in no_decay)]
        classifier_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0, "name": 'ClassifierParams'}    
        opt_parameters.append(classifier_params)
        classifier_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01, "name": 'ClassifierParams'}    
        opt_parameters.append(classifier_params)
        
        #parameters for encoder layers
        for layer in range(5,-1,-1):        
            params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                        and any(nd in n for nd in no_decay)]
            params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                        and not any(nd in n for nd in no_decay)]
            
            layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0, 'name': f'EncoderParams_layer{layer}'}
            opt_parameters.append(layer_params)
                                           
            layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01, 'name': f'EncoderParams_layer{layer}'}
            opt_parameters.append(layer_params)
            
            lr *= 0.9
            
        #parameters for embedding layers
        params_0 = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
        embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0, 'name': 'EmbeddingParams'} 
        opt_parameters.append(embed_params)
        embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01, 'name': 'EmbeddingParams'} 
        opt_parameters.append(embed_params)   
                
        self.optimiser = torch.optim.AdamW(opt_parameters, lr=init_lr)
        
class Optim_LLRDAdamW_gain(Optim_BaseAdamW):
    '''Builds an optimiser with layer wise learning rate decay (LLRD) on AdamW.
    Increases LR throughout layers, so classifier fine-tuning head has lowest learning rate.'''
    def __init__(self, model, init_lr):        
        opt_parameters = []
        named_parameters = list(model.named_parameters()) 
        
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        head_lr = init_lr * 0.9
        lr = init_lr
        
        #parameters for classifier
        params_0 = [p for n,p in named_parameters if "classifier" in n and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if "classifier" in n and not any(nd in n for nd in no_decay)]
        classifier_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0, "name": 'ClassifierParams'}    
        opt_parameters.append(classifier_params)
        classifier_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01, "name": 'ClassifierParams'}    
        opt_parameters.append(classifier_params)
        
        #parameters for encoder layers
        for layer in range(5,-1,-1):        
            params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                        and any(nd in n for nd in no_decay)]
            params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                        and not any(nd in n for nd in no_decay)]
            
            layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0, 'name': f'EncoderParams_layer{layer}'}
            opt_parameters.append(layer_params)
                                           
            layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01, 'name': f'EncoderParams_layer{layer}'}
            opt_parameters.append(layer_params)
            
            lr *= 1.1
            
        #parameters for embedding layers
        params_0 = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
        embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0, 'name': 'EmbeddingParams'} 
        opt_parameters.append(embed_params)
        embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01, 'name': 'EmbeddingParams'} 
        opt_parameters.append(embed_params)   
                
        self.optimiser = torch.optim.AdamW(opt_parameters, lr=init_lr)
        
    
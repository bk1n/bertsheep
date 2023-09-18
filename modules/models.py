from transformers import RobertaForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F

# CUSTOM MODULES
class CustomClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(self, features, **kwargs):
        '''Build a custom classifier head for ChemBERTa'''
        x = features[:, 0, :]
        x = self.dropout(F.relu(self.dense(x)))
        x = self.dropout(F.relu(self.dense(x)))
        x = self.dropout(F.relu(self.dense(x)))
        x = self.out_proj(x)
        x = torch.tanh(x)
        return x
    
class VerboseExecution(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for name, layer in self.model.named_modules():
            if 'roberta.encoder.layer.2.output' in name:
                layer.__name__ = name
                layer.register_forward_hook(
                    lambda layer, _, output: print(f'{layer.__name__}: {output.shape}')
                )
    
    def forward(self, *args):
        return self.model(*args)
    
class ActivationHook(nn.Module):
    def __init__(self, model, layers) -> None:
        super().__init__()
        self.model = model
        self.layers = layers
        self._activations = {layer: torch.empty(0) for layer in layers}
        
        for layer_name in layers:
            layer = dict([*self.model.named_modules()])[layer_name]
            layer.register_forward_hook(self.getActivation(layer_name))

    def getActivation(self, layer_name: str):
        def fn(_, __, output):
            self._activations[layer_name] = output
        return fn
                
    def forward(self, *args):
        return self.model(*args)

# MODELS
class Model_Base():
    def __init__(self, model_link):
        self.model = RobertaForSequenceClassification.from_pretrained(model_link, num_labels = 1, output_attentions=True)
        
class Model_CustomClassifier(Model_Base):
    def __init__(self, model_link):
        super().__init__(model_link)
        self.model.classifier = CustomClassifier(config=self.model.config)
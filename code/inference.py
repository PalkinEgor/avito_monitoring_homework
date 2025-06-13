import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import ModuleUtilsMixin


def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class ModelForSequenceClassification(nn.Module, ModuleUtilsMixin):
    def __init__(self, MODEL_NAME, num_classes=4):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            MODEL_NAME, torch_dtype="auto",
        )
        config = self.model.config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        x = average_pool(outputs.last_hidden_state, attention_mask)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def inference(self, tokenizer, device, text):
        inputs = tokenizer(
            [text], max_length=512,
            padding=False, truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = self.forward(**inputs)
            probs = F.softmax(logits, dim=1)
        return probs[0].cpu().tolist()
    

class AppModel:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.small_model = ModelForSequenceClassification('intfloat/multilingual-e5-small').to(self.device)
        self.small_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
        small_checkpoint = torch.load('model/e5_small.pt', weights_only=False, map_location=self.device)
        self.small_model.load_state_dict(small_checkpoint['model_state_dict'])

        # self.large_model = ModelForSequenceClassification('intfloat/multilingual-e5-large').to(self.device)
        # self.large_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        # large_checkpoint = torch.load('model/e5_large.pt', weights_only=False, map_location=self.device)                
        # self.large_model.load_state_dict(large_checkpoint['model_state_dict'])
    
    def inference(self, text, name):
        IDX_TO_LABEL = [
            "Нейтральное",
            "Оскорбление",
            "Угроза",
            "Домогательство",
        ]
        # MODEL = {'small': self.small_model, 'large': self.large_model}.get(name, self.small_model)
        return IDX_TO_LABEL[np.argmax(self.small_model.inference(self.small_tokenizer, self.device, text))]
    

if __name__ == '__main__':
    model = AppModel()
    print(model.inference('Привет'))
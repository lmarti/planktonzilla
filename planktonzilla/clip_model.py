import torch.nn as nn
import open_clip
from transformers.modeling_outputs import ImageClassifierOutput

class ClipClassifier(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: str,
        repo_path: str,
        num_labels: int,
        id2label: dict = None,
        label2id: dict = None,
    ):
        super().__init__()

        if repo_path:
            clip_model, _, _ = open_clip.create_model_and_transforms(repo_path)

        else:
            clip_model, _, _ = open_clip.create_model_and_transforms(name, pretrained)

        self.id2label = id2label
        self.label2id = label2id
        self.num_labels = num_labels
        
        self.name_or_path = name + pretrained
        self.model = clip_model.visual.trunk

        # Head de clasificación
        self.model.head = nn.Linear(1024, num_labels)

    def forward(self, pixel_values, labels=None, output_attentions=None, output_hidden_states=None, return_dict=True):
        logits = self.model(pixel_values)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            # order: loss, logits, hidden_states, attentions
            return (loss, logits, None, None)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

import torch

from typing import Optional
from torch import nn
from transformers import PreTrainedModel


class VisionTextClassifier(PreTrainedModel):

    def __init__(self, config, vision_model, text_model, num_labels=1):
        super().__init__(config)
        self.config = config

        self.vision_model = vision_model
        self.text_model = text_model
        self.num_labels = num_labels

        self.classification_head = \
            nn.Linear(self.vision_model.config.hidden_size
                      + self.text_model.config.hidden_size, self.num_labels)

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                labels=None,
                token_type_ids: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = True):
        return_dict = return_dict if \
            return_dict is not None else self.config.return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]

        pooled_output = torch.cat((image_embeds, text_embeds), dim=1)
        logits = self.classification_head(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                    loss_fct = nn.MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else logits

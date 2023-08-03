from torch.nn import MSELoss

from transformers.modeling_outputs import (
    Seq2SeqModelOutput,
    Seq2SeqSequenceClassifierOutput,
    SequenceClassifierOutput,
)
from transformers.adapters.heads.base import PredictionHead


class RegressionHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        id2label=None,
        use_pooler=False,
        bias=True,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "classification",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()}
            if id2label is not None
            else None,
            "use_pooler": use_pooler,
            "bias": bias,
        }
        self.build(model)

    def forward(
        self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs
    ):
        if cls_output is None:
            if self.config["use_pooler"]:
                cls_output = kwargs.pop("pooled_output")
            else:
                cls_output = outputs[0][:, 0]
        logits = super().forward(cls_output)
        loss = None
        labels = kwargs.pop("labels", None)
        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

        if return_dict:
            if isinstance(outputs, Seq2SeqModelOutput):
                return Seq2SeqSequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    decoder_hidden_states=outputs.decoder_hidden_states,
                    decoder_attentions=outputs.decoder_attentions,
                    cross_attentions=outputs.cross_attentions,
                    encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                    encoder_hidden_states=outputs.encoder_hidden_states,
                    encoder_attentions=outputs.encoder_attentions,
                )
            else:
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        else:
            outputs = (logits,) + outputs[1:]
            if labels is not None:
                outputs = (loss,) + outputs
            return outputs

import evaluate
from transformers import Trainer
import torch

MODELS = ['bert-base-uncased']  # TODO add other models
SPECIAL_TOKENS = ['[bd]', '[br]', '[address]', '[overview]', '[sqft]']
FINE_TUNNING_FORMAT = "[bd] {bed} [br] {bath} [sqft] {sqft} [address] " \
                      "{street} {city} {state} [overview] {overview}"


class SmoothL1Trainer(Trainer):
    def __init__(self, *rargs, **kwargs):
        beta = kwargs.pop("beta", 0.5)
        super().__init__(*rargs, **kwargs)
        self.criterion = torch.nn.SmoothL1Loss(beta=beta)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = self.criterion(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def get_metrics_func():
    """
    creates a function that can be used to compute metrics
    :return: a function that calculates metrics
    """
    r2 = evaluate.load("r_squared")
    mse = evaluate.load("mse")
    mae = evaluate.load("mae")

    def compute_metrics(pred):
        """
        computes metrics
        :param pred: predictions
        :return: metrics dictionary
        """
        labels = pred.label_ids
        preds = pred.predictions
        metrics = {'r2': r2.compute(references=labels, predictions=preds)}
        metrics.update(mse.compute(references=labels, predictions=preds))
        metrics.update(mae.compute(references=labels, predictions=preds))
        return metrics

    return compute_metrics

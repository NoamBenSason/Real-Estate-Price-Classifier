import evaluate
import pandas as pd
# import wandb

from datasets import Dataset
from preprocessing import format_dataframe
from transformers import TrainingArguments, Trainer, \
    AutoModelForSequenceClassification, AutoTokenizer
import torch

MODELS = ['bert-base-uncased']
SPECIAL_TOKENS = ['[bd]', '[br]', '[address]', '[overview]', '[sqft]']
FINE_TUNNING_FORMAT = "[bd]{bed}[br]{bath}[sqft]{sqft}[overview]{overview}[SEP]"


class SmoothL1Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        beta = kwargs.pop("beta", 0.5)
        super().__init__(*args, **kwargs)
        self.criterion = torch.nn.SmoothL1Loss(beta=self.beta)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = self.criterion(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def tokenize_func(row, tokenizer, with_label=True):
    tokenized_inputs = tokenizer(row["description"], truncation=True)
    if with_label:
        tokenized_inputs["label"] = row['label']

    return tokenized_inputs


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


def train_model(model, tokenizer, train_dataset, validation_dataset,
                save_strategy, config=None, use_wandb=False):
    """
    trains a model for a single run
    :param model: model to train
    :param tokenizer: tokenizer to use
    :param train_dataset: training dataset
    :param validation_dataset: validation dataset
    :param save_strategy: save strategy
    :param config: train config to use config
    :param use_wandb: whether to use wandb
    :return:
    """
    train_args = TrainingArguments(output_dir="./results",
                                   save_strategy=save_strategy,
                                   evaluation_strategy="steps",
                                   report_to=["wandb"] if use_wandb else [
                                       'none'])
    beta = config['beta'] if config is not None else 0.5
    if config is not None:
        train_args.learning_rate = config['learning_rate']
        train_args.num_train_epochs = config['epoch']
        train_args.weight_decay = config['weight_decay']


    trainer = SmoothL1Trainer(model=model,
                              args=train_args,
                              train_dataset=train_dataset,
                              eval_dataset=validation_dataset,
                              compute_metrics=get_metrics_func(),
                              tokenizer=tokenizer,
                              beta = beta
                              )

    trainer.train()
    return trainer, trainer.evaluate(eval_dataset=validation_dataset)


def fine_tune_model(model_name, special_tokens, train_dataset,
                    validation_dataset, save_strategy,
                    config=None, use_wandb=False):
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Update tokenizer and size
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    # Encode data
    encoded_train_dataset = train_dataset.map(
        lambda x: tokenize_func(x, tokenizer), batched=True)
    encoded_validation_dataset = validation_dataset.map(
        lambda x: tokenize_func(x, tokenizer), batched=True)

    trainer, eval_results = train_model(
        model, tokenizer, encoded_train_dataset, encoded_validation_dataset,
        save_strategy, config, use_wandb
    )

    return trainer.predict(encoded_validation_dataset), eval_results


def convert_data(data):
    formatted_data = format_dataframe(data, FINE_TUNNING_FORMAT)
    formatted_df = pd.DataFrame(formatted_data,
                                columns=['description', 'label'])

    return Dataset.from_pandas(formatted_df)


def main():
    train_dataset = convert_data('train_data.csv')
    validation_dataset = convert_data('validation_data.csv')

    for model_name in MODELS:
        predictions, eval_results = fine_tune_model(
            model_name, SPECIAL_TOKENS, train_dataset, validation_dataset, "no")

        print(eval_results)


if __name__ == "__main__":
    main()

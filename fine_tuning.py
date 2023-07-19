import evaluate
import pandas as pd
# import wandb

from datasets import Dataset
from preprocessing import format_dataframe, parse_address
from transformers import TrainingArguments, Trainer, \
    AutoModelForSequenceClassification, AutoTokenizer
import torch
from data_augmentation import DataAugmentation
import argparse


MODELS = ['bert-base-uncased']  # TODO add other models
SPECIAL_TOKENS = ['[bd]', '[br]', '[address]', '[overview]', '[sqft]']
FINE_TUNNING_FORMAT = "[bd] {bed} [br] {bath} [sqft] {sqft} [address] " \
                      "{street} {city} {state} [overview] {overview}"


class SmoothL1Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        beta = kwargs.pop("beta", 0.5)
        super().__init__(*args, **kwargs)
        self.criterion = torch.nn.SmoothL1Loss(beta=beta)

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
                                   evaluation_strategy="epoch",
                                   logging_strategy="steps",
                                   logging_steps=50,
                                   report_to=["wandb"] if use_wandb else [
                                       'none'],
                                   remove_unused_columns=False
                                   )
    # beta = config['beta'] if config is not None else 0.5
    if config is not None:
        train_args.learning_rate = config['learning_rate']
        train_args.num_train_epochs = config['epoch']
        train_args.weight_decay = config['weight_decay']

    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=train_dataset,
                      eval_dataset=validation_dataset,
                      compute_metrics=get_metrics_func(),
                      tokenizer=tokenizer)
                      # beta=beta


    trainer.train()
    return trainer, trainer.evaluate(eval_dataset=validation_dataset)


def get_data_augmentor(tokenizer, del_p):
    aug = DataAugmentation()

    def augmentor(batch):
        batch['overview'] = aug.random_remove(batch['overview'], del_p)
        augmented_batch_description = [
            FINE_TUNNING_FORMAT.format(
                bed=bed, bath=bath, sqft=sqft, street=street, city=city, state=state,  overview=overview
            )
            for bed, bath, sqft, street, city, state, overview in zip(
                batch['bed'],
                batch['bath'],
                batch['sqft'],
                batch['street'],
                batch['city'],
                batch['state'],
                batch['overview']
            )
        ]

        tokenized_inputs = tokenizer(augmented_batch_description, truncation=True)
        tokenized_inputs['label'] = batch['price']

        return tokenized_inputs

    return augmentor


def fine_tune_model(model_name, special_tokens, train_dataset,
                    validation_dataset, save_strategy,
                    config=None, use_wandb=False, use_augment=False, del_p=0.1):
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Update tokenizer and size
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    # Encode data
    if not use_augment:
        train_dataset = train_dataset.map(
            lambda x: tokenize_func(x, tokenizer), batched=True)
        train_dataset = train_dataset.remove_columns(['description'])
    else:
        train_dataset.set_transform(get_data_augmentor(tokenizer, del_p))
    validation_dataset = validation_dataset.map(
        lambda x: tokenize_func(x, tokenizer), batched=True)
    validation_dataset = validation_dataset.remove_columns(['description'])

    trainer, eval_results = train_model(
        model, tokenizer, train_dataset, validation_dataset,
        save_strategy, config, use_wandb
    )

    return trainer.predict(validation_dataset), eval_results


def convert_data(data):
    formatted_data = format_dataframe(data, FINE_TUNNING_FORMAT)
    formatted_df = pd.DataFrame(formatted_data,
                                columns=['description', 'label'])

    return Dataset.from_pandas(formatted_df)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--augment", default=False,type=lambda x: x == "True",help="use "
                                                                 "augmented data")
    args.add_argument("--del_p", default=0.1,type=float, help="probability to "
                                                         "delete")

    args = args.parse_args()
    # print(args)
    if not args.augment:
        train_dataset = convert_data('train_data.csv')
    else:
        train_dataset = pd.read_csv('train_data_with_aug.csv')
        train_dataset = Dataset.from_pandas(train_dataset)
    validation_dataset = convert_data('validation_data.csv')

    for model_name in MODELS:
        predictions, eval_results = fine_tune_model(
            model_name, SPECIAL_TOKENS, train_dataset, validation_dataset,
            "no", use_augment=args.augment, del_p=args.del_p)

        print(eval_results)


if __name__ == "__main__":
    main()

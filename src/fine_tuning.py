import pandas as pd

from datasets import Dataset
from preprocessing import format_dataframe
from transformers import TrainingArguments, \
    AutoModelForSequenceClassification, AutoTokenizer
from data_augmentation import DataAugmentation
import argparse
from fine_tuning_utils import FINE_TUNNING_FORMAT, MODELS, SPECIAL_TOKENS, \
    SmoothL1Trainer,get_metrics_func


def tokenize_func(row, tokenizer, with_label=True):
    tokenized_inputs = tokenizer(row["description"], truncation=True)
    if with_label:
        tokenized_inputs["label"] = row['label']

    return tokenized_inputs


def train_model(model, tokenizer, train_dataset, validation_dataset,
                save_strategy, config=None, use_wandb=False,seed=42):
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
    train_args = TrainingArguments(output_dir="../results",
                                   save_strategy=save_strategy,
                                   evaluation_strategy="epoch",
                                   logging_strategy="steps",
                                   logging_steps=50,
                                   report_to=["wandb"] if use_wandb else [
                                       'none'],
                                   remove_unused_columns=False,
                                   seed=seed
                                   )
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
                              beta=beta)

    trainer.train()

    return trainer, trainer.evaluate(eval_dataset=validation_dataset)


def get_data_augmentor(tokenizer, del_p):
    aug = DataAugmentation()

    def augmentor(batch):
        batch['overview'] = aug.random_remove(batch['overview'], del_p)
        augmented_batch_description = [
            FINE_TUNNING_FORMAT.format(
                bed=bed, bath=bath, sqft=sqft, street=street, city=city,
                state=state, overview=overview
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

        tokenized_inputs = tokenizer(augmented_batch_description,
                                     truncation=True)
        tokenized_inputs['label'] = batch['price']

        return tokenized_inputs

    return augmentor


def fine_tune_model(model_name, special_tokens, train_dataset,
                    validation_dataset, save_strategy,
                    config=None, use_wandb=False, use_augment=False,
                    del_p=0.1,seed=42):
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
        save_strategy, config, use_wandb,seed=seed
    )

    output = trainer.predict(validation_dataset), eval_results

    del model
    # torch._C._cuda_emptyCache()
    return output, trainer


def convert_data(data):
    formatted_data = format_dataframe(data, FINE_TUNNING_FORMAT)
    formatted_df = pd.DataFrame(formatted_data,
                                columns=['description', 'label'])

    return Dataset.from_pandas(formatted_df)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--augment", default=False, type=lambda x: x == "True",
                      help="use "
                           "augmented data")
    args.add_argument("--del_p", default=0.1, type=float, help="probability to "
                                                               "delete")
    args.add_argument("--seed", default=3, type=int,
                      help="seed for fine tuning")

    args = args.parse_args()
    # print(args)
    if not args.augment:
        train_dataset = convert_data('train_data.csv')
    else:
        train_dataset = pd.read_csv('../csvs/train_data_with_aug.csv')
        train_dataset = Dataset.from_pandas(train_dataset)
    validation_dataset = convert_data('validation_data.csv')

    for model_name in MODELS:
        predictions, _ = fine_tune_model(
            model_name, SPECIAL_TOKENS, train_dataset, validation_dataset,
            "no", use_augment=args.augment, del_p=args.del_p)

        print(f"Model: {model_name}")
        print(f"Predictions: {predictions}")


if __name__ == "__main__":
    main()

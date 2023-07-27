import datasets
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForImagesAndTextClassification, AutoConfig, TrainingArguments

from fine_tuning_utils import SmoothL1Trainer, FINE_TUNNING_FORMAT, get_metrics_func, SPECIAL_TOKENS
from preprocessing import format_dataframe


def get_transform_func(processor, with_labels=True):
    def transform(batch):
        processed_inputs = processor(batch["image"], batch["text"],
                                     truncation=True, return_tensors="pt",
                                     max_length=256, padding=True)
        if with_labels:
            processed_inputs["labels"] = batch['labels']

        return processed_inputs

    return transform


def get_multi_model_data(file_name, image_download_dir):

    tuples = format_dataframe(file_name, FINE_TUNNING_FORMAT, with_image=True,
                              image_download_dir=image_download_dir)

    texts = [item[0] for item in tuples]
    images = [item[1] for item in tuples]
    prices = [item[2] for item in tuples]

    images_pil = [Image.open(im[0]) for im in images]  # Takes time

    dict_data = {"text": texts, "image": images_pil, "labels": prices}

    dataset = datasets.Dataset.from_dict(dict_data)

    return dataset


def collate_fn(batch):
    return {
        'input_ids': torch.stack(
            [x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'token_type_ids': torch.stack([x['token_type_ids'] for x in batch]),
        'pixel_values': torch.stack(
            [x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def train_vilt_model(model, processor, train_dataset, eval_dataset,
                     save_strategy, config=None, use_wandb=False, seed=42):
    train_args = TrainingArguments(output_dir="./results",
                                   save_strategy=save_strategy,
                                   evaluation_strategy="epoch",
                                   logging_strategy="steps",
                                   logging_steps=50,
                                   report_to=["wandb"] if use_wandb else [
                                       'none'],
                                   remove_unused_columns=False,
                                   seed=seed)

    beta = config['beta'] if config is not None else 0.5
    if config is not None:
        train_args.learning_rate = config['learning_rate']
        train_args.num_train_epochs = config['epoch']
        train_args.weight_decay = config['weight_decay']

    trainer = SmoothL1Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=get_metrics_func(),
        data_collator=collate_fn,
        tokenizer=processor,
        beta=beta
    )
    trainer.train()
    return trainer, trainer.evaluate(eval_dataset=eval_dataset)


def fine_tune_model(special_tokens, train_dataset, eval_dataset,
                    save_strategy, wandb_config=None, use_wandb=False, seed = 42):
    config = AutoConfig.from_pretrained("dandelin/vilt-b32-mlm", num_labels=1,
                                        num_images=1,
                                        max_position_embeddings=256)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm",
                                              num_labels=1, config=config,
                                              num_images=1, do_resize=True,
                                              size={"shortest_edge": 64},
                                              do_pad=True)
    model = ViltForImagesAndTextClassification.from_pretrained(
        "dandelin/vilt-b32-mlm", ignore_mismatched_sizes=True, config=config)

    processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
    model.resize_token_embeddings(len(processor.tokenizer))

    train_dataset.set_transform(get_transform_func(
        processor))

    eval_dataset.set_transform(get_transform_func(
        processor))

    trainer, eval_results = train_vilt_model(
        model, processor, train_dataset, eval_dataset,
        save_strategy, wandb_config, use_wandb, seed=seed
    )
    outputs = trainer.predict(eval_dataset), eval_results
    return outputs, trainer


def main():
    train_dataset = get_multi_model_data("train_data.csv", "images")
    validation_dataset = get_multi_model_data("validation_data.csv",
                                              "validation_images")

    predictions, trainer = fine_tune_model(SPECIAL_TOKENS, train_dataset,
                                           validation_dataset, "no")

    print(f"Predictions: {predictions}")


if __name__ == '__main__':
    main()

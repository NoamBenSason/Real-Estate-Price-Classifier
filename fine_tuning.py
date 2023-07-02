from transformers import TrainingArguments, Trainer
import evaluate
import wandb


def get_metrics_func():
    """
    creates a function that can be used to compute metrics
    :return: a function that calculates metrics
    """
    r2 = evaluate.load("r_squared")
    rmse = evaluate.load("mse")
    mae = evaluate.load("mae")

    def compute_metrics(pred):
        """
        computes metrics
        :param pred: predictions
        :return: metrics dictionary
        """
        labels = pred.label_ids
        preds = pred.predictions
        r2_score = r2.compute(references=labels, predictions=preds)
        mse_score = rmse.compute(references=labels, predictions=preds)
        mae_score = mae.compute(references=labels, predictions=preds)
        return {"r2": r2_score, "mse": mse_score, "mae": mae_score}

    return compute_metrics


def train_model(model, tokenizer, train_dataset, validation_dataset,
                save_strategy, wandb_config):
    """
    trains a model for a single run
    :param model: model to train
    :param tokenizer: tokenizer to use
    :param train_dataset: training dataset
    :param validation_dataset: validation dataset
    :param save_strategy: save strategy
    :param wandb_config: wandb config
    :return:
    """
    train_args = TrainingArguments(output_dir="./results",
                                       save_strategy=save_strategy,
                                       evaluation_strategy="steps",
                                       report_to=["wandb"],
                                       fp16=True,
                                       learning_rate=wandb_config.lr,
                                       adam_beta1=wandb_config.adam_beta1,
                                       adam_beta2=wandb_config.adam_beta2
                                       )

    trainer = Trainer(model=model, args=train_args,
                          train_dataset=train_dataset,
                          eval_dataset=validation_dataset,
                          compute_metrics=get_metrics_func(),
                          tokenizer=tokenizer, fp16=True)

    trainer.train()
    return trainer, trainer.evaluate(eval_dataset=validation_dataset)

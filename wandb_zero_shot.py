import wandb

from transformers import BertForMaskedLM, RobertaForMaskedLM, ElectraForMaskedLM

from zero_shot import zero_shot, evaluate_zero_shot, BATCH_SIZE
from preprocessing import format_dataframe

MODELS_DICT = {
    # model name: model object, mask format
    "bert-base-uncased": (BertForMaskedLM, "[MASK]"),
    "roberta-base": (RobertaForMaskedLM, "<mask>"),
    "google/electra-base-generator": (ElectraForMaskedLM, "[MASK]")
}


def get_config():
    sweep_config = {'method': 'grid',
                    'metric': {'name': 'mse', 'goal': 'maximize'},
                    'name': f"zero_shot"}

    param_dict = {
        'model_name': {'values': ["bert-base-uncased", "roberta-base",
                                  "google/electra-base-generator"]}
    }

    sweep_config['parameters'] = param_dict
    return sweep_config


def wandb_zero_shot(config=None):
    with wandb.init(config=config, group="zero_shot"):
        config = wandb.config
        model_name = config["model_name"]
        model_for_lm, mask_format = MODELS_DICT[model_name]
        val_data = format_dataframe("validation_data.csv", "{overview}")
        y_hat = []
        y = []
        for i in range(0, len(val_data), BATCH_SIZE):
            truncated_formatted_sentences, true_data_completion, predicted_prices = zero_shot(
                model_name,
                model_for_lm,
                mask_format,
                val_data[i:i + BATCH_SIZE])
            y_hat.extend(predicted_prices)
            y.extend(true_data_completion)

        losses = evaluate_zero_shot(y_hat, y)
        wandb.log({"r2_squared": losses[0],
                   "mse": losses[1],
                   "mae": losses[2]})


if __name__ == '__main__':
    sweep_id = wandb.sweep(get_config(), project="anlp_project",
                           entity="selling_bat_yam")
    wandb.agent(sweep_id, wandb_zero_shot, count=3)

import wandb
from transformers import BertForMaskedLM, RobertaForMaskedLM, ElectraForMaskedLM
from zero_shot import save_to_buffer, save_results, N_EXAMPLE_SAMPLES, zero_shot, BATCH_SIZE, ceil
from preprocessing import build_df_from_data, format_dataframe

MODELS_DICT = {
    # model name: model object, mask format
    "bert-base-uncased": (BertForMaskedLM, "[MASK]"),
    "roberta-base": (RobertaForMaskedLM, "<mask>"),
    "google/electra-base-generator": (ElectraForMaskedLM, "[MASK]")
}


def get_config():
    sweep_config = {}
    sweep_config['method'] = 'random'
    sweep_config['metric'] = {'name': 'mse', 'goal': 'maximize'}
    sweep_config['name'] = f"zero_shot"

    param_dict = {
        'model_name': {'values': ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]}
    }

    sweep_config['parameters'] = param_dict
    return sweep_config


def wandb_zero_shot(config=None):
    with wandb.init(config=config):
        config = wandb.config
        buffer = ""
        model_name = config["model_name"]
        model_for_lm, mask_format = MODELS_DICT[model_name]
        train_data = format_dataframe("train_data.csv", "{overview}")

        avg_losses = [0, 0, 0]
        for i in range(0, len(train_data), BATCH_SIZE):
            losses, truncated_formatted_sentences, true_data_completion, predicted_prices = zero_shot(
                model_name,
                model_for_lm,
                mask_format,
                train_data[i:i + BATCH_SIZE])
            if i == 0:
                examples = [truncated_formatted_sentences[:N_EXAMPLE_SAMPLES],
                            predicted_prices[:N_EXAMPLE_SAMPLES],
                            true_data_completion[:N_EXAMPLE_SAMPLES]]
            avg_losses[0] += losses[0]
            avg_losses[1] += losses[1]
            avg_losses[2] += losses[2]
        num_batches = ceil(len(train_data) / BATCH_SIZE)
        wandb.log({"r2_squared": losses[0] / num_batches,
                   "mse": losses[1] / num_batches,
                   "mae": losses[2] / num_batches})
        buffer = save_to_buffer(model_name, [avg / num_batches for avg in avg_losses], examples, buffer)
        save_results(buffer)


if __name__ == '__main__':
    sweep_id = wandb.sweep(get_config(), project="zero_shot", entity="selling_bat_yam")
    wandb.agent(sweep_id, wandb_zero_shot, count=1)

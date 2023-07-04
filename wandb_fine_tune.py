import wandb
from datetime import datetime
from fine_tuning import fine_tune_model, SPECIAL_TOKENS,convert_data


def get_time():
    """
    :return: the current time
    """
    now = datetime.now()
    return now.strftime("%d-%m-%Y__%H-%M-%S")


def get_config():
    sweep_config = {'method': 'random',
                    'metric': {'name': 'eval/mse', 'goal': 'minimize'}}

    param_dict = {
        'model_name': {'values': ['bert-base-uncased', 'bert-large-uncased',
                                  'roberta-base', 'roberta-large',
                                  'google/electra-base-generator',
                                  'google/electra-large-generator']},
        'epoch': {'value': 15},
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 5e-5,
            'max': 1e-3
        },
        'weight_decay': {
            'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
    }

    sweep_config['parameters'] = param_dict
    return sweep_config


def wandb_run(config=None):
    with wandb.init(config=config, name=f"selling_bat_yam_{get_time()}"):
        config = wandb.config
        train = convert_data("train_data.csv")
        val = convert_data("validation_data.csv")
        fine_tune_model(config['model_name'], SPECIAL_TOKENS, train,
                        val, "no", config, True)


def main():
    sweep_config = get_config()
    sweep_id = wandb.sweep(sweep_config, project="anlp_project",
                           entity="selling_bat_yam")
    wandb.agent(sweep_id, wandb_run, count=1000)
    # config = {
    #     'model_name': 'bert-base-uncased',
    #     'epoch': 1,
    #     'learning_rate': 0.0001,
    #     'weight_decay': 0.0
    # }

    # wandb_run(config)


if __name__ == '__main__':
    main()

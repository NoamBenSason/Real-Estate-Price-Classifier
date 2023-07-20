import pandas as pd
from datasets import Dataset

import wandb
from datetime import datetime
from fine_tuning import fine_tune_model, SPECIAL_TOKENS, convert_data
import argparse


def get_time():
    """
    :return: the current time
    """
    now = datetime.now()
    return now.strftime("%d-%m-%Y__%H-%M-%S")


def get_config(name, augment, del_p):
    sweep_config = {'method': 'random',
                    'name': name,
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
        'beta': {
            'values': [0.0, 0.2, 0.5, 1., 1.5]
        },
        'augment': {
            'value': augment
        },
        'del_p': {
            'value': del_p
        }
    }

    sweep_config['parameters'] = param_dict
    return sweep_config


def wandb_run(config=None):
    with wandb.init(config=config, name=f"selling_bat_yam_{get_time()}",
                    group="fine_tune_aug"):
        config = wandb.config
        if config['augment']:
            train = pd.read_csv('train_data_with_aug.csv')
            train = Dataset.from_pandas(train)
        else:
            train = convert_data("train_data.csv")
        val = convert_data("validation_data.csv")
        fine_tune_model(config['model_name'], SPECIAL_TOKENS, train,
                        val, "no", config, True, config["augment"],
                        config["del_p"])


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--sweep_name",type=str)
    args.add_argument("--augment", default=False, type=lambda x: x == "True",
                      help="use augmented data")
    args.add_argument("--del_p", default=0.1, type=float, help="probability to "
                                                               "delete")

    args = args.parse_args()
    sweep_config = get_config(args.sweep_name,args.augment, args.del_p)
    sweep_id = wandb.sweep(sweep_config, project="anlp_project",
                           entity="selling_bat_yam")

    print(sweep_id)
    wandb.agent(sweep_id, wandb_run, count=1000)
    # config = {
    #     'model_name': 'bert-base-uncased',
    #     'epoch': 1,
    #     'learning_rate': 0.0001,
    #     'weight_decay': 0.0,
    #     'beta': 0.0,
    #     'augment': False,
    #     'del_p': 0.1
    # }
    #
    # wandb_run(config)


if __name__ == '__main__':
    main()

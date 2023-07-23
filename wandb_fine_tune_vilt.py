import torch
import argparse
import wandb
from fine_tune_vilt import fine_tune_model, SPECIAL_TOKENS,get_multi_model_data
from wandb_fine_tune import get_time
import pandas as pd
from datasets import Dataset


def get_config(name, augment, del_p):
    sweep_config = {'method': 'random',
                    'name': name,
                    # 'metric': {'name': 'eval/mse', 'goal': 'minimize'} # todo set metric?
                    }

    param_dict = {
        'epoch': {'distribution': 'int_uniform', 'min': 10, 'max': 25},
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


def run_wandb_vilt(config=None):
    with wandb.init(config=config, name=f"selling_bat_yam_{get_time()}",
                    group="fine_tune_vilt"):
        config = wandb.config

        if config['augment']:
            train_dataset = pd.read_csv('train_data_with_aug.csv')
            train_dataset = Dataset.from_pandas(train_dataset) # todo maybe change later format of images
        else:
            train_dataset = get_multi_model_data("train_data.csv", "images")
        validation_dataset = get_multi_model_data("validation_data.csv", "validation_images")
        if torch.cuda.is_available():
            torch._C._cuda_emptyCache()

        fine_tune_model(SPECIAL_TOKENS, train_dataset, validation_dataset,
                        "no", wandb_config=config, use_wandb=True)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--sweep_name", type=str)
    args.add_argument("--augment", default=False, type=lambda x: x == "True",
                      help="use augmented data")
    args.add_argument("--del_p", default=0.1, type=float, help="probability to delete")

    args = args.parse_args()
    sweep_config = get_config(args.sweep_name, args.augment, args.del_p)
    sweep_id = wandb.sweep(sweep_config, project="anlp_project", entity="selling_bat_yam")

    print(sweep_id)
    wandb.agent(sweep_id, run_wandb_vilt, count=1000)


if __name__ == '__main__':
    main()

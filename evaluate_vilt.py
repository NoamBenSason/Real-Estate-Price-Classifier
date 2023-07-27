import argparse
import json
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import ViltProcessor, AutoConfig

from fine_tune_vilt import get_multi_model_data, fine_tune_model, get_transform_func

SPECIAL_TOKENS = ['[bd]', '[br]', '[address]', '[overview]', '[sqft]']

CONFIG = {
    'learning_rate': 1e-4,
    'model_name': "dandelin/vilt-b32-mlm",
    "del_p": 0,
    'epoch': 15,
    'weight_decay': 0.5,
    'beta': 0.5

}


def get_model_avg_score_and_std(scores):
    results = {
        'avg': {
            'r2_score': np.average([i['test_r2'] for i in scores]),
            'mean_squared_error': np.average([i['test_mse'] for i in scores]),
            'mean_absolute_error': np.average([i['test_mae'] for i in scores]),
        },
        'std': {
            'r2_score': np.std([i['test_r2'] for i in scores]),
            'mean_squared_error': np.std([i['test_mse'] for i in scores]),
            'mean_absolute_error': np.std([i['test_mae'] for i in scores]),
        }
    }

    return results


def get_models_predictions(train_dataset, validation_dataset,
                           test_dataset_in_dist, test_dataset_out_dist, seed, augment=False,
                           del_p=0):
    prediction_results = {}

    # if torch.cuda.is_available():
    #     torch._C._cuda_emptyCache()
    scores_in_dist = []
    scores_out_dist = []
    model_config = CONFIG

    for i in range(seed):
        output, trainer = fine_tune_model(SPECIAL_TOKENS, train_dataset, validation_dataset,
                                          "no", wandb_config=model_config,
                                          seed=i)

        # pros_test_in_dist = test_dataset_in_dist.map(
        #     lambda x: tokenize_func(x, trainer.tokenizer), batched=True)
        # pros_test_in_dist = pros_test_in_dist.remove_columns(
        #     ['description'])
        #
        # pros_test_out_dist = test_dataset_out_dist.map(
        #     lambda x: tokenize_func(x, trainer.tokenizer), batched=True)
        # pros_test_out_dist = pros_test_out_dist.remove_columns(
        #     ['description'])
        #
        # scores_in_dist.append(trainer.evaluate(pros_test_in_dist,
        #                                        metric_key_prefix='test'))
        #
        # scores_out_dist.append(trainer.evaluate(pros_test_out_dist,
        #                                         metric_key_prefix='test'))

        # del trainer
        scores_in_dist.append(trainer.evaluate(eval_dataset=test_dataset_in_dist, metric_key_prefix='test'))
        scores_out_dist.append(trainer.evaluate(eval_dataset=test_dataset_out_dist, metric_key_prefix='test'))

        prediction_results = {'in_dist': get_model_avg_score_and_std(scores_in_dist),
                              'out_dist': get_model_avg_score_and_std(scores_out_dist)}

    return prediction_results


def evaluate_models(train_dataset, validation_dataset, test_dataset_in_dist, test_dataset_out_dist,
                    args):
    results = {}

    # Run without augmentation
    results['Without Augmentation'] = get_models_predictions(train_dataset, validation_dataset,
                                                             test_dataset_in_dist, test_dataset_out_dist,
                                                             args.seed
                                                             )

    return results


def write_results_to_file(results, file_name):
    with open(f'results/{file_name}', 'w') as f:
        f.write(json.dumps(results))


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--seed", default=3, type=int,
                      help="seed for fine tuning")
    args.add_argument("--out_name", default="results", type=str, help="name of output file")

    args = args.parse_args()
    # print(args)
    # if not args.augment:
    train_dataset = get_multi_model_data("train_data.csv", "images")
    # else:
    #     train_dataset = pd.read_csv('train_data_with_aug.csv')
    #     train_dataset = Dataset.from_pandas(train_dataset)
    validation_dataset = get_multi_model_data("validation_data.csv",
                                              "validation_images")

    test_dataset_in_dist = get_multi_model_data("test_data_in_dist.csv", "test_in_dist_images")
    test_dataset_out_dist = get_multi_model_data("test_data_out_dist.csv", "test_out_dist_images")
    vilt_config = AutoConfig.from_pretrained("dandelin/vilt-b32-mlm", num_labels=1,
                                        num_images=1,
                                        max_position_embeddings=256)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm",
                                              num_labels=1, config=vilt_config,
                                              num_images=1, do_resize=True,
                                              size={"shortest_edge": 64},
                                              do_pad=True)

    test_dataset_in_dist.set_transform(get_transform_func(processor))
    test_dataset_out_dist.set_transform(get_transform_func(processor))

    # results = evaluate_models(
    #     MODELS,
    #     train_dataset.select([i for i in range(1)]),
    #     validation_dataset.select([i for i in range(1)]),
    #     test_dataset_in_dist.select([i for i in range(1)]),
    #     test_dataset_out_dist.select([i for i in range(1)]),
    #     args
    # )
    results = evaluate_models(
        train_dataset,
        validation_dataset,
        test_dataset_in_dist,
        test_dataset_out_dist,
        args
    )
    outputfile = args.out_name
    write_results_to_file(results, outputfile)


if __name__ == "__main__":
    main()

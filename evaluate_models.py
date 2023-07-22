import argparse
import json
import numpy as np
import pandas as pd

from datasets import Dataset

from fine_tuning import fine_tune_model, tokenize_func, convert_data

SPECIAL_TOKENS = ['[bd]', '[br]', '[address]', '[overview]', '[sqft]']
MODELS = ['bert-base-uncased']


def get_model_avg_score_and_std(scores):
    results = {
            'Average Score': {
                'r2_score': np.average([i['test_r2'] for i in scores]),
                'mean_squared_error': np.average([i['test_mse'] for i in scores]),
                'mean_absolute_error': np.average([i['test_mae'] for i in scores]),
            },
            'STD': {
                'r2_score': np.std([i['test_r2'] for i in scores]),
                'mean_squared_error': np.std([i['test_mse'] for i in scores]),
                'mean_absolute_error': np.std([i['test_mae'] for i in scores]),
            }
    }

    return results


def get_models_predictions(models, train_dataset, validation_dataset, test_dataset, seed, augment=False,
                           del_p=0):
    prediction_results = {}

    for model_name in models:
        scores = []

        for i in range(seed):
            output, trainer = fine_tune_model(
                model_name, SPECIAL_TOKENS, train_dataset, validation_dataset,
                "no", use_augment=augment, del_p=del_p)

            tokenized_test_dataset = test_dataset.map(
                lambda x: tokenize_func(x, trainer.tokenizer), batched=True)
            tokenized_test_dataset = tokenized_test_dataset.remove_columns(['description'])

            scores.append(trainer.evaluate(tokenized_test_dataset, metric_key_prefix='test'))

        prediction_results[model_name] = get_model_avg_score_and_std(scores)

    return prediction_results


def evaluate_models(models, train_dataset, validation_dataset, test_dataset, args):
    results = {}

    if not args.augment:
        # Run without augmentation
        results['Without Augmentation'] = get_models_predictions(
            models, train_dataset, validation_dataset, test_dataset, args.seed
        )

    # Run with augmentation
    else:
        results['With Augmentation'] = {}
        for del_p in args.del_p_list:
            results['With Augmentation'][f'{del_p}'] = get_models_predictions(
                models, train_dataset, validation_dataset, test_dataset, args.seed, args.augment, del_p
            )

    return results


def write_results_to_file(results, file_name):
    with open(f'results/{file_name}.txt', 'w') as f:
        f.write(json.dumps(results))


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--seed", default=3, type=int, help="seed for fine tuning")
    args.add_argument("--augment", default=False, type=lambda x: x == "True", help="use augmented data")
    args.add_argument("--del_p_list", default=[0.0], nargs='+', type=float,
                      help="list of probabilities to delete words")

    args = args.parse_args()
    # print(args)
    if not args.augment:
        train_dataset = convert_data('train_data.csv')
    else:
        train_dataset = pd.read_csv('train_data_with_aug.csv')
        train_dataset = Dataset.from_pandas(train_dataset)

    validation_dataset = convert_data('validation_data.csv')

    # TODO: for test
    # results = evaluate_models(
    #     MODELS,
    #     train_dataset.select([i for i in range(1)]),
    #     validation_dataset.select([i for i in range(1)]),
    #     validation_dataset.select([i for i in range(1)]),
    #     args
    # )
    #
    # print(results)


if __name__ == "__main__":
    main()

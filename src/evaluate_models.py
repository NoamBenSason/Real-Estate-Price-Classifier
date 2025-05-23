import argparse
import json
import numpy as np
import pandas as pd

from datasets import Dataset

from src.fine_tuning import fine_tune_model, tokenize_func, convert_data

SPECIAL_TOKENS = ['[bd]', '[br]', '[address]', '[overview]', '[sqft]']
MODELS = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator'
          'bert-large-uncased', 'roberta-large',
          'google/electra-large-generator'
          ]

ROBERTA_CONFIG = {
    'epoch': 15,
    'learning_rate': 0.0000671,
    'weight_decay': 0.5,
    'beta': 1.5,
}

BERT_CONFIG = {
    'epoch': 15,
    'learning_rate': 0.0001,
    'weight_decay': 0.4,
    'beta': 1,
}

ELECTRA_CONFIG = {
    'epoch': 15,
    'learning_rate': 0.00005,
    'weight_decay': 0.5,
    'beta': 1,
}

MODELS_CONFIG = {
    'bert-base-uncased': BERT_CONFIG,
    'roberta-base': ROBERTA_CONFIG,
    'google/electra-base-generator': ELECTRA_CONFIG,
    'bert-large-uncased': BERT_CONFIG,
    'roberta-large': ROBERTA_CONFIG,
    'google/electra-large-generator': ELECTRA_CONFIG,
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


def get_models_predictions(models, train_dataset, validation_dataset,
                           test_dataset_in_dist, test_dataset_out_dist, seed, augment=False,
                           del_p=0):
    """
    Gets the statistics on a prediction of a single model. The model is trained over multiple seeds and the
    results are averaged
    """
    prediction_results = {}



    for model_name in models:
        scores_in_dist = []
        scores_out_dist = []
        model_config = MODELS_CONFIG[model_name]
        model_config['model_name'] = model_name
        model_config['augment'] = augment
        model_config['del_p'] = del_p

        for i in range(seed):
            output, trainer = fine_tune_model(
                model_name, SPECIAL_TOKENS, train_dataset, validation_dataset,
                "no", use_augment=augment, del_p=del_p, config=model_config,
                seed=i)

            pros_test_in_dist = test_dataset_in_dist.map(
                lambda x: tokenize_func(x, trainer.tokenizer), batched=True)
            pros_test_in_dist = pros_test_in_dist.remove_columns(
                ['description'])

            pros_test_out_dist = test_dataset_out_dist.map(
                lambda x: tokenize_func(x, trainer.tokenizer), batched=True)
            pros_test_out_dist = pros_test_out_dist.remove_columns(
                ['description'])

            scores_in_dist.append(trainer.evaluate(pros_test_in_dist,
                                                   metric_key_prefix='test'))

            scores_out_dist.append(trainer.evaluate(pros_test_out_dist,
                                                    metric_key_prefix='test'))

            # del trainer

        prediction_results[model_name] = {'in_dist': get_model_avg_score_and_std(scores_in_dist),
                                          'out_dist': get_model_avg_score_and_std(scores_out_dist)}

    return prediction_results


def evaluate_models(models, train_dataset, validation_dataset, test_dataset_in_dist, test_dataset_out_dist,
                    args):
    results = {}

    if not args.augment:
        # Run without augmentation
        results['Without Augmentation'] = get_models_predictions(
            models, train_dataset, validation_dataset, test_dataset_in_dist, test_dataset_out_dist,
            args.seed
        )

    # Run with augmentation
    else:
        results['With Augmentation'] = {}
        for del_p in args.del_p_list:
            results['With Augmentation'][f'{del_p}'] = get_models_predictions(
                models, train_dataset, validation_dataset, test_dataset_in_dist, test_dataset_out_dist,
                args.seed, args.augment, del_p
            )

    return results


def write_results_to_file(results, file_name):
    with open(f'results/{file_name}', 'w') as f:
        f.write(json.dumps(results))


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--seed", default=3, type=int,
                      help="seed for fine tuning")
    args.add_argument("--augment", default=False, type=lambda x: x == "True",
                      help="use augmented data")
    args.add_argument('-p', "--del_p_list", nargs='+',
                      type=float,
                      help="list of probabilities to delete words")
    args.add_argument("--out_name", default="results", type=str, help="name of output file")

    args = args.parse_args()
    if not args.augment:
        train_dataset = convert_data('train_data.csv')
    else:
        train_dataset = pd.read_csv('../csvs/train_data_with_aug.csv')
        train_dataset = Dataset.from_pandas(train_dataset)

    validation_dataset = convert_data('validation_data.csv')
    test_dataset_in_dist = convert_data('test_data_in_dist.csv')
    test_dataset_out_dist = convert_data('test_data_out_dist.csv')

    results = evaluate_models(
        MODELS,
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

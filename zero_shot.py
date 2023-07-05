from transformers import AutoTokenizer, BertForMaskedLM, RobertaForMaskedLM, ElectraForMaskedLM
import torch
from preprocessing import build_df_from_data, format_dataframe
from evaluate import load
from math import ceil

SLICE_DATA_FOR_DEBUG = 3
X_INDX_IN_TUPLE = 0
Y_INDX_IN_TUPLE = 1
N_EXAMPLE_SAMPLES = 3
BATCH_SIZE = 50
# FORMAT_STR = "The price of this house is {mask_format}.{mask_format}$ million."
FORMAT_STR = "The price of this house is ({mask_format}.{mask_format})$ million."

MODELS = [
    # model name, model object, mask format
    ("bert-base-uncased", BertForMaskedLM, "[MASK]"),
    ("roberta-base", RobertaForMaskedLM, "<mask>"),
    ("google/electra-base-generator", ElectraForMaskedLM, "[MASK]")
]


def predictions_to_numbers(predicted_words):
    """
    gets the words that the model completed a list and converts them to list of
    numbers representing the prediction in millions
    :param predicted_words: list of completed words by the model, length of num_samples *2
    :return:
    """
    num_predictions = []
    for i in range(0, len(predicted_words), 2):
        num_predictions.append(float(f"{predicted_words[i]}.{predicted_words[i + 1]}".replace(" ", "")))
    return num_predictions


def extract_data(lst, index):
    """
    extract only the samples or only the labels from a tuple of both
    :param lst: list of tuples in format of (sample,label)
    :param index: 0 for extracting samples list, 1 for extracting labels list
    :return: samples list or labels list
    """
    return [item[index] for item in lst]


def get_numerical_tokens(tokenizer):
    num_list = list(map(str, range(101)))
    s = ' '.join(num_list)
    numerical_ids = tokenizer.encode(s, add_special_tokens=False)
    return numerical_ids


def save_to_buffer(model_name, losses, examples, buffer):
    samples = examples[0]
    predictions = examples[1]
    labels = examples[2]

    buffer += "-" * 112 + "\n"  # Seperator
    buffer += f"Model name: {model_name}\n"
    buffer += f"R2 Squard loss: {losses[0]}\n"
    buffer += f"MSE loss: {losses[1]}\n"
    buffer += f"MAE loss: {losses[2]}\n\n"

    for i in range(N_EXAMPLE_SAMPLES):
        buffer += f"Sample {i}:\n{samples[i]}\n\n"
        buffer += f"Price prediction:\n{predictions[i]}\n"
        buffer += f"True price:\n{labels[i]}\n"
        buffer += "-" * 112 + "\n"
    return buffer


def save_results(buffer):
    with open('results_zero_shot.txt', 'w') as f:
        f.write(buffer)


def evaluate(predictions, labels):
    r2_metric = load("r_squared")
    mse_metric = load("mse")
    mae_metric = load("mae")
    r_squared = r2_metric.compute(predictions=predictions, references=labels)
    mse = mse_metric.compute(predictions=predictions, references=labels)['mse']
    mae = mae_metric.compute(predictions=predictions, references=labels)['mae']

    return [r_squared, mse, mae]


def zero_shot(model_name, model_for_lm, mask_format, train_data):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_for_lm.from_pretrained(model_name)

    max_length_tokenizer = tokenizer.max_model_input_sizes[model_name]

    masked_data_sentences = extract_data(train_data, X_INDX_IN_TUPLE)
    true_data_completion = extract_data(train_data, Y_INDX_IN_TUPLE)

    added_format = FORMAT_STR.format(mask_format=mask_format)

    overview_truncation_len = max_length_tokenizer - len(tokenizer.encode(added_format))
    truncated_masked_data_sentences = tokenizer.batch_decode(
        tokenizer(masked_data_sentences, max_length=overview_truncation_len, truncation=True)['input_ids'])

    truncated_formatted_sentences = [s + added_format for s in truncated_masked_data_sentences]
    inputs = tokenizer(truncated_formatted_sentences, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        # logits.shape: [number of sentences, tokens (maximum in all sentences), probs]
        logits = model(**inputs).logits

    max_val = torch.max(logits)
    mask_token_indexs = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
    numerical_ids = get_numerical_tokens(tokenizer)

    # maximazing only the numerical token to only them will get picked
    for indx in numerical_ids:
        logits[mask_token_indexs[0], mask_token_indexs[1], indx] = logits[mask_token_indexs[0],
        mask_token_indexs[1], indx] + max_val

    predicted_token_id = logits[mask_token_indexs[0], mask_token_indexs[1]].argmax(axis=-1)
    predicted_words = tokenizer.batch_decode(predicted_token_id)
    predicted_prices = predictions_to_numbers(predicted_words)

    losses = evaluate(predicted_prices, true_data_completion)

    return losses, truncated_formatted_sentences, true_data_completion, predicted_prices


def main():
    # Uncomment if there are not csv files saved
    # df_train, df_test = build_df_from_data()

    buffer = ""
    for model_name, model_for_lm, mask_format in MODELS:
        train_data = format_dataframe("train_data.csv", "{overview}")
        avg_losses = [0, 0, 0]
        examples = []
        for i in range(0, len(train_data), BATCH_SIZE):
            losses, truncated_formatted_sentences, true_data_completion, predicted_prices = zero_shot(
                model_name, model_for_lm, mask_format, train_data[i:i + BATCH_SIZE])
            if i == 0:
                examples = [truncated_formatted_sentences[:N_EXAMPLE_SAMPLES],
                            predicted_prices[:N_EXAMPLE_SAMPLES],
                            true_data_completion[:N_EXAMPLE_SAMPLES]]
            avg_losses[0] += losses[0]
            avg_losses[1] += losses[1]
            avg_losses[2] += losses[2]
        num_batches = ceil(len(train_data) / BATCH_SIZE)
        buffer = save_to_buffer(model_name, [avg / num_batches for avg in avg_losses], examples, buffer)
    save_results(buffer)


if __name__ == '__main__':
    main()

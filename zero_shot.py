from transformers import AutoTokenizer, BertForMaskedLM, RobertaForMaskedLM, ElectraForMaskedLM
import torch
from sklearn.metrics import mean_squared_error
from preprocessing import build_df_from_data, format_dataframe

SLICE_DATA_FOR_DEBUG = 10
X_INDX_IN_TUPLE = 0
Y_INDX_IN_TUPLE = 1

MODELS = [
    # model name, model object, mask format
    ("bert-base-uncased", BertForMaskedLM, "[MASK]"),
    # ("roberta-base", RobertaForMaskedLM, "<mask>"),
    # ("google/electra-base-generator",ElectraForMaskedLM,"[MASK]")
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
        num_predictions.append(float(f"{predicted_words[i]}.{predicted_words[i + 1]}"))
    return num_predictions


def extract_data(lst, index):
    """
    extract only the samples or only the labels from a tuple of both
    :param lst: list of tuples in format of (sample,label)
    :param index: 0 for extracting samples list, 1 for extracting labels list
    :return: samples list or labels list
    """
    return [item[index] for item in lst]


def zero_shot(tokenizer, model, masked_data_sentences, true_data_completion):
    inputs = tokenizer(masked_data_sentences, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        # logits.shape: [number of sentances, tokens (maximum in all sentances), probs]
        logits = model(**inputs).logits

    mask_token_indexs = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)

    predicted_token_id = logits[0, mask_token_indexs[1]].argmax(axis=-1)
    predicted_words = tokenizer.batch_decode(predicted_token_id)

    print(predicted_words)
    print(predictions_to_numbers(predicted_words))
    print(f"MSE={mean_squared_error(predictions_to_numbers(predicted_words), true_data_completion)}")


if __name__ == '__main__':
    # Uncomment if there are not csv files saved
    # df_train, df_test = build_df_from_data()

    for model_name, model_for_lm, mask_format in MODELS:
        format_str = "{overview}" + f" The price of this house is {mask_format}.{mask_format}$ million."
        train_data = format_dataframe("train_data.csv", format_str)[:SLICE_DATA_FOR_DEBUG]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = model_for_lm.from_pretrained(model_name)

        masked_data_sentences = extract_data(train_data, X_INDX_IN_TUPLE)
        true_data_completion = extract_data(train_data, Y_INDX_IN_TUPLE)

        zero_shot(tokenizer, model, masked_data_sentences, true_data_completion)

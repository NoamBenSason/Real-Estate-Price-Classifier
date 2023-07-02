from typing import Union

import numpy as np
import pandas as pd
import glob
import json
from tqdm import tqdm
import fire
from sklearn.model_selection import train_test_split


def txt_file_to_series(txt_file_path):
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        house_data = json.load(f)
    house_data['address'] = house_data['address'].replace(u'\xa0', u' ')
    series = pd.Series(house_data)
    return series


def build_raw_df(base_data_dir="data"):
    rows = []
    for path in tqdm(list(glob.glob(f"{base_data_dir}/**/*.txt", recursive=True)),
                     desc="Loading data from txt files"):
        rows.append(txt_file_to_series(path))
    df = pd.DataFrame(rows)
    return df


def parse_numeric_cols(df):
    for col in ['bed', 'bath', 'sqft']:
        df[col] = df[col].apply(lambda x: np.nan if x == '--' or not bool(x) else int(x.replace(",", "")))
    df['price'] = df['price'].apply(lambda x: np.nan if x == '--' or not bool(x)
    else int(x.replace(",", "").replace("$", "")) / 1e6)
    return df


def parse_address(df):
    def parse_row(row):
        street_raw, city_raw, state_raw = row['address'].split(", ")

        if 'UNIT' in street_raw:
            street_raw = street_raw[:street_raw.index('UNIT') - 1]
        elif 'APT' in street_raw:
            street_raw = street_raw[:street_raw.index('APT') - 1]
        elif 'SPACE' in street_raw:
            street_raw = street_raw[:street_raw.index('SPACE') - 1]
        elif '#' in street_raw:
            street_raw = street_raw[:street_raw.index('#') - 1]
        street = street_raw[street_raw.index(' ') + 1:]

        city = city_raw

        state = state_raw[:2]

        return street, city, state

    new_df = df.copy()
    split_data = pd.DataFrame(list(df.apply(parse_row, axis=1)), columns=['street', 'city', 'state'])
    new_df = new_df.drop(columns=['address'])
    new_df = pd.concat([new_df, split_data], axis=1)
    return new_df


def split_train_test_by_city(df, random_state: int = 42, train_size: float = 0.85):
    train_sets = []
    test_sets = []
    for city in df['city'].unique():
        samples_in_city = df[df['city'] == city]
        if len(samples_in_city) == 1:
            train_sets.append(samples_in_city)
            continue
        train_city, test_city = train_test_split(samples_in_city,
                                                 train_size=train_size,
                                                 random_state=random_state)
        train_sets.append(train_city)
        test_sets.append(test_city)
    all_train = pd.concat(train_sets)
    all_test = pd.concat(test_sets)
    return all_train, all_test


def reorder_columns(df):
    order = [
        'zpid',
        'bed',
        'bath',
        'sqft',
        'street',
        'city',
        'state',
        'overview',
        'price'
    ]
    return df[order]


def build_df_from_data(train_path: str = "train_data.csv",
                       test_path: str = "test_data.csv",
                       base_data_dir: str = "data"):
    df = build_raw_df(base_data_dir)
    df = parse_numeric_cols(df)
    df = parse_address(df)
    df_train, df_test = split_train_test_by_city(df)

    df_train = reorder_columns(df_train)
    df_test = reorder_columns(df_test)

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    return df_train, df_test


def format_dataframe(df_or_df_path: Union[pd.DataFrame, str], format_str: str,
                     nan_value: str = "NaN"):
    """
    Accepts either a path to dataframe or a dataframe.
    :param df_or_df_path: The dataframe or the path to it
    :param format_str: The format string of the data. To enter a variable in the format, enter it with
                       curly braces - {}
                       For example, to inject the overview into the string, write {overview} in the relevant
                       place in the string.
                       Available labels:
                       - zpid
                       - bed
                       - bath
                       - sqft
                       - street
                       - city
                       - state
                       - overview
                       - price
    :param nan_value: The value to assign to NaN elements
    :return: A list of tuples of length 2. The first element is the formatted string, the second element is
             the price of the sample, in million dollars.
    """
    if isinstance(df_or_df_path, str):
        df_or_df_path = pd.read_csv(df_or_df_path)
    df = df_or_df_path
    out = []
    for _, row in tqdm(list(df.iterrows()),
                       desc="Formatting data"):
        elements_map = dict(row)
        for col in ['bed', 'bath', 'sqft']:
            elements_map[col] = int(elements_map[col]) if not np.isnan(elements_map[col]) else nan_value
        current_str = format_str.format(**elements_map)
        out.append((current_str, row['price']))
    return out


if __name__ == '__main__':
    # train, test = build_df_from_data()
    # str_format = "[bd]{bed}[br]{bath}[QF]{sqft}[OV]{overview}[SEP]The Price of the apartment is [MASK] million US dollars"
    # x = format_dataframe(train, str_format)
    fire.Fire(build_df_from_data)

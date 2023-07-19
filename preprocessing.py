import os.path
from typing import Union

import numpy as np
import pandas as pd
import glob
import json

import requests
from tqdm import tqdm
import fire
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from sklearn.model_selection import train_test_split

from data_augmentation import augment_data


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


def parse_images(df):
    df['images'] = df['images'].apply(lambda x: " ".join([y for y in x if y.endswith('jpg')]))
    # df = df.rename(columns={'images': 'image'})
    return df


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
        'images',
        'price'
    ]
    return df[order]


def build_df_from_data(train_path: str = "train_data.csv",
                       val_path: str = "validation_data.csv",
                       base_data_dir: str = "data"):

    df = build_raw_df(base_data_dir)
    df = parse_images(df)
    df = parse_numeric_cols(df)
    df = parse_address(df)
    df_train, df_test = split_train_test_by_city(df)

    df_train = reorder_columns(df_train)
    df_test = reorder_columns(df_test)

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(val_path, index=False)
    return df_train, df_test


def download_url(args):
    input_path, out_path, force_download = args[0], args[1], args[2]
    if not os.path.isfile(out_path) or force_download:
        # wget.download(input_path, out_path)
        img_data = requests.get(input_path).content
        with open(out_path, 'wb') as handler:
            handler.write(img_data)
    return out_path


def download_parallel(args):
    cpus = cpu_count()
    results = ThreadPool(cpus - 1).imap_unordered(download_url, args)


def format_dataframe(df_or_df_path: Union[pd.DataFrame, str], format_str: str,
                     nan_value: str = "NaN", with_image: bool = False,
                     image_force_download: bool = False,
                     n_images: int = 1,
                     image_download_dir:str = "images"):
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
                       - address
                       - overview
                       - price
    :param nan_value: The value to assign to NaN elements
    :param with_image: If true, the returned tuples will have 3 elements: formatted string, local path to
                       an image of the house, and the price of the sample. Otherwise, the returned tuples
                       will be as described below.
    :return: A list of tuples of length 2. The first element is the formatted string, the second element is
             the price of the sample, in million dollars.
    """

    if isinstance(df_or_df_path, str):
        df_or_df_path = pd.read_csv(df_or_df_path)
    df = df_or_df_path
    out = []
    if with_image:
        os.makedirs(image_download_dir, exist_ok=True)

    for _, row in tqdm(list(df.iterrows()),
                       desc="Formatting data"):
        elements_map = dict(row)
        for col in ['bed', 'bath', 'sqft']:
            elements_map[col] = int(elements_map[col]) if not np.isnan(elements_map[col]) else nan_value
        elements_map['address'] = f"{row['street']}, {row['city']},{row['state']}"
        current_str = format_str.format(**elements_map)

        if with_image:
            input_paths = row['images'].split()[:n_images]
            if len(input_paths) == 0:
                continue
            out_paths = [os.path.join(image_download_dir, f"{row['zpid']}_{i}.jpg") for i in range(len(input_paths))]
            for in_path, out_path in zip(input_paths, out_paths):
                download_url((in_path, out_path, image_force_download))
            out.append((current_str, out_paths, row['price']))
        else:
            out.append((current_str, row['price']))
    return out


def augment_dataframe(df: pd.DataFrame, random_state: int = 42):
    return augment_data(df, random_state)


if __name__ == '__main__':
    fire.Fire(build_df_from_data)
    str_format = "[bd]{bed}[br]{bath}[QF]{sqft}[OV]{overview}[SEP]The Price of the apartment is [MASK] million US dollars"
    train, test = build_df_from_data()
    x = format_dataframe(train, str_format, with_image=True, image_download_dir="images")
    a = 1
    # augmented_df = augment_dataframe(train)
    # augmented_df.to_csv("train_data_with_aug.csv", index=False)

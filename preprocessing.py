import numpy as np
import pandas as pd
import glob
import json
from tqdm import tqdm
import fire


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


def build_df_from_data(save_path: str = "california_data.csv", base_data_dir: str = "data"):
    df = build_raw_df(base_data_dir)
    df = parse_numeric_cols(df)
    df = parse_address(df)
    df = reorder_columns(df)
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    fire.Fire(build_df_from_data)

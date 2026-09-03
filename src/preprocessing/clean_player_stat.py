from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_FILE = BASE_DIR / 'data/raw/players_stat.csv'
CLEAN_FILE = BASE_DIR / 'data/processed/players_stat_cleaned.csv'

ROLE_MAPPING = {
    'jett': 'Duelist', 'raze': 'Duelist', 'phoenix': 'Duelist', 'reyna': 'Duelist', 'yoru': 'Duelist', 'neon': 'Duelist', 'iso': 'Duelist', 'waylay': 'Duelist',
    'omen': 'Controller', 'viper': 'Controller', 'brimstone': 'Controller', 'astra': 'Controller', 'harbor': 'Controller', 'clove': 'Controller',
    'sova': 'Initiator', 'breach': 'Initiator', 'skye': 'Initiator', 'kayo': 'Initiator', 'fade': 'Initiator', 'gekko': 'Initiator', 'tejo': 'Initiator', 'veto': 'Initiator',
    'killjoy': 'Sentinel', 'cypher': 'Sentinel', 'sage': 'Sentinel', 'chamber': 'Sentinel', 'deadlock': 'Sentinel', 'vyse': 'Sentinel'
}

PERCENT_COLUMNS = ['kast_t', 'kast_ct', 'kast_all', 'hsp_t', 'hsp_ct', 'hsp_all']
NUMERIC_COLUMNS = ['rating2_t', 'rating2_ct', 'rating2_all', 'acs_t', 'acs_ct', 'acs_all', 'adr_t', 'adr_ct', 'adr_all']


def load_data(file_path=RAW_FILE):
    return pd.read_csv(file_path, low_memory=False)


def add_roles(df):
    df['agent'] = df['agent'].astype('string').str.lower().str.strip()
    df['role'] = df['agent'].map(ROLE_MAPPING)
    return df


def drop_invalid_rows(df):
    required_columns = ['fb_all', 'fd_all', 'team', 'name', 'agent', 'kills_all', 'role']
    return df.dropna(subset=required_columns).copy()


def convert_numeric_columns(df):
    for column in PERCENT_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column].astype('string').str.replace('%', '', regex=False), errors='coerce').astype('float64')
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce').astype('float64')
    return df


def fill_missing_by_role(df):
    columns_to_fill = PERCENT_COLUMNS + NUMERIC_COLUMNS
    for column in columns_to_fill:
        if column in df.columns:
            df[column] = df[column].fillna(df.groupby('role')[column].transform('mean'))
            df[column] = df[column].fillna(df[column].mean())
    return df


def clean_data(df):
    df = add_roles(df)
    df = drop_invalid_rows(df)
    df = convert_numeric_columns(df)
    return fill_missing_by_role(df)


def save_data(df, file_path=CLEAN_FILE):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f'Đã làm sạch xong! Dữ liệu được lưu tại: {file_path}')


def main():
    print('Đang đọc dữ liệu...')
    save_data(clean_data(load_data()))


if __name__ == '__main__':
    main()
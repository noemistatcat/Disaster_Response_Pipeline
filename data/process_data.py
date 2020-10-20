import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the datasets
    :param messages_filepath: filepath where messages are saved
    :param categories_filepath: filepath where category labels are saved
    :return: merged pandas dataframe
    """
    # load messages dataset
    messages = pd.read_csv(str(messages_filepath))
    # load categories dataset
    categories = pd.read_csv(str(categories_filepath))
    # merge datasets
    df = messages.merge(categories, on="id")
    return df


def clean_data(df):
    """
    Performs cleaning and transformations to the raw dataset
    :param df: Merged dataframe from load_data() step
    :return: Cleaned pandas dataframe
    """
    # create a dataframe of the 36 individual category columns
    cat = df.categories.str.split(pat=";", expand=True)
    # select the first row of the categories dataframe
    row = np.array(cat[:1])

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [row[0][n][:-2] for n in range(0, len(row[0]))]
    # rename the columns of `categories`
    cat.columns = category_colnames
    for column in cat:
    # set each value to be the last character of the string
        cat[column] = cat[column].str.slice(start=-1)

    # convert column from string to numeric
        cat[column] = pd.to_numeric(cat[column])

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, cat], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    # Clean df.related (it has "2" value which is an error)
    df['related'] = np.where(df['related'] == 2, 1, 0)
    return df


def save_data(df, database_filename):
    """
    Saves the pandas dataframe to a SQLite database
    :param df: Cleaned pandas dataframe
    :param database_filename: Filename for the database
    :return: None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
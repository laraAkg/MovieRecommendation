import pandas as pd

class DataLoader:
    @staticmethod
    def load_data():
        """
        Loads and processes movie-related data from CSV files.
        This function reads three CSV files: 'movies_metadata.csv', 'credits.csv', 
        and 'keywords.csv', located in the 'data/' directory. It performs the 
        following operations:
        - Reads the CSV files into pandas DataFrames.
        - Cleans and casts the 'id' column in each DataFrame to ensure it is an integer.
        - Merges the three DataFrames on the 'id' column.
        Returns:
            pandas.DataFrame: A merged DataFrame containing data from all three 
            input files.
        """
        movies_df = pd.read_csv('data/movies_metadata.csv', low_memory=False)
        credits_df = pd.read_csv('data/credits.csv')
        keywords_df = pd.read_csv('data/keywords.csv')
        movies_df = movies_df[movies_df['id'].apply(lambda x: str(x).isdigit())]
        movies_df['id'] = movies_df['id'].astype(int)
        credits_df['id'] = credits_df['id'].astype(int)
        keywords_df['id'] = keywords_df['id'].astype(int)
        df = movies_df.merge(credits_df, on='id').merge(keywords_df, on='id')
        return df
import pandas as pd

class DataLoader:
    @staticmethod
    def load_data():
        """Load and merge data from CSV files."""
        movies_df = pd.read_csv('data/movies_metadata.csv', low_memory=False)
        credits_df = pd.read_csv('data/credits.csv')
        keywords_df = pd.read_csv('data/keywords.csv')

        # Clean and cast IDs
        movies_df = movies_df[movies_df['id'].apply(lambda x: str(x).isdigit())]
        movies_df['id'] = movies_df['id'].astype(int)
        credits_df['id'] = credits_df['id'].astype(int)
        keywords_df['id'] = keywords_df['id'].astype(int)

        # Merge DataFrames
        df = movies_df.merge(credits_df, on='id').merge(keywords_df, on='id')
        return df
import pandas as pd
import ast

class DataCleaner:
    @staticmethod
    def parse_column(val):
        """Parse a JSON-like string into a Python object."""
        try:
            return ast.literal_eval(val) if pd.notnull(val) else []
        except:
            return []

    @staticmethod
    def get_names(obj_list, key='name', max_items=5):
        """Extract names from a list of dictionaries."""
        return ', '.join([obj[key] for obj in obj_list[:max_items] if key in obj])

    @staticmethod
    def get_director(crew_list):
        """Extract the director's name from the crew list."""
        for member in crew_list:
            if member.get('job') == 'Director':
                return member.get('name', '')
        return ''

    @staticmethod
    def combine_features(row):
        """Combine multiple features into a single string for vectorization."""
        overview = str(row['overview']) if pd.notnull(row['overview']) else ''
        genres = DataCleaner.get_names(row['genres']) * 2
        keywords = DataCleaner.get_names(row['keywords']) * 2
        cast = DataCleaner.get_names(row['cast'])
        director = DataCleaner.get_director(row['crew']) * 2
        tagline = str(row['tagline']) if pd.notnull(row['tagline']) else ''
        return ' '.join([overview, genres, keywords, cast, director, tagline]).strip()

    @staticmethod
    def clean_data(df):
        """Clean and process the DataFrame."""
        # Parse List-of-Dict columns
        for col in ['genres', 'keywords', 'cast', 'crew', 'production_countries', 'production_companies']:
            df[col] = df[col].apply(DataCleaner.parse_column)

        # Generate combined features
        df['combined_features'] = df.apply(DataCleaner.combine_features, axis=1)

        # Clean data
        df.dropna(subset=['title', 'combined_features'], inplace=True)
        df = df[df['combined_features'].str.strip() != '']
        df.drop_duplicates(subset='title', inplace=True)

        # Normalize titles and clean additional fields
        df['title'] = df['title'].str.strip()
        df['production_countries'] = df['production_countries'].apply(lambda x: DataCleaner.get_names(x))
        df['production_companies'] = df['production_companies'].apply(lambda x: DataCleaner.get_names(x))
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)

        # Remove unwanted columns
        columns_to_drop = [
            'adult', 'belongs_to_collection', 'budget', 'video', 'status',
            'spoken_languages', 'poster_path', 'popularity', 'revenue', 'imdb_id', 'homepage'
        ]
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

        return df
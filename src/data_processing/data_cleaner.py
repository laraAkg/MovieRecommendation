import pandas as pd
import ast

class DataCleaner:
    @staticmethod
    def parse_column(val):
        """
        Parses a given value and attempts to evaluate it as a Python literal.
        If the value is not null, it tries to evaluate it using `ast.literal_eval`.
        If the evaluation fails or the value is null, it returns an empty list.
        Args:
            val (any): The value to be parsed. Typically expected to be a string.
        Returns:
            list: The evaluated Python literal if successful, or an empty list otherwise.
        """
        
        try:
            return ast.literal_eval(val) if pd.notnull(val) else []
        except:
            return []

    @staticmethod
    def get_names(obj_list, key='name', max_items=5):
        """
        Extracts and concatenates the values associated with a specified key from a list of dictionaries.
        Args:
            obj_list (list): A list of dictionaries to extract values from.
            key (str, optional): The key to look for in each dictionary. Defaults to 'name'.
            max_items (int, optional): The maximum number of items to include in the result. Defaults to 5.
        Returns:
            str: A comma-separated string of values corresponding to the specified key from the dictionaries.
                 Only includes dictionaries where the key is present.
        """
        
        return ', '.join([obj[key] for obj in obj_list[:max_items] if key in obj])

    @staticmethod
    def get_director(crew_list):
        """
        Extracts the name of the director from a list of crew members.
        Args:
            crew_list (list): A list of dictionaries, where each dictionary represents 
                              a crew member and contains details such as their job and name.
        Returns:
            str: The name of the director if found in the crew list, otherwise an empty string.
        """
        
        for member in crew_list:
            if member.get('job') == 'Director':
                return member.get('name', '')
        return ''

    @staticmethod
    def clean_data(df):
        """
        Cleans and preprocesses a DataFrame containing movie data.
        This function performs the following operations:
        1. Parses specific columns containing list-of-dictionaries into a usable format.
        3. Removes rows with missing or empty 'title' .
        4. Drops duplicate rows based on the 'title' column.
        5. Normalizes and cleans specific fields such as 'title', 'production_countries', and 'production_companies'.
        6. Converts the 'release_date' column to a standardized date format (YYYY-MM-DD).
        7. Converts the 'vote_average' column to numeric, filling invalid values with 0.
        8. Removes unnecessary columns from the DataFrame.
        Args:
            df (pd.DataFrame): The input DataFrame containing movie data.
        Returns:
            pd.DataFrame: The cleaned and preprocessed DataFrame.
        """
        # Parse List-of-Dict columns
        for col in ['genres', 'keywords', 'cast', 'crew', 'production_countries', 'production_companies']:
            df[col] = df[col].apply(DataCleaner.parse_column)

        # Clean data
        df.dropna(subset=['title'], inplace=True)
        df = df.drop_duplicates(subset='title', inplace=False)
        
        # Normalize titles and clean additional fields
        df.loc[:, 'title'] = df['title'].str.strip()
        df.loc[:, 'production_countries'] = df['production_countries'].apply(lambda x: DataCleaner.get_names(x))
        df.loc[:, 'production_companies'] = df['production_companies'].apply(lambda x: DataCleaner.get_names(x))
        df.loc[:, 'release_date'] = pd.to_datetime(df['release_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df.loc[:, 'vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)

        # Remove unwanted columns
        columns_to_drop = [
            'adult', 'belongs_to_collection', 'budget', 'video', 'status',
            'spoken_languages', 'poster_path', 'popularity', 'revenue', 'imdb_id', 'homepage'
        ]
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

        return df
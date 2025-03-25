import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

class HybridRecommender:
    def __init__(self, data_path: str, tfidf_max_features: int = 5000):
        self.df = pd.read_csv(data_path)
        self.tfidf_max_features = tfidf_max_features
        self.feature_matrix = None
        self.indices = None
        print("Recommender initialisiert.")

        # Gewichte (kannst du anpassen)
        # Normalisiert auf 100%
        self.weights = {
            'combined': 0.35,   
            'genre': 0.5,     
            'cast': 0.1,      
            'cat': 0.025,     
            'num': 0.025      
        }




    def preprocess(self):
        # Missing Values behandeln
        self.df['duration_mins'] = self.df['duration'].str.extract(r'(\d+)').astype(float)
        self.df['duration_mins'] = self.df['duration_mins'].fillna(self.df['duration_mins'].median())
        self.df['description'] = self.df['description'].fillna('')
        self.df['cast'] = self.df['cast'].fillna('')
        self.df['listed_in'] = self.df['listed_in'].fillna('')
        self.df['country'] = self.df['country'].fillna('Country unknown')
        self.df['rating'] = self.df['rating'].fillna('Rating unknown')
        self.df['type'] = self.df['type'].fillna('Type unknown')
        self.df['release_year'] = self.df['release_year'].fillna(self.df['release_year'].median())
        self.df['director'] = self.df['director'].fillna('Director unknown')
        self.df['is_recent'] = (self.df['release_year'] >= 2020).astype(int)

        print("Preprocessing abgeschlossen.")

    def vectorize_and_combine(self):
        # Combined-Text: Beschreibung + Regisseur
        self.df['combined_text'] = (
            self.df['description'] + ' ' + self.df['director']
        )

        # TF-IDF auf Combined Text (description + director)
        tfidf_combined = TfidfVectorizer(stop_words='english', max_features=self.tfidf_max_features)
        combined_tfidf = tfidf_combined.fit_transform(self.df['combined_text'])

        # Genre-TFIDF
        tfidf_genre = TfidfVectorizer(tokenizer=lambda x: x.split(', '), lowercase=False)
        genre_tfidf = tfidf_genre.fit_transform(self.df['listed_in'])

        # Cast-TFIDF (separat!)
        tfidf_cast = TfidfVectorizer(tokenizer=lambda x: x.split(', '), lowercase=False)
        cast_tfidf = tfidf_cast.fit_transform(self.df['cast'])

        # One-Hot für type, country, rating
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_cat = ohe.fit_transform(self.df[['type', 'country', 'rating']])

        # Numerische Features
        scaler = StandardScaler()
        X_num = scaler.fit_transform(self.df[['release_year', 'duration_mins', 'is_recent']])

        # Sparse-Matrix kombinieren
        self.matrices = {
            'combined': combined_tfidf,
            'genre': genre_tfidf,
            'cast': cast_tfidf,
            'cat': csr_matrix(X_cat),
            'num': csr_matrix(X_num)
        }

        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()
        print("Feature-Matrizen erstellt und gewichtet kombinierbar!")

    def get_recommendations(self, title: str, n_recommendations: int = 5):
        idx = self.indices.get(title)
        if idx is None:
            return f"'{title}' wurde nicht im Dataset gefunden."

        # Similarities pro Block berechnen
        sims = {}
        for key, matrix in self.matrices.items():
            sims[key] = cosine_similarity(
                matrix[idx], matrix
            )[0]

        # Gewichte anwenden
        combined_sim = sum(
            sims[key] * self.weights[key] for key in sims
        )

        sim_indices = combined_sim.argsort()[::-1][1:n_recommendations+1]
        return self.df[['title', 'type', 'country', 'rating', 'listed_in', 'director', 'cast', 'description']].iloc[sim_indices]

    def explain_recommendation(self, base_title: str, recommended_title: str):
        base = self.df[self.df['title'] == base_title].iloc[0]
        rec = self.df[self.df['title'] == recommended_title].iloc[0]
        explanation = {}

        base_genres = set(base['listed_in'].split(', '))
        rec_genres = set(rec['listed_in'].split(', '))
        overlap = base_genres.intersection(rec_genres)
        explanation['Genre Overlap'] = f"{len(overlap)} gemeinsame Genres ({', '.join(overlap)})" if overlap else "keine"

        # Cast Overlap neu hinzufügen
        base_cast = set(base['cast'].split(', '))
        rec_cast = set(rec['cast'].split(', '))
        cast_overlap = base_cast.intersection(rec_cast)
        explanation['Cast Overlap'] = f"{len(cast_overlap)} gleiche Schauspieler ({', '.join(cast_overlap)})" if cast_overlap else "keine"

        explanation['Rating Match'] = "Ja" if base['rating'] == rec['rating'] else "Nein"
        explanation['Country Match'] = "Ja" if base['country'] == rec['country'] else "Nein"
        explanation['Director Match'] = "Ja" if base['director'] == rec['director'] and base['director'] != "Unknown" else "Nein"
        explanation['Release Year Abstand'] = abs(base['release_year'] - rec['release_year'])

        # Finaler Ähnlichkeitswert
        base_idx = base.name
        rec_idx = rec.name
        final_score = sum(
            cosine_similarity(
                self.matrices[key][base_idx], self.matrices[key][rec_idx]
            )[0][0] * self.weights[key]
            for key in self.matrices
        )
        explanation['Gesamt-Ähnlichkeit'] = f"{final_score:.2f}"

        return explanation

if __name__ == "__main__":
    recommender = HybridRecommender('data/netflix_titles_clean.csv')
    recommender.preprocess()
    recommender.vectorize_and_combine()
    print(recommender.get_recommendations('Breaking Bad'))
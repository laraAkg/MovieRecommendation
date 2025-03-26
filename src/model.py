import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import logging
import joblib
from data_processing.database_utils import get_mongo_collection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridRecommender:
    """
    Ein hybrides Empfehlungssystem, das mehrere Features kombiniert,
    um personalisierte Empfehlungen basierend auf Kosinus-Ähnlichkeit zu generieren.
    """
    def __init__(self, tfidf_max_features: int = 5000):
        # Daten aus MongoDB laden
        collection = get_mongo_collection('netflix_db', 'netflix_content')
        logging.info(f"Number of documents in collection: {collection.count_documents({})}")
        self.df = pd.DataFrame(list(collection.find()))

        # Sicherstellen, dass alle erforderlichen Spalten vorhanden sind.
        required_columns = [
            'title', 'description', 'director', 'cast', 'listed_in', 
            'type', 'country', 'rating', 'release_year', 
            'duration_mins', 'duration_seasons'
        ]
        for col in required_columns:
            if col not in self.df.columns:
                # Für numerische Spalten 0, ansonsten None verwenden.
                self.df[col] = 0 if col in ['duration_mins', 'duration_seasons'] else None

        self.tfidf_max_features = tfidf_max_features
        self.matrices = {}
        self.indices = None
        self.weights = {
            'combined': 0.35,
            'genre': 0.5,
            'cast': 0.1,
            'cat': 0.025,
            'num': 0.025
        }
        logging.info("Recommender initialized.")

    def vectorize_and_combine(self):
        """
        Vektorisiert Text- und kategorische Features, skaliert numerische Features
        und kombiniert sie in spärlichen Matrizen.
        """
        def clean_unknown(value, unknown_terms=["unknown", "unknown director", "unknown cast"]):
            """
            Gibt einen leeren String zurück, falls der Wert None ist oder ein "unknown"-Term enthält.
            """
            if not isinstance(value, str) or not value:
                return ""
            if any(term in value.lower() for term in unknown_terms):
                return ""
            return value

        # Bereinigte Versionen für Director und Cast erstellen.
        self.df['director_clean'] = self.df['director'].apply(lambda x: clean_unknown(x))
        self.df['cast_clean'] = self.df['cast'].apply(lambda x: clean_unknown(x))
        
        # Kombinierter Text aus Beschreibung und bereinigtem Regisseur.
        # Falls 'description' None ist, wird ein leerer String verwendet.
        self.df['combined_text'] = self.df['description'].fillna('') + ' ' + self.df['director_clean']

        # TF-IDF Vektorisierung für Textfeatures.
        tfidf_combined = TfidfVectorizer(stop_words='english', max_features=self.tfidf_max_features)
        tfidf_genre = TfidfVectorizer(tokenizer=lambda x: x.split(', '), lowercase=False)
        tfidf_cast = TfidfVectorizer(tokenizer=lambda x: x.split(', '), lowercase=False)

        self.matrices = {
            'combined': tfidf_combined.fit_transform(self.df['combined_text']),
            # Falls listed_in None ist, mit leerem String auffüllen.
            'genre': tfidf_genre.fit_transform(self.df['listed_in'].fillna('')),
            'cast': tfidf_cast.fit_transform(self.df['cast_clean']),
            'cat': csr_matrix(
                OneHotEncoder(handle_unknown='ignore', sparse_output=True)
                .fit_transform(self.df[['type', 'country', 'rating']].fillna(''))
            ),
            # Für numerische Features werden fehlende Werte mit 0 ersetzt.
            'num': csr_matrix(
                StandardScaler().fit_transform(self.df[['release_year', 'duration_mins']].fillna(0))
            )
        }

        # Indizes für Titel erstellen.
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()
        logging.info("Feature-Matrizen erstellt und bereit für die gewichtete Kombination.")

    def get_recommendations(self, title: str, n_recommendations: int = 6):
        """
        Gibt Empfehlungen basierend auf einem Titel zurück.
        """
        idx = self.indices.get(title)
        if idx is None:
            return []

        # Ähnlichkeiten für alle Matrizen berechnen.
        sims = {key: cosine_similarity(matrix[idx], matrix)[0] for key, matrix in self.matrices.items()}
        # Gewichtete Kombination der Ähnlichkeiten.
        combined_sim = sum(sims[key] * self.weights[key] for key in sims)
        # Top-N-Empfehlungen auswählen.
        sim_indices = combined_sim.argsort()[::-1][1:n_recommendations + 1]
        recommendations = []
        for i in sim_indices:
            recommendations.append({
                "title": self.df.iloc[i]['title'],
                "type": self.df.iloc[i]['type'],
                "country": self.df.iloc[i]['country'],
                "rating": self.df.iloc[i]['rating'],
                "listed_in": self.df.iloc[i]['listed_in'],
                "description": self.df.iloc[i]['description'],
                "explanation": self.explain_recommendation(
                    self.df.iloc[idx]['title'], self.df.iloc[i]['title']
                ),
                "similarity": f"{combined_sim[i]:.2f}"
            })
        return recommendations

    def explain_recommendation(self, base_title: str, recommended_title: str) -> dict:
        """
        Erklärt, warum ein bestimmter Titel empfohlen wurde.
        """
        base = self.df[self.df['title'] == base_title].iloc[0]
        rec = self.df[self.df['title'] == recommended_title].iloc[0]

        explanation = {
            "Genre Overlap": self._get_overlap(base.get('listed_in'), rec.get('listed_in')),
            "Cast Overlap": self._get_cast_match(base.get('cast'), rec.get('cast')),
            "Rating Match": "Yes" if base.get('rating') == rec.get('rating') else "No",
            "Country Match": "Yes" if base.get('country') == rec.get('country') else "No",
            "Director Match": self._get_director_match(base.get('director'), rec.get('director')),
            "Release Year Difference": abs(base.get('release_year', 0) - rec.get('release_year', 0))
        }
        return explanation

    @staticmethod
    def _get_overlap(base_feature, rec_feature) -> str:
        """
        Berechnet die Überschneidung zwischen zwei Features.
        Falls einer der beiden Werte fehlt oder leer ist, wird "None" zurückgegeben.
        """
        base_feature = base_feature if isinstance(base_feature, str) else ""
        rec_feature = rec_feature if isinstance(rec_feature, str) else ""
        if not base_feature or not rec_feature:
            return "None"
        base_set = set(base_feature.split(', '))
        rec_set = set(rec_feature.split(', '))
        overlap = base_set.intersection(rec_set)
        return f"{len(overlap)} shared ({', '.join(overlap)})" if overlap else "None"

    @staticmethod
    def _get_director_match(base_director, rec_director) -> str:
        base_director = base_director if isinstance(base_director, str) else ""
        rec_director = rec_director if isinstance(rec_director, str) else ""
        if not base_director or not rec_director:
            return "No (Missing Director)"
        if "unknown" in base_director.lower() or "unknown" in rec_director.lower():
            return "No (Unknown Director)"
        return "Yes" if base_director == rec_director else "No"

    @staticmethod
    def _get_cast_match(base_cast, rec_cast) -> str:
        # Fehlende Werte abfangen
        base_cast = base_cast if isinstance(base_cast, str) else ""
        rec_cast = rec_cast if isinstance(rec_cast, str) else ""
        if not base_cast or not rec_cast:
            return "No (Missing Cast)"
        if "unknown" in base_cast.lower() or "unknown" in rec_cast.lower():
            return "No (Unknown Cast)"
        
        # Aufteilen und trimmen
        base_list = [x.strip() for x in base_cast.split(',') if x.strip()]
        rec_list = [x.strip() for x in rec_cast.split(',') if x.strip()]
        base_set = set(map(str.lower, base_list))
        rec_set = set(map(str.lower, rec_list))
        
        # Exakte Übereinstimmung
        if base_set == rec_set and base_set:
            return "Yes"
        
        # Teilüberschneidung
        overlap = base_set & rec_set
        if overlap:
            # Originalnamen sammeln, um die Schreibweise beizubehalten
            matched = [item for item in base_list if item.lower() in overlap]
            matched += [item for item in rec_list if item.lower() in overlap and item not in matched]
            return f"{len(overlap)} shared ({', '.join(matched)})"
        
        return "No"

    def save_model(self, filepath: str):
        try:
            joblib.dump(self, filepath)
            logging.info(f"Modell erfolgreich gespeichert unter: {filepath}")
        except Exception as e:
            logging.error(f"Fehler beim Speichern des Modells: {e}")

    @staticmethod
    def load_model(filepath: str):
        try:
            model = joblib.load(filepath)
            logging.info(f"Modell erfolgreich geladen von: {filepath}")
            return model
        except Exception as e:
            logging.error(f"Fehler beim Laden des Modells: {e}")
            return None

if __name__ == "__main__":
    recommender = HybridRecommender()
    recommender.vectorize_and_combine()
    recommender.save_model("models/hybrid_recommender_model.pkl")

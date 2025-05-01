# Movie Recommendation System with Explainable AI

## Project Objective & Context

This project aims to develop a **content-based Movie Recommendation System** that leverages **TF-IDF** and **cosine similarity** to recommend movies based on their descriptions. The system incorporates **Explainable AI (XAI)** features to provide insights into why specific recommendations are made. It is implemented as a **Flask web API** with a **MongoDB backend** for data storage.

### Dataset
The dataset used for this project was sourced from [Kaggle]([https://www.kaggle.com/](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)), containing movie metadata.

### Problem Statement
With the growing volume of movies available, users often struggle to find content tailored to their preferences. This project addresses the challenge by building a recommendation system that provides personalized movie suggestions while ensuring transparency through explainable recommendations.

### Goals
- **Recommendation Engine**: Suggest movies based on content similarity.
- **Explainability**: Provide clear explanations for recommendations.
- **Scalable Design**: Modular architecture with a Flask API and MongoDB integration.

### Assumptions and Hypotheses
- Movie descriptions and metadata are sufficient to capture user preferences.
- TF-IDF effectively represents textual data for similarity computation.
- Users value transparency in recommendation systems.

---

## Technical Overview

### Features
- **Content-Based Filtering**: Recommends movies using TF-IDF and cosine similarity.
- **Explainable AI**: Highlights key features contributing to recommendations.
- **Modular Design**: Separation of concerns for data processing, model training, and API.
- **Flask API**: Exposes endpoints for recommendations and explanations.
- **MongoDB Integration**: Stores movie data and user interactions.

### Project Structure
```
MovieRecommendation/
├── src/
│   ├── data_processing/      # ETL: cleaning, feature‐engineering, loaders
│   │   ├── clean_dataset.py
│   │   ├── data_cleaner.py
│   │   ├── data_loader.py
│   │   └── mongodb_handler.py
│   │
│   ├── model/                # Training & serialization
│   │   └── build_model.py    # TF-IDF fitting, cosine matrix, save artifacts
│   │
│   └── main/                 # Flask web layer
│       ├── app.py        
│       ├── explanation.py        
│       ├── recommendation.py        
│       ├── static/          
│       │   └── script.js
│       └── templates/        
│           ├── index.html
│           ├── 404.html
│           └── 500.html
│
├── data/                     # for version-controlled small data or pointers
│   ├── raw/                  # raw CSVs (not in git—add in .gitignore)
│   ├── processed/            # processed artifacts
│   └── preprocess.py         # helper to move raw→processed
│
├── created_model/            # created .pkl model
│   └── light_model.pkl
│
├── .env.example              # env template (MONGO_URI, FLASK_ENV, SECRET_KEY)
├── requirements.txt          # Required dependencies
└── README.md                

```

---

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/MovieRecommendation.git
   cd MovieRecommendation
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate          # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Update `.env` with your MongoDB URI:
     ```
     MONGO_URI=mongodb://localhost:27017/movies
     ```
---

## Running the Project

1. **Data Cleaning & Preparation**  
   ```bash
   python src/data_processing/clean_dataset.py
   ```

2. **Model Training & Serialization**  
   ```bash
   python src/model/build_model.py
   ```

3. **Start the Flask API**  
   ```bash
   python src/main/app.py
   ```  

---

## 📊 API Usage

### Endpoints
1. **Get Recommendations**  
   - **URL**: `/recommend`
   - **Method**: `POST`
   - **Request Body**:
     ```json
     {
       "movie_title": "Inception",
       "top_n": 5
     }
     ```
   - **Response**:
     ```json
     {
       "recommendations": [
         {"title": "Interstellar", "similarity_score": 0.85},
         {"title": "The Prestige", "similarity_score": 0.78}
       ]
     }
     ```

2. **Get Explanation**  
   - **URL**: `/explain`
   - **Method**: `POST`
   - **Request Body**:
     ```json
     {
       "movie_title": "Inception",
       "recommended_title": "Interstellar"
     }
     ```
   - **Response**:
     ```json
     {
       "explanation": "Both movies share themes of science fiction and exploration."
     }
     ```

### Example Call
```bash
curl -X POST http://127.0.0.1:5000/recommend \
-H "Content-Type: application/json" \
-d '{"movie_title": "Inception", "top_n": 5}'
```

---

## Reproducibility & Configuration

### Environment Variables
- **MONGO_URI**: MongoDB connection string (e.g., `mongodb://localhost:27017/movies`).

### Reproducibility Steps
1. Ensure all dependencies are installed using `requirements.txt`.
2. Use the provided `.env.example` to configure your environment.
3. Follow the installation steps to preprocess data, train the model, and start the API.

---

## Use of AI Tools (Copilot & ChatGpt)

- **ChatGPT Assistance:**  
  - Drafted README structure and markdown (`README.md`).  
  - Generated boilerplate code snippets for Flask routes.  
  - Generated boilerplate HTML templates and JavaScript (`.html` & `.js`) for the frontend.  
- **Self-Developed:**  
  - Core TF-IDF computation & cosine similarity logic.  
  - XAI explanation extraction module.  
  - MongoDB integration and API orchestration.

---

## Documentation & Attribution

### External Sources
- **TF-IDF**: [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- **Flask**: [Flask Documentation](https://flask.palletsprojects.com/)
- **MongoDB**: [MongoDB Documentation](https://www.mongodb.com/docs/)

![Screenshot](./path/to/screenshot.png)

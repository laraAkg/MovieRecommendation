# Movie Recommendation System with Explainable AI

## Project Objective & Context

This project aims to develop a *content-based Movie Recommendation System* that leverages *TF-IDF* and *k-Nearest Neighbors (k-NN)* to recommend movies based on their descriptions and metadata. The system incorporates *Explainable AI (XAI)* features to provide insights into why specific recommendations are made. It is implemented as a *Flask web API* with a *MongoDB backend* for data storage.

---

## Dataset

The dataset used for this project was sourced from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset), containing movie metadata such as titles, genres, cast, crew, and more.

---

## Problem Statement

With the growing volume of movies available, users often struggle to find content tailored to their preferences. This project addresses the challenge by building a recommendation system that provides personalized movie suggestions.

---

## Goals

- **Recommendation Engine**: Suggest movies based on content similarity using TF-IDF and k-NN models.
- **Explainability**: Provide clear and transparent explanations for recommendations to enhance user trust.
- **Scalable Design**: Modular architecture with a Flask API and MongoDB integration for efficient data handling and deployment.
- **User Experience**: Deliver a user-friendly interface for seamless interaction and recommendations.
- **Data Insights**: Perform exploratory data analysis to uncover patterns and trends in the movie dataset.

---
## Assumptions and Hypotheses

- **Movie descriptions and metadata** are sufficient to capture user preferences and generate meaningful recommendations.
- **Users value transparency and trust** in recommendation systems, which is achieved through Explainable AI (XAI).
---
## Technical Overview

### Features
- **Content-Based Filtering**: Recommends movies using TF-IDF vectors and cosine similarity for similarity computation, along with k-NN models for enhanced accuracy.
- **Explainable AI**: Highlights key features contributing to recommendations, enhancing user trust.
- **Modular Design**: Separation of concerns for data processing, model training, and API integration.
- **Flask API**: Exposes endpoints for recommendations and explanations, ensuring a user-friendly interface.
- **MongoDB Integration**: Efficiently stores movie metadata and user interactions for scalability and performance.
---

## Project Structure

```
MovieRecommendation/
├── Dockerfile               # Docker configuration file for containerization
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── .gitignore
├── data/                    # Dataset directory
│   ├── credits.csv              # Dataset with movie credits information
│   ├── keywords.csv             # Movie-related keywords
│   └── movies_metadata.csv      # Metadata about movies
├── images/                  # Images used in documentation
│   ├── Error_msg.png            # Screenshot of an error message
│   ├── Movie_detail.png         # Screenshot showing movie details
│   ├── Movie_recommender.png    # Screenshot of the recommendation system
│   └── Suggestions.png          # Screenshot of recommended suggestions
└── src/                     # Source code directory
    └── main/
        ├── all_metrics_performance.png   # Performance metrics visualization
        ├── app.py                   # Main application (e.g., Flask API)
        ├── build_model.py           # Builds the recommendation model
        ├── clean_dataset.py         # Data cleaning and preprocessing
        ├── created_model/           # Saved ML models
        │   ├── knn_model.pkl
        │   └── light_model.pkl
        ├── data_loader.py           # Loads and preprocesses data
        ├── model_evaluator.py       # Evaluates model performance
        ├── mongodb_handler.py       # MongoDB interaction module
        ├── movie_eda.ipynb          # Exploratory Data Analysis (EDA) notebook
        ├── recommendation.py        # Generates movie recommendations
        ├── static/                  # Static files (JS, CSS)
        │   └── script.js
        ├── templates/               # HTML templates for the web app
        │   ├── 404.html
        │   ├── 500.html
        │   └── index.html
        └── xai.ipynb                # Explainable AI (XAI) analysis notebook
```
 ⁠
_


## Installation

1.⁠ ⁠*Clone the Repository*:
   ⁠ bash
   git clone https://github.com/laraAkg/MovieRecommendation.git
   cd MovieRecommendation
    ⁠

2.⁠ ⁠*Set Up a Virtual Environment*:
   - For macOS/Linux:
     ⁠ bash
     python3 -m venv venv
     source venv/bin/activate
      ⁠

   - For Windows:
     ⁠ bash
     python -m venv venv
     venv\Scripts\activate
      ⁠

3.⁠ ⁠*Configure Environment Variables*:
   - Copy ⁠ .env_example ⁠ to ⁠ .env ⁠:
     ⁠ bash
     cp .env_example .env
      ⁠

   - Update the ⁠ .env ⁠ file with your MongoDB details. Use the following structure:
     ⁠ env
     MONGO_URI=mongodb+srv://<USERNAME>:<PASSWORD>@movierecommendation.2y0pw.mongodb.net/?retryWrites=true&w=majority&appName=MovieRecommendation
     DB_NAME=netflix_db
     COLLECTION_NAME=recommendation_data
     SECRET_KEY="SeCrEt_K3y"
      ⁠

   - Replace ⁠ <username> ⁠& ⁠ <password> ⁠with your actual MongoDB credentials.

4.⁠ ⁠*Download the Dataset from Kaggle*:  
   Due to file size limits on GitHub, the raw CSV files are not included in this repository.  
   Please download the following files from Kaggle and place them into the ⁠ data ⁠ folder before running the pipeline:
   
    [movies_metadata.csv](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv)  
    [credits.csv](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=credits.csv)  
    [keywords.csv](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=keywords.csv)  

   Place these files in the following directory structure:

   ```
   MovieRecommendation/
   └── data/
      ├── movies_metadata.csv
      ├── credits.csv
      └── keywords.csv
   ```

5.⁠ ⁠*Install Dependencies*:
   - Ensure you are in the *root directory* of the project (⁠ MovieRecommendation/ ⁠):
     ⁠ bash
     pip install -r requirements.txt
      ⁠

---

## Running the Project

1.⁠ ⁠*Clean and Prepare the Data*:  
   Run the following script to clean the dataset and load it into MongoDB:
   ⁠ bash
   python src/main/clean_dataset.py
    ⁠

2.⁠ ⁠*Build the Models*:  
   Train the recommendation models (TF-IDF and k-NN) and save them as ⁠ .pkl ⁠ files:
   ⁠ bash
   python src/main/build_model.py
    ⁠

3.⁠ ⁠*Start the Flask Server*:  
   Launch the Flask application:
   ⁠ bash
   python src/main/app.py
    ⁠

4.⁠ ⁠*Open the Application*:  
   Once the Flask API is running, open your browser and navigate to [http://127.0.0.1:5002/](http://127.0.0.1:5002/) to access the application.


---   

## API Usage

The Flask API provides the following endpoints:

1.⁠ ⁠⁠ / ⁠ – *Main Page*
   - *Method*: ⁠ GET ⁠ and ⁠ POST ⁠
   - *Description*:
     - On ⁠ GET ⁠: Displays the main page where users can input a movie title and select the recommendation model (TF-IDF or k-NN).
     - On ⁠ POST ⁠: Accepts a movie title and returns a list of recommended movies.

2.⁠ ⁠⁠ /titles ⁠ – *Fetch Movie Titles*
   - *Method*: ⁠ GET ⁠
   - *Description*: Returns a list of all movie titles in the dataset for autocomplete functionality.

---

## Reproducibility & Configuration

**Environment Variables**:  
Ensure the `.env` file is configured with MongoDB credentials.

**Reproducibility Steps**:  
1. Ensure all dependencies are installed using `requirements.txt`.  
2. Use the provided `.env_example` to configure your environment.  
3. Follow the installation steps to preprocess data, train the model, and start the API.
---

## Exploratory Data Analysis (EDA)

Before building the recommendation system, an exploratory data analysis (EDA) was conducted to better understand the dataset and extract meaningful insights. The EDA process included:

1.⁠ ⁠*Data Overview*:
   - Summarized the dataset to identify missing values, outliers, and inconsistencies.
   - Analyzed key statistics.

2.⁠ ⁠*Relationships and Correlations*:
   - Explored relationships between features using a correlation matrix.
   - Created histograms and scatterplots to visualize distributions and relationships, such as:
     - *Vote Count vs. Vote Average*
     - *Budget vs. Vote Average*
     - *Runtime vs. Vote Count*

3.⁠ ⁠*Hypothesis Testing*:
   - Tested hypotheses, such as:
     - Do higher-budget movies tend to have higher ratings?
     - Do newer movies receive more votes?

The EDA was performed in a Jupyter Notebook located in the ⁠ movie_eda.ipynb ⁠ file. This analysis provided the foundation for data cleaning, feature engineering, and model development.

---

## XAI

The `xai.ipynb` notebook demonstrates the use of Explainable AI (XAI) to enhance transparency in the recommendation system. It provides interpretable insights into why certain movies are recommended, using the following techniques:

- **SHAP (SHapley Additive exPlanations)**: Visualizes the contribution of each feature to the model's output, helping explain why a particular movie was suggested.
- **Feature Importance**: Highlights key features such as genre, keywords, and movie descriptions that influence recommendation decisions.
- **Visual Interpretations**: Includes example plots that illustrate which factors had the greatest impact on the similarity between movies.

The goal of the XAI component is to build trust with users by providing clear and understandable explanations behind the recommendations.

---


## Docker

This project has been fully containerized using a custom `Dockerfile` and is now available on Docker Hub.

The Docker image includes all required dependencies and can be used to deploy the application in any Docker-supported environment.

- **Docker Hub:** lara283/movie-recommendation
- **Version:** `latest`

```docker pull lara283/movie_recommendation:latest```

![Docker](images/Docker.png)

---

## Use of AI Tools (Copilot & ChatGPT)

•⁠  ⁠*ChatGPT Assistance*:  
  ChatGPT was used to:
  - Draft the README structure and markdown (⁠ README.md ⁠).
  - Provide suggestions for documentation and clear descriptions of project sections.
  - Generate boilerplate code snippets for Flask routes.
  - Generate boilerplate HTML templates and JavaScript (⁠ .html ⁠ & ⁠ .js ⁠) for the frontend.
  - Assist in formulating installation and configuration instructions.

•⁠  ⁠*GitHub Copilot*:  
  GitHub Copilot was used to:
  - Generate code snippets for tasks like loading environment variables or building models.
  - Suggest implementations for functions such as data cleaning, model training, and API endpoints.
  - Accelerate development through context-aware code completions.

---

## Documentation & Attribution

### External Sources
⁠*TF-IDF*: [Scikit-learn Documentation](http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)  
⁠*Cosine Similarity*: [Scikit-learn Documentation](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)  
⁠*MongoDB*: [MongoDB Documentation](http://www.mongodb.com/docs/)  
⁠*Kaggle Dataset*: [The Movies Dataset](http://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)  
  
---


## Screenshots

### Movie recommender
![Movie recommender](images/Movie_recommender.png)

### Recommendation Example
![Recommendation example](images/Movie_detail.png)

### Movie detail
![Movie detail](images/Movie_detail.png)

### Error message
![Error message](images/Error_msg.png)

---
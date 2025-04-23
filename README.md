# Movie Recommendation System
A content-based movie recommendation system with explainable AI (XAI) features, built in Python. The project includes data processing, model building, recommendation logic, and a Flask web application.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Data Processing](#data-processing)
  - [Model Building](#model-building)
  - [Running the API](#running-the-api)
- [Endpoints](#endpoints)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Content-Based Filtering**: Uses TF-IDF and cosine similarity to recommend movies based on plot, genres, cast, and other features.
- **Explainable Recommendations**: Generates explanations highlighting shared genres, keywords, taglines, etc.
- **Modular Architecture**: Separate modules for data processing, model creation, recommendation logic, and the web app.
- **Scalable Storage**: Integrates with MongoDB for efficient data storage and retrieval.
- **Web Interface**: Flask-based REST API with autocomplete search.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/MovieRecommendation.git
   cd MovieRecommendation/src
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scriptsctivate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Environment variables**: Copy the `.env` file and set your values:
   ```env
   MONGO_URI=<your_mongodb_uri>
   DB_NAME=<database_name>
   COLLECTION_NAME=<collection_name>
   ```

2. **Ensure MongoDB is running** or accessible via the URI provided.

## Usage

### Data Processing

Run the data cleaning and upload pipeline to MongoDB:
```bash
python src/data_processing/clean_dataset.py
```

### Model Building

Build the TF-IDF model and save artifacts:
```bash
python src/model/build_model.py
```

This saves `light_model.pkl` in `created_model/`.

### Running the API

Start the Flask web server:
```bash
python src/app.py
```

Visit `http://localhost:5000` in your browser.

## Endpoints

- `GET /titles` : Returns a JSON array of all movie titles.
- `GET /recommend?title=<movie_title>&top_n=<N>` : Returns top N recommendations for the given title.

## Examples

![UI Screenshot](./docs/images/ui_screenshot.png) <!-- Placeholder for screenshot -->

```bash
curl "http://localhost:5000/recommend?title=Inception&top_n=5"
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

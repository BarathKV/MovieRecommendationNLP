# Movie Recommendation System using NLP Concepts

This project is a movie recommendation system that leverages Natural Language Processing (NLP) techniques to recommend similar movies based on their **overview**, **genres**, and **keywords**. The system processes movie data, converts textual information into vector representations, and uses a K-Nearest Neighbors (KNN) model to find and recommend movies with similar content.

---

## Dataset

The dataset used in this project is sourced from Kaggle. It consists of two CSV files:
1. **Keywords File**: Contains movie IDs and associated keywords in a JSON-like format.
2. **Metadata File**: Contains detailed information about movies, including their title, genres, and overview.

You can download the dataset from the following link:
[Insert Kaggle Dataset Link Here](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

---

## Project Workflow

### 1. Data Cleaning and Preprocessing (`exploration.ipynb`)
- The **keywords** and **metadata** files are cleaned and processed.
- JSON-like formats in the keywords and genres columns are converted into space-separated words for easier processing.

### 2. Vectorization (`vectorizing.ipynb`)
- The **keywords**, **overview**, and **genres** columns are converted into vector representations using:
  - **TF-IDF Vectorization**
  - **Hashing Vectorization**
- The resulting vectors are:
  - Converted into `scipy.sparse_matrix` format.
  - Normalized to ensure uniform scaling.
  - Reduced in dimensionality using **Truncated SVD** to retain only the most important features.

### 3. Word2Vec Model Training (`word2vec.ipynb`)
- A **Word2Vec** model is trained using all the words from the dataset (keywords, genres, and overview).
- Vector representations for each attribute of each movie are extracted using the trained Word2Vec model.

### 4. Word2Vec Vector Processing (`w2v.ipynb`)
- The Word2Vec vector values are:
  - Read as NumPy arrays.
  - Converted into `scipy.sparse_matrix` format.
  - Normalized and reduced in dimensionality using **Truncated SVD**.

### 5. Movie Recommendation (`playgrnd.ipynb`)
- A **K-Nearest Neighbors (KNN)** model is trained using the chosen vectorized matrix (TF-IDF, Hashing, or Word2Vec).
- For a given movie, the KNN model finds the k-nearest neighbors based on cosine similarity.
- The neighboring vector representations correspond to movies with similar content (based on genres, keywords, overview, or a combination of these).

---

## How It Works
1. **Input**: Provide the title of a movie.
2. **Processing**: The system computes the vector representation of the input movie and finds its nearest neighbors using the KNN model.
3. **Output**: A list of recommended movies with similar content is displayed.

---

## Key Features
- **Customizable Recommendations**: Choose the vectorization method (TF-IDF, Hashing, or Word2Vec) to tailor recommendations.
- **Content-Based Filtering**: Recommendations are based on the content of the movie (genres, keywords, and overview).
- **Efficient Processing**: Dimensionality reduction ensures faster computations without losing important information.

---

## Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `pickle`, `ast`, `re`, `scipy`, `scikit-learn`, `gensim`, `scipy`

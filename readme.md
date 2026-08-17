# 🎬 Movie Recommendation System

A Flask-based Machine Learning web application that recommends similar movies using **content-based filtering**, built on movie overviews and genres from the **TMDB 5000 Movies dataset**.

Users search for a movie they like, and the app instantly returns the top 6 most similar films based on cosine similarity between vectorized plot summaries and genres — displayed as an animated movie grid with match percentage and rating.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Cosine%20Similarity-F7931E.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458.svg)
![Deployment](https://img.shields.io/badge/Deployment-Render-46E3B7.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🌐 Live Demo

**🚀 Try it live:**

https://movie-recommendation-system-8to3.onrender.com

> _Note: Render free-tier services may take a few seconds to wake up after inactivity._

---

## ✨ Features

### 🎥 Content-Based Recommendations

- Recommends the top 6 most similar movies based on plot (`overview`) and `genres`.
- Text vectorized with `CountVectorizer` (bag-of-words, 5000 features), preceded by Porter stemming to normalize word variations.
- Similarity computed via cosine similarity across the full movie catalog (~4800 titles).
- Each recommendation shows a similarity match percentage and average rating.

### 🔎 Searchable Autocomplete

- Live search-as-you-type dropdown populated from the actual trained movie list — no invalid titles can be submitted.
- Full keyboard navigation (arrow keys + Enter) for the suggestions dropdown.

### 🎨 User Interface

- Cinematic dark-themed glassmorphism design with animated gradient background.
- Responsive movie card grid, adapting from 6 columns on desktop to 2 on mobile.
- Animated floating shapes and card entrance animations.

### ☁️ Deployment Ready

- Render deployment support.
- Gunicorn production server.
- Dataset (`tmdb_5000_movies.csv`) excluded from the deployed instance via `.gitignore`.

---

## 🛠️ Tech Stack

| Category | Technology |
|-----------|------------|
| Backend | Flask, Python |
| Machine Learning | scikit-learn (`CountVectorizer`, cosine similarity), NLTK (Porter Stemmer) |
| Data Handling | Pandas, NumPy |
| Frontend | HTML5, CSS3, JavaScript |
| UI Design | Glassmorphism, Animations |
| Deployment | Render |
| Version Control | Git & GitHub |

---

## 📂 Project Structure

```text
MovieRecommendationSystem/
│
├── app.py
├── movies.pkl
├── similarity.pkl
├── requirements.txt
├── render.yaml
├── README.md
├── .gitignore
│
└── templates/
    └── index.html
```

> ⚠️ `tmdb_5000_movies.csv`, the training notebook, and any `.ipynb_checkpoints` are excluded via `.gitignore` — only `movies.pkl`, `similarity.pkl`, and the application code are deployed.

---

## 🚀 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Create Virtual Environment

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Trained Artifacts

Place these two files in the project root (generated from the training notebook — see below):

```text
movies.pkl
similarity.pkl
```

### 5. Run Application

```bash
python app.py
```

### 6. Open Browser

```text
http://127.0.0.1:5000
```

---

## 🧠 Model Training Summary

1. **Data** — TMDB 5000 Movies dataset (`tmdb_5000_movies.csv`), ~4800 movies.
2. **Columns used** — `title`, `genres`, `overview`, `vote_count`, `vote_average`.
3. **Genre parsing** — `genres` arrives as a JSON-like string (e.g. `[{"id": 28, "name": "Action"}]`); parsed with `ast.literal_eval` into a plain list of genre names.
4. **Tags column** — `overview + genres` combined into one lowercased text field per movie.
5. **Stemming** — Porter Stemmer applied to normalize word variations (e.g. "loving"/"loved" → "love").
6. **Vectorization** — `CountVectorizer(max_features=5000, stop_words='english')` converts tags into numeric vectors.
7. **Similarity** — `cosine_similarity()` computed across all movie vectors, producing an N × N similarity matrix (stored as `float32` to reduce file size).
8. **Recommendation logic** — for a given movie, sorts all other movies by similarity score and returns the top matches.

### ⚠️ Fixes applied during notebook development

- `df.groupby('title').mean()['vote_count']` failed with `TypeError: dtype 'str' does not support operation 'mean'` in recent pandas — fixed with `df.groupby('title').mean(numeric_only=True)['vote_count']`, since `.mean()` now defaults to including (and failing on) non-numeric columns.
- `similarity` matrix cast to `float32` before saving — halves file size versus the pandas/numpy default `float64`, important since this matrix scales quadratically with the number of movies and can approach GitHub's 100 MB file limit.

---

## 💡 Usage

1. Start typing a movie title in the search box (e.g. "Avatar").
2. Select a movie from the autocomplete suggestions (or press Enter).
3. Click **Get Recommendations**.
4. Browse the 6 recommended movies, each showing its similarity match % and average rating.

---

## 🎯 Recommendation Logic

```python
movie_index = movies[movies['title'] == movie_title].index[0]
distances = similarity[movie_index]

top_matches = sorted(
    enumerate(distances), reverse=True, key=lambda x: x[1]
)[1:7]  # skip the movie itself, take top 6
```

---

## ☁️ Deployment on Render

### Required Files

```text
requirements.txt
render.yaml
app.py
movies.pkl
similarity.pkl
templates/index.html
```

### Deployment Steps

1. Push project to GitHub (dataset/notebook excluded via `.gitignore`).
2. Log in to https://render.com
3. Click **New +** → **Web Service**.
4. Connect your GitHub repository.
5. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --workers 1 --timeout 60 app:app`
   - **Runtime:** Python
6. Click **Create Web Service**.

### Auto-deploy

Every `git push` to `main` triggers an automatic redeploy on Render.

---

## ⚠️ Note on `similarity.pkl` File Size

The similarity matrix scales quadratically with the number of movies (N × N floats). For ~4800 movies stored as `float32`, expect roughly **80–90 MB** — close to GitHub's 100 MB limit.

If it exceeds 100 MB, use Git LFS:

```bash
git lfs install
git lfs track "similarity.pkl"
git add .gitattributes
git add similarity.pkl
git commit -m "Add similarity matrix using Git LFS"
git push origin main
```

And ensure your Render build pulls LFS files if needed (see prior projects' `render.yaml` notes on `git lfs pull`).

---

## 🔮 Future Enhancements

- Fetch real movie posters via the TMDB API (currently uses a placeholder icon).
- Add `keywords` and `cast`/`crew` to the tags column for richer similarity.
- Switch to `TfidfVectorizer` and compare recommendation quality.
- "Popular Movies" homepage section using the `avg_vote`/`num_vote` table computed during training.
- Genre-based filtering alongside similarity search.

---

## 👥 Who Is This For?

- 🎓 Students learning NLP-based recommendation systems.
- 🤖 Machine Learning Engineers.
- 🍿 Movie enthusiasts building portfolio projects.
- 🌐 Flask Developers.

---

## ⚠️ Disclaimer

> This project is intended for educational and learning purposes only.
>
> Recommendations are based on plot/genre similarity only and do not account for personal taste, ratings history, or collaborative filtering signals.

---

## 👨‍💻 Author

**Suraj Prakash Verma**

- 🏢 UST Global
- 🌐 GitHub: https://github.com/surajprakashverma

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🌟 Show Your Support

If you found this project useful:

⭐ Star the repository
🍴 Fork the project
🛠️ Contribute improvements
📢 Share with fellow developers

Happy Coding! 🚀

import ast
import logging
import math
import pickle

import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MOVIES_PATH = "movies.pkl"
SIMILARITY_PATH = "similarity.pkl"
TMDB_CSV_PATH = "tmdb_5000_movies.csv"

movies = pickle.load(open(MOVIES_PATH, "rb"))
similarity = pickle.load(open(SIMILARITY_PATH, "rb"))
tmdb_df = pd.read_csv(TMDB_CSV_PATH)


def parse_genres(genre_str):
    try:
        genres_list = ast.literal_eval(genre_str)
        return [g["name"] for g in genres_list]
    except (ValueError, SyntaxError, TypeError):
        return []


tmdb_df["genres_parsed"] = tmdb_df["genres"].apply(parse_genres)
tmdb_df["release_year"] = pd.to_datetime(
    tmdb_df["release_date"], errors="coerce"
).dt.year

tmdb_details = tmdb_df.drop_duplicates(subset="title", keep="first")[[
    "title",
    "vote_count",
    "vote_average",
    "release_year",
    "runtime",
    "tagline",
    "overview",
    "original_language",
    "genres_parsed",
]].rename(columns={"genres_parsed": "genres"})

movies_with_details = movies[["title"]].merge(tmdb_details, on="title", how="left")


MOVIE_TITLES = sorted(movies["title"].dropna().unique().tolist())


def safe_value(row, column, cast=None, default=None):
    """
    Safely pull a value out of a pandas row, handling missing columns,
    NaN, and optional type casting — returns `default` on any failure
    instead of raising, since not every movie has every field filled in.
    """
    if column not in row:
        return default

    value = row[column]

    if value is None:
        return default

    try:
        if isinstance(value, float) and math.isnan(value):
            return default
    except TypeError:
        pass

    if cast is not None:
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default

    return value


def build_movie_details(title):
    """
    Looks up the CSV-derived details for a single movie title, used
    both for the recommendation list (release_year only, lightweight)
    and the full popup detail view (everything).
    """
    matches = movies_with_details[movies_with_details["title"] == title]

    if matches.empty:
        return {
            "title": title,
            "vote_average": None,
            "vote_count": None,
            "release_year": None,
            "runtime": None,
            "tagline": "",
            "overview": "",
            "original_language": "",
            "genres": [],
        }

    row = matches.iloc[0]

    return {
        "title": title,
        "vote_average": safe_value(row, "vote_average", cast=float),
        "vote_count": safe_value(row, "vote_count", cast=int),
        "release_year": safe_value(row, "release_year", cast=int),
        "runtime": safe_value(row, "runtime", cast=int),
        "tagline": safe_value(row, "tagline", cast=str, default=""),
        "overview": safe_value(row, "overview", cast=str, default=""),
        "original_language": safe_value(row, "original_language", cast=str, default="").upper(),
        "genres": row["genres"] if isinstance(row["genres"], list) else [],
    }


def get_recommendations(movie_title, top_n=50):
    """
    Returns up to top_n recommended movies from the MODEL (title +
    similarity score only, exactly what movies.pkl/similarity.pkl
    provide), enriched with lightweight display details (release
    year, rating) from the CSV for the card view.
    """
    matches = movies[movies["title"] == movie_title]

    if matches.empty:
        return None

    movie_index = matches.index[0]
    distances = similarity[movie_index]

    similar_indices = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1: top_n + 1]

    recommendations = []
    for idx, score in similar_indices:
        title = movies.iloc[idx]["title"]
        details = build_movie_details(title)
        details["similarity"] = round(float(score) * 100, 1)
        recommendations.append(details)

    return recommendations


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", movie_titles=MOVIE_TITLES)


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json(silent=True)

        if not data or "movie_title" not in data:
            return jsonify({"error": "No movie title provided."}), 400

        movie_title = str(data["movie_title"]).strip()

        if movie_title == "":
            return jsonify({"error": "Please select a movie."}), 400

        if movie_title not in MOVIE_TITLES:
            return jsonify({
                "error": f"'{movie_title}' was not found in our database. "
                         f"Please select a movie from the suggestions list."
            }), 404

        recommendations = get_recommendations(movie_title)

        if recommendations is None:
            return jsonify({"error": "Could not generate recommendations for this movie."}), 500

        return jsonify({
            "selected_movie": movie_title,
            "recommendations": recommendations
        }), 200

    except Exception as e:
        return jsonify({"error": f"Recommendation failed: {str(e)}"}), 500


@app.route("/movie-details", methods=["POST"])
def movie_details_route():
    """
    Called when the user clicks a recommended movie card — returns
    the FULL details (tagline, overview, genres, runtime, etc.) for
    the popup modal, sourced from tmdb_5000_movies.csv.
    """
    try:
        data = request.get_json(silent=True)
        if not data or "movie_title" not in data:
            return jsonify({"error": "No movie title provided."}), 400

        movie_title = str(data["movie_title"]).strip()

        if movie_title not in MOVIE_TITLES:
            return jsonify({"error": f"'{movie_title}' was not found in our database."}), 404

        details = build_movie_details(movie_title)

        return jsonify(details), 200

    except Exception as e:
        return jsonify({"error": f"Could not load movie details: {str(e)}"}), 500


@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Route not found."}), 404


@app.errorhandler(500)
def handle_500(e):
    return jsonify({"error": "Internal server error. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=True)

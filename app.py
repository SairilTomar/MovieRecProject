from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

app = Flask(__name__)

# Load data 
df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Movies%20Recommendation.csv')

# Prepare data
df_features = df[['Movie_Director', 'Movie_Cast', 'Movie_Tagline', 'Movie_Language', 'Movie_Overview']].fillna('')
combined_text = (
    df_features['Movie_Director'] + ' ' +
    df_features['Movie_Cast'] + ' ' +
    df_features['Movie_Language'] + ' ' +
    df_features['Movie_Tagline'] + ' ' +
    df_features['Movie_Overview']
)

# TF-IDF and Similarity Matrix
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(combined_text)
Similarity_Score = cosine_similarity(x)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    user_movie = ""
    if request.method == "POST":
        user_movie = request.form["movie"]
        all_titles = df["Movie_Title"].tolist()
        matches = difflib.get_close_matches(user_movie, all_titles)
        if matches:
            close_match = matches[0]
            movie_id = df[df.Movie_Title == close_match]["Movie_ID"].values[0]
            scores = list(enumerate(Similarity_Score[movie_id]))
            sorted_movies = sorted(scores, key=lambda x: x[1], reverse=True)
            for i, movie in enumerate(sorted_movies[:10], 1):
                index = movie[0]
                title = df.iloc[index]["Movie_Title"]
                recommendations.append(f"{i}. {title}")
        else:
            recommendations.append("No close match found. Please try another movie name.")
    return render_template("index.html", recommendations=recommendations, user_movie=user_movie)

# if __name__ == "__main__":
#     import webbrowser
#     webbrowser.open_new("http://127.0.0.1:5000/")
#     app.run(debug=True)

if __name__ == "__main__":
    import webbrowser
    webbrowser.open_new("http://127.0.0.1:5000/")
    app.run(debug=True)


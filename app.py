from flask import Flask, render_template, request, jsonify
from sbrsimp import books, ratings, predictions, gru_preds  # Use existing models/data
import numpy as np
import pandas as pd

app = Flask(__name__)

# Pre-compute useful values
num_users = len(ratings['User_ID'].astype('category').cat.categories)

@app.route("/")
def index():
    return render_template("ui.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = request.form.get("user_id")

    if not user_id or not user_id.startswith("U"):
        return jsonify({"error": "Invalid User ID format. Use U001, U002, etc."}), 400

    try:
        user_idx = int(user_id[1:]) - 1

        if user_idx < 0 or user_idx >= num_users:
            return jsonify([{"error": "User ID not found"}]), 404

        # SVD Recommendations
        user_ratings = predictions[user_idx]
        svd_top_indices = np.argsort(user_ratings)[-5:][::-1]
        svd_books = books.iloc[svd_top_indices][["Book Title", "Book Genre"]].copy()
        svd_books["Source"] = "SVD"

        # GRU Recommendations
        gru_top_indices = np.argsort(gru_preds)[-5:][::-1]
        gru_books = books.iloc[gru_top_indices][["Book Title", "Book Genre"]].copy()
        gru_books["Source"] = "GRU"

        # Combine & Drop duplicates
        combined = pd.concat([svd_books, gru_books]).drop_duplicates(subset=["Book Title"])
        return jsonify(combined.to_dict(orient="records"))

    except Exception as e:
        return jsonify([{"error": f"Exception occurred: {str(e)}"}]), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

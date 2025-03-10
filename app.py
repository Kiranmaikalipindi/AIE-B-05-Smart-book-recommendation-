from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Data (Ensure this path is correct)
DATASET_PATH = r"c:\Users\anany\Downloads\Improved_Real_Book_Dataset (1).xlsx"

try:
    books = pd.read_excel(DATASET_PATH, sheet_name="Books")
    ratings = pd.read_excel(DATASET_PATH, sheet_name="Ratings")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    books = pd.DataFrame(columns=["Book Title", "Book Genre"])  # Fallback empty DataFrame
    ratings = pd.DataFrame(columns=["User_ID", "Book_ID", "Rating"])

# 🔹 Dummy function for book recommendation (Replace with real model logic)
def hybrid_recommend(user_id):
    try:
        # Convert "U001" → "1" → index-based lookup
        user_idx = int(user_id[1:]) - 1  

        # Ensure the index is within valid range
        if user_idx < 0 or user_idx >= len(ratings['User_ID'].unique()):
            return [{"error": "User ID not found"}]
        
        print(f"User Index: {user_idx}")  # Debugging

        num_recommendations = 5
        recommended_books = books.sample(num_recommendations)[["Book Title", "Book Genre"]]
        print(recommended_books)  # Debugging

        return recommended_books.to_dict(orient="records")

    except Exception as e:
        print(f"Error: {str(e)}")  # Print errors
        return [{"error": f"An error occurred: {str(e)}"}]

@app.route("/")
def index():
    return render_template("ui.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = request.form.get("user_id")

    if not user_id or not user_id.startswith("U"):
        return jsonify({"error": "Invalid User ID format. Use U001, U002, etc."}), 400

    recommendations = hybrid_recommend(user_id)

    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # Prevents reloader issues

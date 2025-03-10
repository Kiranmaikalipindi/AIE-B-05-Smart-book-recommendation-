import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 📌 Enable Debug Mode for tf.data Execution
tf.data.experimental.enable_debug_mode()

# 📌 Load Dataset
file_path = r"c:\Users\anany\Downloads\Improved_Real_Book_Dataset (1).xlsx"
xls = pd.ExcelFile(file_path)
books = xls.parse("Books")
ratings = xls.parse("Ratings")

# 📌 Normalize Ratings
scaler = MinMaxScaler()
ratings['Rating'] = scaler.fit_transform(ratings[['Rating']])

# 📌 Convert User and Book IDs to Categorical Codes
user_ids = ratings['User_ID'].astype('category').cat.codes.values
book_ids = ratings['Book_ID'].astype('category').cat.codes.values

num_users = len(ratings['User_ID'].unique())
num_books = len(ratings['Book_ID'].unique())

# 📌 Create User-Book Rating Matrix
ratings_matrix = np.zeros((num_users, num_books))
for user, book, rating in zip(user_ids, book_ids, ratings['Rating']):
    ratings_matrix[user, book] = rating

# 📌 Handle Missing Ratings Using Mean Imputation
mean_user_rating = np.mean(ratings_matrix, axis=1).reshape(-1, 1)
ratings_matrix_adj = ratings_matrix - mean_user_rating  # Normalize by User Mean

# 📌 Train-Test Split
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# 📌 Optimized SVD Implementation
k_value = min(500, min(ratings_matrix.shape) - 1)
U, sigma, Vt = svds(ratings_matrix_adj, k=k_value)
sigma = np.diag(sigma)
predictions = np.dot(np.dot(U, sigma), Vt) + mean_user_rating  # Add Back Mean

# 📌 Scale Back Predictions
predictions = scaler.inverse_transform(predictions)

# 📌 Evaluate SVD MAE & RMSE
test_user_idx = test_data['User_ID'].astype('category').cat.codes.values
test_book_idx = test_data['Book_ID'].astype('category').cat.codes.values
test_preds = [predictions[user, book] for user, book in zip(test_user_idx, test_book_idx)]

mae_svd = mean_absolute_error(scaler.inverse_transform(test_data[['Rating']]), test_preds)
rmse_svd = np.sqrt(mean_squared_error(scaler.inverse_transform(test_data[['Rating']]), test_preds))

# 📌 Content-Based Filtering: Cosine Similarity on Genres
tokenizer = Tokenizer(num_words=10000)  
books['text_features'] = books['Book Title'] + ' ' + books['Book Genre']
tokenizer.fit_on_texts(books['text_features'])
sequences = tokenizer.texts_to_sequences(books['text_features'])
padded_sequences = pad_sequences(sequences, maxlen=150)

# Compute Cosine Similarity between Books
content_similarity = cosine_similarity(padded_sequences)

# 📌 Preparing Training Data for LSTM & GRU Models
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, books['Popularity_Score'], test_size=0.2, random_state=42)

# 📌 Define New Optimizer Instance for Each Model
def create_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=0.001)

# 📌 Clear TensorFlow Session Before Training LSTM
tf.keras.backend.clear_session()

# 📌 Define Optimized LSTM Model
lstm_model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=512, input_length=150),
    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.6),
    LSTM(64),
    Dropout(0.6),
    Dense(32, activation='relu'),
    Dense(1, activation='relu')
])

lstm_model.compile(optimizer=create_optimizer(), loss='mae', metrics=['mae'])

# 📌 Train LSTM Model
lstm_model.fit(X_train, y_train, epochs=150, batch_size=128, validation_data=(X_test, y_test), verbose=1)

# 📌 Clear Session Before Training GRU
tf.keras.backend.clear_session()

# 📌 Define Optimized GRU Model
gru_model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=512, input_length=150),
    Bidirectional(GRU(128, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.7),
    GRU(64),
    Dropout(0.7),
    Dense(32, activation='relu'),
    Dense(1, activation='relu')
])

gru_model.compile(optimizer=create_optimizer(), loss='mae', metrics=['mae'])

# 📌 Train GRU Model
gru_model.fit(X_train, y_train, epochs=150, batch_size=128, validation_data=(X_test, y_test), verbose=1)

# 📌 Evaluate LSTM & GRU Models
lstm_loss, lstm_mae = lstm_model.evaluate(X_test, y_test, verbose=1)
gru_loss, gru_mae = gru_model.evaluate(X_test, y_test, verbose=1)

# 📌 Hybrid Recommendation (SVD + LSTM + Content-Based)
def hybrid_recommend(user_idx, num_recommendations=5):
    collaborative_scores = predictions[user_idx]
    content_scores = np.mean(content_similarity, axis=1)
    lstm_scores = lstm_model.predict(padded_sequences).flatten()

    min_length = min(len(collaborative_scores), len(lstm_scores), len(content_scores))
    collaborative_scores = collaborative_scores[:min_length]
    lstm_scores = lstm_scores[:min_length]
    content_scores = content_scores[:min_length]

    final_scores = (0.3 * collaborative_scores) + (0.5 * lstm_scores) + (0.2 * content_scores)

    recommended_books_idx = np.argsort(final_scores)[-num_recommendations:][::-1]
    recommended_books = books.iloc[recommended_books_idx]
    return recommended_books[['Book Title', 'Book Genre', 'Popularity_Score']]

# 📌 Display Final Results
print("\n🔹 Model Performance:")
print(f"✅ Optimized SVD MAE: {mae_svd:.4f} | RMSE: {rmse_svd:.4f}")
print(f"✅ Improved LSTM MAE: {lstm_mae:.4f}")
print(f"✅ Improved GRU MAE: {gru_mae:.4f}")

# 📌 Display Sample Recommendations
user_sample = np.random.randint(0, num_users)
recommended_books = hybrid_recommend(user_sample, 5)
print(f"\n📚 Recommended Books for User {user_sample}:")
print(recommended_books)

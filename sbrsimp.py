import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Embedding, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


# Load Dataset
file_path = r"c:\Users\anany\Downloads\Improved_Real_Book_Dataset (1).xlsx"
xls = pd.ExcelFile(file_path)
books = xls.parse("Books")
ratings = xls.parse("Ratings")


# Normalize Ratings
scaler = MinMaxScaler()
ratings['Rating'] = scaler.fit_transform(ratings[['Rating']])


# Convert IDs to categorical codes
user_ids = ratings['User_ID'].astype('category').cat.codes.values
book_ids = ratings['Book_ID'].astype('category').cat.codes.values
num_users = len(ratings['User_ID'].unique())
num_books = len(ratings['Book_ID'].unique())


# Create Rating Matrix with Book Mean Imputation
ratings_matrix = np.zeros((num_users, num_books))
for user, book, rating in zip(user_ids, book_ids, ratings['Rating']):
    ratings_matrix[user, book] = rating

for j in range(num_books):
    col = ratings_matrix[:, j]
    nonzero = col[col != 0]
    if len(nonzero) > 0:
        col[col == 0] = nonzero.mean()
        ratings_matrix[:, j] = col


# SVD Implementation
k_value = min(50, min(ratings_matrix.shape) - 1)
U, sigma, Vt = svds(ratings_matrix, k=k_value)
sigma = np.diag(sigma)
predictions = np.dot(np.dot(U, sigma), Vt)
predictions = np.clip(predictions, 0, 1)


# Train-Test Split for Evaluation
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
test_user_idx = test_data['User_ID'].astype('category').cat.codes.values
test_book_idx = test_data['Book_ID'].astype('category').cat.codes.values
test_preds = [predictions[u, b] for u, b in zip(test_user_idx, test_book_idx)]


# Evaluate SVD
mae_svd = mean_absolute_error(test_data['Rating'], test_preds)
rmse_svd = np.sqrt(mean_squared_error(test_data['Rating'], test_preds))


# Accuracy Calculation
def calculate_accuracy(y_true, y_pred, threshold=0.5):
    correct = np.abs(y_true - y_pred) <= threshold
    return np.mean(correct) * 100

accuracy_svd = calculate_accuracy(test_data['Rating'].values, test_preds)


# GRU Content-Based Model
tokenizer = Tokenizer(num_words=10000)
books['text_features'] = books['Book Title'] + ' ' + books['Book Genre']
tokenizer.fit_on_texts(books['text_features'])
sequences = tokenizer.texts_to_sequences(books['text_features'])
padded_sequences = pad_sequences(sequences, maxlen=150)

X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, books['Popularity_Score'], test_size=0.2, random_state=42)


# Define GRU Model with Medium Capacity (Expected Accuracy: ~90–95%)
tf.keras.backend.clear_session()

gru_model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=150),
    Bidirectional(GRU(64, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.4),
    GRU(32),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dense(1, activation='relu')
])

gru_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])


# Train GRU Model FULL 150 Epochs 
gru_model.fit(X_train, y_train, epochs=150, batch_size=64,
              validation_data=(X_test, y_test), verbose=1)


# Evaluate GRU Model
gru_loss, gru_mae = gru_model.evaluate(X_test, y_test, verbose=0)
gru_preds = gru_model.predict(X_test).flatten()
accuracy_gru = calculate_accuracy(y_test.values, gru_preds)


# Final Results
print("\n Model Performance:")
print(f"Optimized SVD MAE: {mae_svd:.4f} | RMSE: {rmse_svd:.4f}")
print(f"Improved GRU MAE: {gru_mae:.4f}")
print(f"Custom GRU Accuracy (±0.5): {accuracy_gru:.2f}%")
print(f"Custom SVD Accuracy (±0.5): {accuracy_svd:.2f}%")

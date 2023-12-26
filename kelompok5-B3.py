import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# Baca dataset
data = pd.read_csv('clothing.csv')

# Menghilangkan data yang tidak relevan
data = data[['Clothing ID', 'Age', 'Rating', 'classname']]

# Menghapus baris dengan nilai NaN
data = data.dropna(subset=['Age', 'Rating'])

# Pisahkan data menjadi data train dan test
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Inisialisasi model Nearest Neighbors
model = NearestNeighbors(metric='cosine', algorithm='brute')

# Training model
model = NearestNeighbors(n_neighbors=5, algorithm='auto')
model.fit(train[['Age', 'Rating']])


def recommend_clothing(age, k=5):
    # Cari tetangga terdekat
    distances, indices = model.kneighbors([[age, 5]], n_neighbors=k+1)
    
    # Ambil indeks dari tetangga terdekat (excluder pertama karena itu inputnya sendiri)
    indices = indices.squeeze()[1:]
    
    # Ambil rekomendasi dari tetangga terdekat berdasarkan rentang umur dan rating tinggi
    recommended_items = train.iloc[indices]
    high_rating_recommendations = recommended_items.copy()[((recommended_items['Age'] >= age - 1) & (recommended_items['Age'] <= age + 1)) | (recommended_items['Age'] == age) & ((recommended_items['Rating'] == 4) | (recommended_items['Rating'] >= 4))]
    
    return high_rating_recommendations

# pemanggilan fungsi
recommended_items = recommend_clothing(28, 50)
print(recommended_items)
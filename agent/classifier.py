import torch
import pandas as pd
import ast
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle


def extract_first_genre(genres_str):
    if pd.isna(genres_str):
        return "Unknown"
    try:
        genres = ast.literal_eval(genres_str)
        return genres[0]["name"] if genres else "Unknown"
    except Exception:
        return "Unknown"

class MovieDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Classif(nn.Module):
    def __init__(self, input_dim, num_classes,dropout = 0.5,hidden_dim=16):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)

ALLOWED_GENRES = {
    "Drama", "Comedy", "Action", "Horror", "Animation"
}

def extract_major_genre(genres_str):
    try:
        genres = ast.literal_eval(genres_str)
        for g in genres:
            if g["name"] in ALLOWED_GENRES:
                return g["name"]
        return None
    except:
        return None


if __name__=="__main__":

    df = pd.read_csv("movies_metadata.csv", low_memory=False)
    print(f"Longueur initiale : {len(df)}")

    # On garde uniquement les lignes exploitables
    df = df[["title", "overview","genres"]].dropna()
    df = df[df["overview"].str.len() > 20]

    df["label"] = df["genres"].apply(extract_major_genre)

    # Garder uniquement les films avec un genre autorisé
    df = df.dropna(subset=["label"])
    print(f"Longueur après crop : {len(df)}")


    # Textes et labels finaux
    texts = df["overview"].tolist()
    labels = df["label"].tolist()

    # Encoder les labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # TF-IDF pour transformer les textes en vecteurs
    vectorizer = TfidfVectorizer(max_features=3000)  # limiter la taille pour rapidité
    X = vectorizer.fit_transform(texts).toarray()

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    X_train,x_val,y_train,y_val = train_test_split(X,y,test_size=0.4,shuffle=True)
    dataset = MovieDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    dataset_val = MovieDataset(x_val, y_val)
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True)

    model = Classif(input_dim=X.shape[1], num_classes=len(label_encoder.classes_))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    epochs = 15 
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for batch_X, batch_y in dataloader_val:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss_val = criterion(outputs, batch_y)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Loss val: {loss_val.item():.4f}")

    torch.save(model.state_dict(), "genre_model.pth")
    print("Modèle enregistré : genre_model.pth")

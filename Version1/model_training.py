# tennis_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import json # To save scaler means/scales and OHE categories

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
def load_and_prepare_data_french_open(path_to_csv):
    df = pd.read_csv(path_to_csv)

    # Filter for completed matches (if 'Comment' column exists and is reliable)
    if 'Comment' in df.columns:
        df = df[df['Comment'] == 'Completed'].copy()
    
    # Drop rows with NaNs in essential columns for features or target
    essential_cols_for_model = [
        'Wsets', 'Lsets', 'WRank', 'LRank', 'WPts', 'LPts', 
        'Best of', 'Surface', 'Round', 'Series', 'Court'
    ]
    df.dropna(subset=essential_cols_for_model, inplace=True)

    processed_rows = []
    for _, row in df.iterrows():
        # Player 1 = Winner, Player 2 = Loser
        processed_rows.append({
            'p1_rank': row['WRank'], 'p2_rank': row['LRank'],
            'p1_pts': row['WPts'], 'p2_pts': row['LPts'],
            'best_of': row['Best of'], # Column name from CSV
            'surface': row['Surface'],
            'round': row['Round'],
            'series_cat': str(row['Series']), # Directly use Series, ensure it's string
            'court_type': str(row['Court']), # Directly use Court, ensure it's string
            'p1_sets_won': int(row['Wsets']), 'p2_sets_won': int(row['Lsets'])
        })
        # Player 1 = Loser, Player 2 = Winner
        processed_rows.append({
            'p1_rank': row['LRank'], 'p2_rank': row['WRank'],
            'p1_pts': row['LPts'], 'p2_pts': row['WPts'],
            'best_of': row['Best of'],
            'surface': row['Surface'],
            'round': row['Round'],
            'series_cat': str(row['Series']),
            'court_type': str(row['Court']),
            'p1_sets_won': int(row['Lsets']), 'p2_sets_won': int(row['Wsets'])
        })

    data = pd.DataFrame(processed_rows)
    
    target_cols = ['p1_sets_won', 'p2_sets_won']
    y = data[target_cols].values.astype(np.float32)
    X_df = data.drop(columns=target_cols)

    numerical_features = ['p1_rank', 'p2_rank', 'p1_pts', 'p2_pts', 'best_of']
    # Categorical features from the frenchopen.csv that will be used.
    # These names must match the keys used when preparing input in Auswertung.ipynb.
    categorical_features = ['surface', 'round', 'series_cat', 'court_type']
    
    # Impute NaNs just in case, though we dropped earlier
    for col in numerical_features:
        median_val = X_df[col].median() 
        X_df[col].fillna(median_val, inplace=True)
        X_df[col] = X_df[col].astype(np.float32)

    for col in categorical_features:
        X_df[col] = X_df[col].astype(str).fillna('Missing_Cat_Val') 

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop' 
    )
    
    X_processed = preprocessor.fit_transform(X_df).astype(np.float32)
    
    # Save preprocessing info for Auswertung.ipynb
    onehot_categories_dict = {}
    for i, col_name in enumerate(categorical_features):
        onehot_categories_dict[col_name] = preprocessor.named_transformers_['cat'].categories_[i].tolist()

    preprocessing_info = {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'scaler_means': preprocessor.named_transformers_['num'].mean_.tolist(),
        'scaler_scales': preprocessor.named_transformers_['num'].scale_.tolist(),
        'onehot_categories': onehot_categories_dict,
        'feature_order': numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    }
    with open('preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=4)
        
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Model erstellen
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.25),
        Dense(32, activation='relu'),
        Dropout(0.25),
        Dense(16, activation='relu'),
        Dense(2, activation='linear'),  
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Training starten
def train_model(path_to_csv, model_save_path="trained_tennis_model.keras"):
    X_train, X_test, y_train, y_test = load_and_prepare_data_french_open(path_to_csv)
    
    print(f"Number of features for model: {X_train.shape[1]}")

    model = build_model(X_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, # Adjusted batch_size for smaller dataset
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping], verbose=1)

    model.save(model_save_path)
    print(f"Modell gespeichert unter: {model_save_path}")
    
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {mae:.4f}, Test MSE: {loss:.4f}")
    return model, history

if __name__ == "__main__":
    csv_file_path = 'frenchopen_2024.csv' # Use the frenchopen.csv
    train_model(csv_file_path)
    print("Training abgeschlossen. Preprocessing info saved to preprocessing_info.json")
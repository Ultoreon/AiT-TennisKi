# tennis_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import json # To save scaler means/scales and OHE categories

import tensorflow as tf
from tensorflow.keras.models import Model # Changed from Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda # Added Input, Model
# Removed Concatenate as tf.concat is used directly
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping

# Custom layer to enforce score constraints
class EnforceScoreConstraintLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        raw_scores, best_of_unscaled = inputs # raw_scores: (batch, 2), best_of_unscaled: (batch, 1)

        # Ensure best_of_unscaled is float for calculations
        best_of_unscaled = tf.cast(best_of_unscaled, tf.float32)

        # Calculate S_win (sets needed to win) for each sample in the batch
        S_win = tf.math.ceil(best_of_unscaled / 2.0) # Shape: (batch, 1)
        max_loser_sets = S_win - 1.0 # Shape: (batch, 1)

        o1 = raw_scores[:, 0:1] # Shape: (batch, 1), raw prediction for P1
        o2 = raw_scores[:, 1:2] # Shape: (batch, 1), raw prediction for P2

        # Determine the winner based on raw predictions o1, o2
        # p1_wins_condition is true if o1 >= o2. P1 wins ties.
        p1_wins_condition = o1 >= o2
        p1_wins = tf.cast(p1_wins_condition, tf.float32)
        # p2_wins is true if o1 < o2.
        p2_wins = tf.cast(tf.logical_not(p1_wins_condition), tf.float32)

        # Predict loser's score: use relu on raw score (to make it non-negative)
        # then clip it by max_loser_sets.
        o1_pos_for_loser_score = tf.nn.relu(o1) # P1's potential score if P1 loses
        o2_pos_for_loser_score = tf.nn.relu(o2) # P2's potential score if P2 loses
        
        # Candidate score for P1 if P1 loses (non-negative and clipped)
        s1_as_loser = tf.clip_by_value(o1_pos_for_loser_score, 0.0, max_loser_sets)
        # Candidate score for P2 if P2 loses (non-negative and clipped)
        s2_as_loser = tf.clip_by_value(o2_pos_for_loser_score, 0.0, max_loser_sets)

        # Final scores: winner gets S_win, loser gets their candidate score
        final_s1 = p1_wins * S_win        + p2_wins * s1_as_loser
        final_s2 = p1_wins * s2_as_loser  + p2_wins * S_win
        
        # Concatenate to form the output tensor (batch_size, 2)
        constrained_scores = tf.concat([final_s1, final_s2], axis=1)
        return constrained_scores

    def get_config(self):
        # Required for saving/loading models with custom layers
        config = super().get_config()
        return config

def load_and_prepare_data_french_open(path_to_csv):
    df = pd.read_csv(path_to_csv)

    if 'Comment' in df.columns:
        df = df[df['Comment'] == 'Completed'].copy()
    
    essential_cols_for_model = [
        'Wsets', 'Lsets', 'WRank', 'LRank', 'WPts', 'LPts', 
        'Best of', 'Surface', 'Round', 'Series', 'Court'
    ]
    df.dropna(subset=essential_cols_for_model, inplace=True)

    processed_rows = []
    for _, row in df.iterrows():
        try:
            best_of_val = int(row['Best of'])
        except ValueError:
            print(f"Warning: Could not parse 'Best of' value: {row['Best of']} for a row. Skipping.")
            continue
        # Ensure sets are integers
        try:
            wsets_val = int(row['Wsets'])
            lsets_val = int(row['Lsets'])
        except ValueError:
            print(f"Warning: Could not parse 'Wsets' or 'Lsets' for a row. Skipping.")
            continue

        processed_rows.append({
            'p1_rank': row['WRank'], 'p2_rank': row['LRank'],
            'p1_pts': row['WPts'], 'p2_pts': row['LPts'],
            'best_of': best_of_val, 
            'surface': row['Surface'],
            'round': row['Round'],
            'series_cat': str(row['Series']), 
            'court_type': str(row['Court']), 
            'p1_sets_won': wsets_val, 'p2_sets_won': lsets_val
        })
        processed_rows.append({
            'p1_rank': row['LRank'], 'p2_rank': row['WRank'],
            'p1_pts': row['LPts'], 'p2_pts': row['WPts'],
            'best_of': best_of_val,
            'surface': row['Surface'],
            'round': row['Round'],
            'series_cat': str(row['Series']),
            'court_type': str(row['Court']),
            'p1_sets_won': lsets_val, 'p2_sets_won': wsets_val
        })

    if not processed_rows:
        raise ValueError("No data rows were processed. Check CSV content, filtering, and 'Best of' parsing.")

    data = pd.DataFrame(processed_rows)
    
    target_cols = ['p1_sets_won', 'p2_sets_won']
    y = data[target_cols].values.astype(np.float32)
    X_df = data.drop(columns=target_cols)

    # Extract unscaled 'best_of' for the custom layer input BEFORE it's scaled
    best_of_unscaled = X_df['best_of'].values.astype(np.float32).reshape(-1, 1)

    numerical_features = ['p1_rank', 'p2_rank', 'p1_pts', 'p2_pts', 'best_of']
    categorical_features = ['surface', 'round', 'series_cat', 'court_type']
    
    for col in numerical_features:
        median_val = X_df[col].median() 
        X_df[col].fillna(median_val, inplace=True)
        X_df[col] = X_df[col].astype(np.float32) # Ensure type after imputation

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
    
    onehot_categories_dict = {}
    cat_transformer = preprocessor.named_transformers_['cat']
    if hasattr(cat_transformer, 'get_feature_names_out'): # For newer scikit-learn
        ohe_feature_names = list(cat_transformer.get_feature_names_out(categorical_features))
    else: # For older scikit-learn
        ohe_feature_names = list(cat_transformer.get_feature_names(categorical_features))
        
    for i, col_name in enumerate(categorical_features):
        onehot_categories_dict[col_name] = cat_transformer.categories_[i].tolist()

    preprocessing_info2 = {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'scaler_means': preprocessor.named_transformers_['num'].mean_.tolist(),
        'scaler_scales': preprocessor.named_transformers_['num'].scale_.tolist(),
        'onehot_categories': onehot_categories_dict,
        'feature_order': numerical_features + ohe_feature_names
    }
    with open('preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info2, f, indent=4)
        
    # Split all three: X_processed, y, and best_of_unscaled
    X_train, X_test, y_train, y_test, best_of_train, best_of_test = train_test_split(
        X_processed, y, best_of_unscaled, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, best_of_train, best_of_test

# Model erstellen (Functional API)
def build_model_functional(input_dim_features, best_of_index=4):
    input_features = Input(shape=(input_dim_features,), name='input_features')
    x = Dense(64, activation='relu')(input_features)
    x = Dropout(0.25)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(16, activation='relu')(x)
    raw_scores = Dense(2, activation='linear', name='raw_scores')(x)

    def enforce_score_constraint(raw_scores):
        # Extract best_of from the input features (assume it's at index 4)
        best_of = tf.expand_dims(input_features[:, best_of_index], axis=1)
        best_of = tf.cast(best_of, tf.float32)
        S_win = tf.math.ceil(best_of / 2.0)
        max_loser_sets = S_win - 1.0

        o1 = raw_scores[:, 0:1]
        o2 = raw_scores[:, 1:2]
        p1_wins = tf.cast(o1 >= o2, tf.float32)
        p2_wins = 1.0 - p1_wins

        s1_as_loser = tf.clip_by_value(tf.nn.relu(o1), 0.0, max_loser_sets)
        s2_as_loser = tf.clip_by_value(tf.nn.relu(o2), 0.0, max_loser_sets)

        final_s1 = p1_wins * S_win + p2_wins * s1_as_loser
        final_s2 = p1_wins * s2_as_loser + p2_wins * S_win
        return tf.concat([final_s1, final_s2], axis=1)

    constrained_scores = Lambda(enforce_score_constraint, name="enforce_score_constraint")(raw_scores)
    model = Model(inputs=input_features, outputs=constrained_scores)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Training starten
def train_model(path_to_csv, model_save_path="trained_tennis_model2.keras"):
    # Load data, now including best_of_unscaled splits
    X_train, X_test, y_train, y_test, best_of_train, best_of_test = load_and_prepare_data_french_open(path_to_csv)
    
    print(f"Number of features for model (main input): {X_train.shape[1]}")

    model = build_model_functional(X_train.shape[1])
    model.summary() # Print model structure

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    # Pass inputs as a list: [X_train, best_of_train]
    history = model.fit(
        [X_train, best_of_train], 
        y_train, 
        epochs=100, 
        batch_size=16,
        validation_data=([X_test, best_of_test], y_test), # Validation data also as a list
        callbacks=[early_stopping], 
        verbose=1
    )

    model.save(model_save_path)
    print(f"Modell gespeichert unter: {model_save_path}")
    
    # Evaluate using the list input format
    loss, mae = model.evaluate([X_test, best_of_test], y_test, verbose=0)
    print(f"Test MAE: {mae:.4f}, Test MSE: {loss:.4f}")

    # Example prediction to check output structure
    if len(X_test) > 0:
        print("\nExample prediction structure check:")
        sample_pred_input = [X_test[:1], best_of_test[:1]]
        sample_pred_output = model.predict(sample_pred_input)
        
        input_bo = best_of_test[:1][0,0] # Get the 'best_of' for this sample
        pred_s1_float = sample_pred_output[0,0]
        pred_s2_float = sample_pred_output[0,1]
        
        print(f"Input 'best_of': {input_bo}")
        print(f"Predicted scores (float): p1_sets={pred_s1_float:.2f}, p2_sets={pred_s2_float:.2f}")
        
        # For interpretation, scores would typically be rounded
        s1_rounded = round(pred_s1_float)
        s2_rounded = round(pred_s2_float)
        print(f"Rounded predicted scores: p1_sets={s1_rounded}, p2_sets={s2_rounded}")
        
        S_win_sample = np.ceil(input_bo / 2.0)
        max_loser_sets_sample = S_win_sample - 1
        
        is_valid_structure = False
        # Check if one player has S_win_sample and the other has <= max_loser_sets_sample
        # Using float predictions directly from layer for this check as rounding is just for display
        # The layer ensures this structure for its float outputs.
        if (abs(pred_s1_float - S_win_sample) < 1e-3 and pred_s2_float <= max_loser_sets_sample + 1e-3 and pred_s2_float >= -1e-3) or \
           (abs(pred_s2_float - S_win_sample) < 1e-3 and pred_s1_float <= max_loser_sets_sample + 1e-3 and pred_s1_float >= -1e-3) :
            is_valid_structure = True
        print(f"Is score structure valid (one player wins S_win, other <= S_win-1 and >=0 sets, using float outputs)? {is_valid_structure}")
        print(f"Expected S_win: {S_win_sample}, Max loser sets: {max_loser_sets_sample}")

    return model, history

if __name__ == "__main__":
    # Create a dummy frenchopen_2024.csv for testing if it doesn't exist
    csv_file_path = 'frenchopen_2024.csv'
    try:
        # Attempt to read, if it fails, create dummy data
        pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"{csv_file_path} not found, creating a dummy CSV for demonstration.")
        dummy_data_list = []
        for i in range(50): # Create 50 match records
            best_of = np.random.choice([3,5])
            s_win = int(np.ceil(best_of / 2.0))
            w_sets = s_win
            l_sets = np.random.randint(0, s_win) # Loser sets from 0 to S_win-1
            
            dummy_data_list.append({
                'Comment': 'Completed',
                'Wsets': w_sets, 'Lsets': l_sets,
                'WRank': np.random.randint(1, 150), 'LRank': np.random.randint(1, 150),
                'WPts': np.random.randint(500, 10000), 'LPts': np.random.randint(300, 8000),
                'Best of': best_of,
                'Surface': np.random.choice(['Clay', 'Hard', 'Grass']),
                'Round': np.random.choice(['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']),
                'Series': np.random.choice(['Grand Slam', 'ATP 250', 'ATP 500', 'Masters 1000']),
                'Court': np.random.choice(['Outdoor', 'Indoor']),
                'Tournament': 'Dummy Open', 'Date': f'2024-01-{np.random.randint(1,29)}',
                'Winner': f'PlayerW{i}', 'Loser': f'PlayerL{i}'
            })
        dummy_df = pd.DataFrame(dummy_data_list)
        # Add a few rows with NaNs or non-completed to test filtering
        dummy_df.loc[len(dummy_df)] = dummy_df.loc[0].copy()
        dummy_df.loc[len(dummy_df)-1, 'Comment'] = 'Scheduled'
        dummy_df.loc[len(dummy_df)] = dummy_df.loc[1].copy()
        dummy_df.loc[len(dummy_df)-1, 'WRank'] = np.nan 
        
        dummy_df.to_csv(csv_file_path, index=False)
        print(f"Dummy {csv_file_path} created.")

    train_model(csv_file_path)
    print("Training abgeschlossen. Preprocessing info saved to preprocessing_info.json")
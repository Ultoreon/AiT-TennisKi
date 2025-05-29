import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf

# --- 1. Daten laden und grundlegend bereinigen ---
try:
    df = pd.read_csv('2024.csv', sep=';', decimal=',')
except UnicodeDecodeError:
    df = pd.read_csv('2024.csv', sep=';', decimal=',', encoding='latin1')

print(f"Ursprüngliche Datenmenge: {len(df)}")

rank_cols = ['WRank', 'LRank']
pts_cols = ['WPts', 'LPts']
set_cols = ['Wsets', 'Lsets'] # Nur Gesamt-Sätze relevant

for col in rank_cols + pts_cols + set_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

df = df[df['Comment'] == 'Completed'].copy()
essential_cols = ['WRank', 'LRank', 'WPts', 'LPts', 'Wsets', 'Lsets', 'Surface', 'Best of']
df.dropna(subset=essential_cols, inplace=True)
df['Best of'] = pd.to_numeric(df['Best of'], errors='coerce')
df.dropna(subset=['Best of'], inplace=True)
df['Best of'] = df['Best of'].astype(int)
print(f"Datenmenge nach Bereinigung: {len(df)}")

# --- 2. Feature Engineering & Daten Augmentation (Fokus Bo5) ---
data_for_df = []

# Nur Best-of-5 Spiele für Training und Vorhersage-Logik
possible_scores_bo5 = [(3,0), (3,1), (3,2), (0,3), (1,3), (2,3)]
score_to_label_map_training = {score: i for i, score in enumerate(possible_scores_bo5)}
label_to_score_map_training = {i: score for i, score in enumerate(possible_scores_bo5)}
N_CLASSES = len(possible_scores_bo5)

print(f"Score to Label Map (Bo5): {score_to_label_map_training}")
print(f"Anzahl Klassen: {N_CLASSES}")

df_bo5 = df[df['Best of'] == 5].copy()
print(f"Anzahl Best-of-5 Spiele im Datensatz: {len(df_bo5)}")

for index, row in df_bo5.iterrows():
    w_rank = row['WRank']
    l_rank = row['LRank']
    w_pts = row['WPts']
    l_pts = row['LPts']
    w_sets = int(row['Wsets'])
    l_sets = int(row['Lsets'])
    surface = row['Surface']
    # best_of ist hier immer 5

    # Überprüfen, ob das Satzergebnis gültig ist für Bo5
    if not ((w_sets == 3 and l_sets < 3) or (l_sets == 3 and w_sets < 3)):
        # print(f"Ungültiges Bo5 Ergebnis übersprungen: W {w_sets} L {l_sets}")
        continue

    score_tuple_winner_first = (w_sets, l_sets)
    score_tuple_loser_first = (l_sets, w_sets)

    if score_tuple_winner_first not in score_to_label_map_training:
        # Sollte nicht passieren nach obiger Prüfung, aber sicher ist sicher
        continue

    # Sample 1: P1=Winner, P2=Loser
    data_for_df.append({
        'P1_Rank': w_rank, 'P1_Pts': w_pts,
        'P2_Rank': l_rank, 'P2_Pts': l_pts,
        'Surface': surface, 'Best_of': 5.0, # Als float für den Scaler
        'Target_Label': score_to_label_map_training[score_tuple_winner_first]
    })

    # Sample 2: P1=Loser, P2=Winner (Augmentation)
    data_for_df.append({
        'P1_Rank': l_rank, 'P1_Pts': l_pts,
        'P2_Rank': w_rank, 'P2_Pts': w_pts,
        'Surface': surface, 'Best_of': 5.0,
        'Target_Label': score_to_label_map_training[score_tuple_loser_first]
    })

if not data_for_df:
    raise ValueError("Keine passenden Bo5-Daten nach der Filterung gefunden.")

Xy_df = pd.DataFrame(data_for_df)
print(f"Anzahl aufbereiteter Samples für Bo5: {len(Xy_df)}")

# --- 3. Datenaufbereitung für das Modell ---
feature_columns = ['P1_Rank', 'P1_Pts', 'P2_Rank', 'P2_Pts', 'Surface', 'Best_of']
X_df = Xy_df[feature_columns]
y_array = Xy_df['Target_Label'].values

numerical_features_train = ['P1_Rank', 'P1_Pts', 'P2_Rank', 'P2_Pts', 'Best_of']
categorical_features_train = ['Surface']

# Preprocessor
# Wichtig: sparse_output=False für dichte Ausgabe, die Dense Layer erwartet
preprocessor_train = ColumnTransformer([
    ('numerical', MinMaxScaler(), numerical_features_train),
    ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_train)
], remainder='passthrough')

X_train_df, X_temp_df, y_train_arr, y_temp_arr = train_test_split(
    X_df, y_array, test_size=0.3, random_state=42, stratify=y_array if len(np.unique(y_array)) > 1 else None
)
X_val_df, X_test_df, y_val_arr, y_test_arr = train_test_split(
    X_temp_df, y_temp_arr, test_size=0.5, random_state=42, stratify=y_temp_arr if len(np.unique(y_temp_arr)) > 1 else None
)

X_train_processed_arr = preprocessor_train.fit_transform(X_train_df)
X_val_processed_arr = preprocessor_train.transform(X_val_df)
X_test_processed_arr = preprocessor_train.transform(X_test_df)

print(f"Shape von X_train_processed_arr: {X_train_processed_arr.shape}")
if X_train_processed_arr.shape[0] == 0:
    raise ValueError("X_train_processed_arr ist leer.")

# --- 4. Modelltraining mit Tensorflow/Keras ---
input_shape_model = (X_train_processed_arr.shape[1],) # korrigierter Variablenname

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape_model),
    tf.keras.layers.Dropout(0.3), # Angepasst
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2), # Angepasst
    tf.keras.layers.Dense(N_CLASSES, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True) # Mehr Geduld
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_tennis_model.keras', save_best_only=True, monitor='val_loss')

history = model.fit(
    X_train_processed_arr, y_train_arr,
    epochs=100, # Erhöht, da EarlyStopping aktiv
    batch_size=32, # Angepasst
    validation_data=(X_val_processed_arr, y_val_arr),
    callbacks=[early_stopping, model_checkpoint]
)

model = tf.keras.models.load_model('best_tennis_model.keras')
loss, accuracy = model.evaluate(X_test_processed_arr, y_test_arr)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- 5. Speichern des Modells ---
model.save('tennis_set_predictor_final.keras')
print("\nModell als 'tennis_set_predictor_final.keras' gespeichert.")


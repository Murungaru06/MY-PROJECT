"""
train_model.py
──────────────
Run this ONCE to train the ANN and save:
  - student_performance_ann.keras
  - scaler.pkl
  - label_encoders.pkl

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import random, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# ── Reproducibility ────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ── Load Dataset ────────────────────────────────────────────────
# Update path if needed
df = pd.read_csv("STUDENT_PERFORMANCE__1_.csv")
print(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

# ── Target: Risk Level ──────────────────────────────────────────
def classify_risk(g):
    if g < 10:  return 0   # At Risk
    elif g < 14: return 1  # Average
    else:        return 2  # High Performer

df['risk_level'] = df['G3'].apply(classify_risk)

# ── Encode Categorical Columns ──────────────────────────────────
df_model = df.copy()
cat_cols = [c for c in df_model.columns
            if 'str' in str(df_model[c].dtype).lower() or df_model[c].dtype == object]

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

df_model = df_model.apply(pd.to_numeric, errors='coerce').fillna(0)

# ── Features / Target ───────────────────────────────────────────
X = df_model.drop(columns=['G3', 'risk_level']).astype(np.float32)
y = df_model['risk_level'].astype(int)

feature_names = list(X.columns)
print(f"Features ({len(feature_names)}): {feature_names}")

# ── Train / Val / Test Split ────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X.values, y.values, test_size=0.30, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── Scale ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

NUM_CLASSES = 3
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat   = to_categorical(y_val,   NUM_CLASSES)

# ── Build ANN ────────────────────────────────────────────────────
input_dim = X_train_s.shape[1]

model = Sequential(name='Student_Performance_ANN')
model.add(tf.keras.Input(shape=(input_dim,)))

model.add(Dense(256, activation='relu', name='hidden_1'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu', name='hidden_2'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu', name='hidden_3'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu', name='hidden_4'))
model.add(Dropout(0.2))

model.add(Dense(NUM_CLASSES, activation='softmax', name='output_layer'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Train ────────────────────────────────────────────────────────
early_stop = EarlyStopping(monitor='val_loss', patience=20,
                           restore_best_weights=True, verbose=1)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=8, min_lr=1e-6, verbose=1)

print("\n🚀 Training started …")
history = model.fit(
    X_train_s, y_train_cat,
    validation_data=(X_val_s, y_val_cat),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ── Evaluate ─────────────────────────────────────────────────────
y_pred = np.argmax(model.predict(X_test_s, verbose=0), axis=1)
acc    = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred,
      target_names=['At Risk', 'Average', 'High Performer']))

# ── Save Artefacts ───────────────────────────────────────────────
model.save('student_performance_ann.keras')
joblib.dump(scaler,          'scaler.pkl')
joblib.dump(label_encoders,  'label_encoders.pkl')
joblib.dump(feature_names,   'feature_names.pkl')

print("\n✅ Saved:")
print("   student_performance_ann.keras")
print("   scaler.pkl")
print("   label_encoders.pkl")
print("   feature_names.pkl")
print("\nRun:  streamlit run app.py")

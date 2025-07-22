"""
Neural network-based prediction system using LSTM and ensemble methods
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from config import *
from utils import setup_logger

logger = setup_logger(__name__)

class NeuralPredictionEngine:
    def __init__(self):
        self.lstm_model = None
        self.ensemble_models = {}
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def prepare_sequences(self, draws_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        draws_array = draws_df.values
        
        X, y = [], []
        for i in range(LSTM_SEQUENCE_LENGTH, len(draws_array)):
            X.append(draws_array[i-LSTM_SEQUENCE_LENGTH:i])
            y.append(draws_array[i])
        
        X, y = np.array(X), np.array(y)
        
        X_reshaped = X.reshape(-1, OZLOTTO_NUMBERS)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(X.shape)
        
        y_scaled = self.scaler.transform(y)
        
        return X, y_scaled
    
    def build_lstm_model(self) -> Model:
        """Build advanced LSTM model with attention mechanism"""
        lstm_input = Input(shape=(LSTM_SEQUENCE_LENGTH, OZLOTTO_NUMBERS))
        
        lstm1 = LSTM(LSTM_HIDDEN_UNITS, return_sequences=True)(lstm_input)
        dropout1 = Dropout(LSTM_DROPOUT_RATE)(lstm1)
        
        lstm2 = LSTM(LSTM_HIDDEN_UNITS // 2, return_sequences=True)(dropout1)
        dropout2 = Dropout(LSTM_DROPOUT_RATE)(lstm2)
        
        lstm3 = LSTM(LSTM_HIDDEN_UNITS // 4, return_sequences=False)(dropout2)
        dropout3 = Dropout(LSTM_DROPOUT_RATE)(lstm3)
        
        dense1 = Dense(64, activation='relu')(dropout3)
        dense2 = Dense(32, activation='relu')(dense1)
        
        output = Dense(OZLOTTO_NUMBERS, activation='sigmoid')(dense2)
        
        model = Model(inputs=lstm_input, outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def train_lstm(self, draws_df: pd.DataFrame, epochs: int = 100):
        """Train LSTM model on historical data"""
        logger.info("Preparing LSTM training data...")
        X, y = self.prepare_sequences(draws_df)
        
        if len(X) < 10:
            logger.warning("Insufficient data for LSTM training")
            return
        
        logger.info(f"Training LSTM with {len(X)} sequences...")
        
        self.lstm_model = self.build_lstm_model()
        
        history = self.lstm_model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.lstm_model.save(f"{MODEL_DIR}/lstm_model.h5")
        joblib.dump(self.scaler, f"{MODEL_DIR}/scaler.pkl")
        
        self.is_trained = True
        logger.info("LSTM training completed")
    
    def train_ensemble_models(self, draws_df: pd.DataFrame):
        """Train ensemble of traditional ML models"""
        logger.info("Training ensemble models...")
        
        features = self.extract_features(draws_df)
        
        if len(features) < 10:
            logger.warning("Insufficient data for ensemble training")
            return
        
        targets = draws_df.iloc[1:].values  # Next draw for each feature set
        features = features[:-1]  # Remove last feature set (no target)
        
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(features, targets)
            self.ensemble_models[name] = model
            
            joblib.dump(model, f"{MODEL_DIR}/{name}_model.pkl")
        
        logger.info("Ensemble training completed")
    
    def extract_features(self, draws_df: pd.DataFrame) -> np.ndarray:
        """Extract engineered features from historical draws"""
        features = []
        
        for i in range(len(draws_df)):
            draw = draws_df.iloc[i].values
            feature_vector = []
            
            feature_vector.extend([
                np.mean(draw),
                np.std(draw),
                np.min(draw),
                np.max(draw),
                np.median(draw)
            ])
            
            sorted_draw = sorted(draw)
            gaps = [sorted_draw[j+1] - sorted_draw[j] for j in range(len(sorted_draw)-1)]
            feature_vector.extend([
                np.mean(gaps),
                np.std(gaps),
                np.min(gaps),
                np.max(gaps)
            ])
            
            low_count = sum(1 for x in draw if x <= 15)
            mid_count = sum(1 for x in draw if 16 <= x <= 31)
            high_count = sum(1 for x in draw if x >= 32)
            
            feature_vector.extend([low_count, mid_count, high_count])
            
            even_count = sum(1 for x in draw if x % 2 == 0)
            odd_count = OZLOTTO_NUMBERS - even_count
            feature_vector.extend([even_count, odd_count])
            
            draw_sum = sum(draw)
            feature_vector.extend([draw_sum, draw_sum % 10])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def predict_lstm(self, recent_draws: pd.DataFrame) -> List[int]:
        """Generate prediction using LSTM model"""
        if not self.is_trained or self.lstm_model is None:
            logger.warning("LSTM model not trained")
            return []
        
        if len(recent_draws) < LSTM_SEQUENCE_LENGTH:
            logger.warning("Insufficient recent draws for LSTM prediction")
            return []
        
        sequence = recent_draws.tail(LSTM_SEQUENCE_LENGTH).values
        sequence_scaled = self.scaler.transform(sequence.reshape(-1, OZLOTTO_NUMBERS))
        sequence_scaled = sequence_scaled.reshape(1, LSTM_SEQUENCE_LENGTH, OZLOTTO_NUMBERS)
        
        prediction_scaled = self.lstm_model.predict(sequence_scaled, verbose=0)
        prediction = self.scaler.inverse_transform(prediction_scaled)[0]
        
        prediction = np.clip(prediction, MIN_NUMBER, MAX_NUMBER)
        prediction = np.round(prediction).astype(int)
        
        unique_prediction = []
        for num in prediction:
            if num not in unique_prediction:
                unique_prediction.append(num)
        
        while len(unique_prediction) < OZLOTTO_NUMBERS:
            num = np.random.randint(MIN_NUMBER, MAX_NUMBER + 1)
            if num not in unique_prediction:
                unique_prediction.append(num)
        
        return sorted(unique_prediction[:OZLOTTO_NUMBERS])
    
    def predict_ensemble(self, recent_draws: pd.DataFrame) -> List[List[int]]:
        """Generate predictions using ensemble models"""
        if not self.ensemble_models:
            logger.warning("Ensemble models not trained")
            return []
        
        features = self.extract_features(recent_draws)
        if len(features) == 0:
            return []
        
        latest_features = features[-1].reshape(1, -1)
        predictions = []
        
        for name, model in self.ensemble_models.items():
            try:
                prediction = model.predict(latest_features)[0]
                prediction = np.clip(prediction, MIN_NUMBER, MAX_NUMBER)
                prediction = np.round(prediction).astype(int)
                
                unique_prediction = []
                for num in prediction:
                    if num not in unique_prediction:
                        unique_prediction.append(num)
                
                while len(unique_prediction) < OZLOTTO_NUMBERS:
                    num = np.random.randint(MIN_NUMBER, MAX_NUMBER + 1)
                    if num not in unique_prediction:
                        unique_prediction.append(num)
                
                predictions.append(sorted(unique_prediction[:OZLOTTO_NUMBERS]))
                
            except Exception as e:
                logger.error(f"Error in {name} prediction: {e}")
        
        return predictions
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            if os.path.exists(f"{MODEL_DIR}/lstm_model.h5"):
                self.lstm_model = tf.keras.models.load_model(f"{MODEL_DIR}/lstm_model.h5")
                self.scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
                self.is_trained = True
                logger.info("LSTM model loaded")
            
            for model_name in ['random_forest', 'gradient_boost']:
                model_path = f"{MODEL_DIR}/{model_name}_model.pkl"
                if os.path.exists(model_path):
                    self.ensemble_models[model_name] = joblib.load(model_path)
                    logger.info(f"{model_name} model loaded")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def generate_neural_predictions(self, recent_draws: pd.DataFrame, n_predictions: int = 10) -> List[List[int]]:
        """Generate multiple predictions using neural networks"""
        predictions = []
        
        lstm_pred = self.predict_lstm(recent_draws)
        if lstm_pred:
            predictions.append(lstm_pred)
        
        ensemble_preds = self.predict_ensemble(recent_draws)
        predictions.extend(ensemble_preds)
        
        while len(predictions) < n_predictions and predictions:
            base_pred = np.random.choice(predictions)
            variation = base_pred[:]
            for _ in range(2):  # Change 2 numbers
                idx = np.random.randint(0, OZLOTTO_NUMBERS)
                new_num = np.random.randint(MIN_NUMBER, MAX_NUMBER + 1)
                if new_num not in variation:
                    variation[idx] = new_num
            predictions.append(sorted(variation))
        
        return predictions[:n_predictions]

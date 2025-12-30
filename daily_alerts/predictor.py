"""
Stock Prediction Model using RandomForest
Fast, practical model for daily predictions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from .data_fetcher import calculate_technical_indicators, prepare_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for stock prediction results"""
    ticker: str
    current_price: float
    predicted_direction: str  # 'UP' or 'DOWN'
    confidence: float
    signal: str  # 'STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL'
    short_term_prediction: float  # 1-day return prediction
    medium_term_prediction: float  # 5-day return prediction
    long_term_prediction: float  # 20-day return prediction
    rsi: float
    macd_signal: str
    trend: str
    sector: str
    market_cap: float


class StockPredictor:
    """Multi-horizon stock prediction model"""

    def __init__(self, config: Dict):
        self.config = config
        self.models = {}  # Horizon -> Model
        self.scalers = {}  # Horizon -> Scaler
        self.horizons = [1, 5, 20]  # Short, medium, long-term
        self.model_cache_path = Path(config.get('paths', {}).get('model_cache', 'models/cached'))
        self.model_cache_path.mkdir(parents=True, exist_ok=True)

        # Thresholds from config
        pred_config = config.get('prediction', {})
        self.strong_buy_threshold = pred_config.get('strong_buy_threshold', 0.7)
        self.buy_threshold = pred_config.get('buy_threshold', 0.55)
        self.sell_threshold = pred_config.get('sell_threshold', 0.45)
        self.strong_sell_threshold = pred_config.get('strong_sell_threshold', 0.3)

    def _create_labels(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """Create binary labels: 1 = price went up, 0 = price went down"""
        future_returns = df['Close'].shift(-horizon) / df['Close'] - 1
        return (future_returns > 0).astype(int)

    def _create_return_targets(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """Create return targets for regression"""
        return df['Close'].shift(-horizon) / df['Close'] - 1

    def train(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[int, float]:
        """Train models for all horizons on combined stock data"""
        logger.info("Training prediction models...")
        accuracies = {}

        for horizon in self.horizons:
            logger.info(f"Training {horizon}-day horizon model...")

            # Combine all stock data
            all_features = []
            all_labels = []

            for ticker, df in stock_data.items():
                features_df, feature_cols = prepare_features(df)
                if len(features_df) < 100:
                    continue

                labels = self._create_labels(df, horizon)
                labels = labels.loc[features_df.index]

                # Remove last 'horizon' rows (no future data)
                features_df = features_df.iloc[:-horizon]
                labels = labels.iloc[:-horizon]

                # Remove NaN labels
                valid_idx = ~labels.isna()
                features_df = features_df[valid_idx]
                labels = labels[valid_idx]

                all_features.append(features_df)
                all_labels.append(labels)

            if not all_features:
                logger.warning(f"No valid training data for {horizon}-day horizon")
                continue

            X = pd.concat(all_features)
            y = pd.concat(all_labels)

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    n_jobs=-1,
                    random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                scores.append(accuracy_score(y_val, y_pred))

            # Train final model on all data
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=42
            )
            model.fit(X_scaled, y)

            self.models[horizon] = model
            self.scalers[horizon] = scaler
            accuracies[horizon] = np.mean(scores)

            logger.info(f"  {horizon}-day model accuracy: {np.mean(scores):.1%}")

        return accuracies

    def predict(self, df: pd.DataFrame, ticker: str) -> Optional[PredictionResult]:
        """Generate prediction for a single stock"""
        if df is None or len(df) < 60:
            return None

        try:
            features_df, feature_cols = prepare_features(df)
            if len(features_df) < 1:
                return None

            latest_features = features_df.iloc[-1:].copy()
            latest_row = df.iloc[-1]

            predictions = {}
            confidences = {}

            for horizon in self.horizons:
                if horizon not in self.models:
                    continue

                scaler = self.scalers[horizon]
                model = self.models[horizon]

                X_scaled = scaler.transform(latest_features)
                pred_proba = model.predict_proba(X_scaled)[0]

                # Probability of going up
                up_prob = pred_proba[1] if len(pred_proba) > 1 else 0.5
                predictions[horizon] = up_prob
                confidences[horizon] = max(up_prob, 1 - up_prob)

            if not predictions:
                return None

            # Use short-term prediction for signal
            short_term_prob = predictions.get(1, 0.5)
            confidence = confidences.get(1, 0.5)

            # Determine signal
            if short_term_prob >= self.strong_buy_threshold:
                signal = "STRONG BUY"
                direction = "UP"
            elif short_term_prob >= self.buy_threshold:
                signal = "BUY"
                direction = "UP"
            elif short_term_prob <= self.strong_sell_threshold:
                signal = "STRONG SELL"
                direction = "DOWN"
            elif short_term_prob <= self.sell_threshold:
                signal = "SELL"
                direction = "DOWN"
            else:
                signal = "HOLD"
                direction = "NEUTRAL"

            # Get additional indicators
            rsi = latest_row.get('RSI', 50)
            macd = latest_row.get('MACD', 0)
            macd_signal_val = latest_row.get('MACD_signal', 0)

            if macd > macd_signal_val:
                macd_signal = "BULLISH"
            elif macd < macd_signal_val:
                macd_signal = "BEARISH"
            else:
                macd_signal = "NEUTRAL"

            # Determine trend
            sma5 = latest_row.get('SMA_5', latest_row['Close'])
            sma20 = latest_row.get('SMA_20', latest_row['Close'])
            if sma5 > sma20 * 1.02:
                trend = "UPTREND"
            elif sma5 < sma20 * 0.98:
                trend = "DOWNTREND"
            else:
                trend = "SIDEWAYS"

            return PredictionResult(
                ticker=ticker,
                current_price=float(latest_row['Close']),
                predicted_direction=direction,
                confidence=float(confidence),
                signal=signal,
                short_term_prediction=float(predictions.get(1, 0.5)),
                medium_term_prediction=float(predictions.get(5, 0.5)),
                long_term_prediction=float(predictions.get(20, 0.5)),
                rsi=float(rsi) if not pd.isna(rsi) else 50,
                macd_signal=macd_signal,
                trend=trend,
                sector=str(latest_row.get('sector', 'Unknown')),
                market_cap=float(latest_row.get('market_cap', 0))
            )

        except Exception as e:
            logger.error(f"Error predicting {ticker}: {e}")
            return None

    def predict_all(self, stock_data: Dict[str, pd.DataFrame]) -> List[PredictionResult]:
        """Generate predictions for all stocks"""
        results = []

        for ticker, df in stock_data.items():
            result = self.predict(df, ticker)
            if result:
                results.append(result)

        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def filter_signals(self, results: List[PredictionResult]) -> Dict[str, List[PredictionResult]]:
        """Filter and categorize results by signal type"""
        signals = {
            'STRONG BUY': [],
            'BUY': [],
            'HOLD': [],
            'SELL': [],
            'STRONG SELL': []
        }

        for result in results:
            signals[result.signal].append(result)

        # Sort each category by confidence
        for signal_type in signals:
            signals[signal_type].sort(key=lambda x: x.confidence, reverse=True)

        return signals

    def save_models(self, filename: str = "stock_models.pkl"):
        """Save trained models to disk"""
        save_path = self.model_cache_path / filename
        with open(save_path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'timestamp': datetime.now().isoformat()
            }, f)
        logger.info(f"Models saved to {save_path}")

    def load_models(self, filename: str = "stock_models.pkl") -> bool:
        """Load trained models from disk"""
        load_path = self.model_cache_path / filename
        if not load_path.exists():
            return False

        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
                self.models = data['models']
                self.scalers = data['scalers']
                logger.info(f"Models loaded from {load_path} (trained: {data.get('timestamp', 'unknown')})")
                return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False


if __name__ == "__main__":
    # Test the predictor
    from data_fetcher import get_sp500_tickers, fetch_all_stocks
    import yaml

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Fetch data for test stocks
    tickers = get_sp500_tickers()[:20]
    stock_data = fetch_all_stocks(tickers)

    # Train and predict
    predictor = StockPredictor(config)
    predictor.train(stock_data)

    results = predictor.predict_all(stock_data)
    signals = predictor.filter_signals(results)

    print("\n=== STRONG BUY Signals ===")
    for r in signals['STRONG BUY'][:5]:
        print(f"{r.ticker}: ${r.current_price:.2f} | Conf: {r.confidence:.1%} | RSI: {r.rsi:.0f}")

    print("\n=== STRONG SELL Signals ===")
    for r in signals['STRONG SELL'][:5]:
        print(f"{r.ticker}: ${r.current_price:.2f} | Conf: {r.confidence:.1%} | RSI: {r.rsi:.0f}")

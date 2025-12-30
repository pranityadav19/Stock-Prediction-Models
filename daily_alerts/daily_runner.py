#!/usr/bin/env python3
"""
Daily Stock Prediction Runner
Orchestrates data fetching, prediction, and email alerts
"""

import os
import sys
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from daily_alerts.data_fetcher import get_sp500_tickers, fetch_all_stocks
from daily_alerts.predictor import StockPredictor
from daily_alerts.email_sender import StockEmailSender

# Try to import Google Sheets tracker
try:
    from daily_alerts.google_sheets_tracker import StockSheetsTracker
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DailyStockRunner:
    """Main runner for daily stock predictions"""

    def __init__(self, config_path: str = None):
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.predictor = StockPredictor(self.config)
        self.email_sender = StockEmailSender(self.config)

        # Filters from config
        pred_config = self.config.get('prediction', {})
        self.min_volume = pred_config.get('min_volume', 1000000)
        self.min_price = pred_config.get('min_price', 5.0)
        self.max_stocks_per_signal = pred_config.get('max_stocks_per_signal', 10)

    def run(self, retrain: bool = False, test_mode: bool = False) -> bool:
        """Run the full daily analysis pipeline"""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info(f"Starting Daily Stock Analysis - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        try:
            # Step 1: Get S&P 500 tickers
            logger.info("\n📋 Fetching S&P 500 tickers...")
            tickers = get_sp500_tickers()

            if test_mode:
                tickers = tickers[:30]  # Test with 30 stocks
                logger.info(f"TEST MODE: Using {len(tickers)} stocks")

            # Step 2: Fetch stock data
            logger.info(f"\n📊 Fetching data for {len(tickers)} stocks...")
            stock_data = fetch_all_stocks(tickers, days=252, max_workers=15)
            logger.info(f"Successfully fetched data for {len(stock_data)} stocks")

            if not stock_data:
                logger.error("No stock data fetched! Aborting.")
                return False

            # Step 3: Filter stocks by volume and price
            filtered_data = {}
            for ticker, df in stock_data.items():
                latest = df.iloc[-1]
                avg_volume = df['Volume'].tail(20).mean()

                if latest['Close'] >= self.min_price and avg_volume >= self.min_volume:
                    filtered_data[ticker] = df

            logger.info(f"Filtered to {len(filtered_data)} stocks (price >= ${self.min_price}, volume >= {self.min_volume:,})")

            # Step 4: Load or train models
            model_loaded = False
            if not retrain:
                model_loaded = self.predictor.load_models()

            if not model_loaded or retrain:
                logger.info("\n🧠 Training prediction models...")
                accuracies = self.predictor.train(filtered_data)
                for horizon, acc in accuracies.items():
                    logger.info(f"  {horizon}-day model accuracy: {acc:.1%}")
                self.predictor.save_models()

            # Step 5: Generate predictions
            logger.info("\n🔮 Generating predictions...")
            all_results = self.predictor.predict_all(filtered_data)
            logger.info(f"Generated predictions for {len(all_results)} stocks")

            # Step 5b: Export to Google Sheets
            if SHEETS_AVAILABLE and not test_mode:
                logger.info("\n📊 Exporting to Google Sheets...")
                try:
                    sheets = StockSheetsTracker()
                    # Convert PredictionResult objects to dicts
                    pred_dicts = [
                        {
                            'ticker': r.ticker,
                            'signal': r.signal,
                            'confidence': r.confidence,
                            'current_price': r.current_price,
                            'rsi': r.rsi,
                            'trend': r.trend,
                            'macd_signal': r.macd_signal,
                        }
                        for r in all_results
                    ]
                    sheets.export_predictions(pred_dicts)
                    logger.info(f"Exported to: {sheets.get_spreadsheet_url()}")
                except Exception as e:
                    logger.warning(f"Failed to export to Google Sheets: {e}")

            # Step 6: Filter and categorize signals
            signals = self.predictor.filter_signals(all_results)

            # Log summary
            logger.info("\n📈 Signal Summary:")
            for signal_type, results in signals.items():
                if results:
                    logger.info(f"  {signal_type}: {len(results)} stocks")

            # Show top picks
            logger.info("\n🚀 Top STRONG BUY Picks:")
            for result in signals.get('STRONG BUY', [])[:5]:
                logger.info(f"  {result.ticker}: ${result.current_price:.2f} | "
                          f"Confidence: {result.confidence:.0%} | RSI: {result.rsi:.0f} | "
                          f"{result.trend}")

            logger.info("\n🔻 Top STRONG SELL Picks:")
            for result in signals.get('STRONG SELL', [])[:5]:
                logger.info(f"  {result.ticker}: ${result.current_price:.2f} | "
                          f"Confidence: {result.confidence:.0%} | RSI: {result.rsi:.0f} | "
                          f"{result.trend}")

            # Step 7: Send email alert
            if not test_mode:
                logger.info("\n📧 Sending email alerts...")
                email_sent = self.email_sender.send_alert(signals, all_results)
                if email_sent:
                    logger.info("Email alerts sent successfully!")
                else:
                    logger.warning("Failed to send email alerts")
            else:
                logger.info("\nTEST MODE: Skipping email send")

                # Save sample email for preview
                subject, html = self.email_sender.create_email_content(signals, all_results)
                sample_path = Path(__file__).parent / "sample_email.html"
                with open(sample_path, 'w') as f:
                    f.write(html)
                logger.info(f"Sample email saved to {sample_path}")

            # Done
            elapsed = datetime.now() - start_time
            logger.info(f"\n✅ Daily analysis completed in {elapsed.total_seconds():.1f} seconds")

            return True

        except Exception as e:
            logger.error(f"Error in daily runner: {e}", exc_info=True)
            return False


def main():
    parser = argparse.ArgumentParser(description='Daily Stock Prediction Runner')
    parser.add_argument('--retrain', action='store_true', help='Force retrain models')
    parser.add_argument('--test', action='store_true', help='Run in test mode (fewer stocks, no email)')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()

    runner = DailyStockRunner(config_path=args.config)
    success = runner.run(retrain=args.retrain, test_mode=args.test)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

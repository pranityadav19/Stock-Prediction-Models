"""
Results Tracker
Verifies stock predictions against actual returns using Google Sheets.
"""

import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Google Sheets tracker
try:
    from .google_sheets_tracker import StockSheetsTracker, GSPREAD_AVAILABLE
except ImportError:
    GSPREAD_AVAILABLE = False


class StockResultsTracker:
    """Tracks prediction results using Google Sheets as the data source."""

    def __init__(self):
        if not GSPREAD_AVAILABLE:
            raise ImportError("gspread not available. Run: pip install gspread google-auth")
        self.sheets = StockSheetsTracker()

    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Fetch current prices for a list of tickers."""
        prices = {}
        try:
            # Batch download for efficiency
            data = yf.download(tickers, period='1d', progress=False)
            if 'Close' in data.columns:
                # Single ticker
                if len(tickers) == 1:
                    prices[tickers[0]] = float(data['Close'].iloc[-1])
                else:
                    for ticker in tickers:
                        if ticker in data['Close'].columns:
                            price = data['Close'][ticker].iloc[-1]
                            if not pd.isna(price):
                                prices[ticker] = float(price)
            elif isinstance(data.columns, pd.MultiIndex):
                # Multiple tickers
                for ticker in tickers:
                    try:
                        price = data['Close'][ticker].iloc[-1]
                        if not pd.isna(price):
                            prices[ticker] = float(price)
                    except:
                        pass
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")

        return prices

    def track_results(self, prediction_date: str, horizon_days: int = 5) -> Dict[str, Any]:
        """
        Track results for predictions made on a specific date.

        Args:
            prediction_date: Date when predictions were made (YYYY-MM-DD)
            horizon_days: Number of trading days to check (default 5 = 1 week)

        Returns:
            Results summary
        """
        # Get predictions from Google Sheets
        spreadsheet = self.sheets.get_or_create_spreadsheet()
        if not spreadsheet:
            logger.error("Could not access Google Sheets")
            return None

        sheet = spreadsheet.sheet1
        all_values = sheet.get_all_values()

        if len(all_values) <= 1:
            logger.warning("No predictions in Google Sheets")
            return None

        # Filter to predictions for the target date that haven't been evaluated yet
        predictions = []
        row_indices = []  # Track row numbers for updating
        for i, row in enumerate(all_values[1:], start=2):  # Skip header
            if len(row) < 5:
                continue
            row_date = row[0]
            ticker = row[1]
            signal = row[2]

            # Skip if already evaluated (Result column is not empty)
            if len(row) >= 12 and row[11]:  # Result column
                continue

            if row_date == prediction_date and signal != 'HOLD':
                predictions.append({
                    'ticker': ticker,
                    'signal': signal,
                    'confidence': float(row[3]) / 100 if row[3] else 0.5,
                    'current_price': float(row[4]) if row[4] else 0,
                })
                row_indices.append(i)

        if not predictions:
            logger.warning(f"No unevaluated predictions found for {prediction_date}")
            return None

        logger.info(f"Tracking {len(predictions)} predictions from {prediction_date}")

        # Get current prices
        tickers = [p['ticker'] for p in predictions]
        current_prices = self.get_current_prices(tickers)

        # Calculate results
        results = []
        correct = 0
        total = 0

        for pred, row_idx in zip(predictions, row_indices):
            ticker = pred['ticker']
            if ticker not in current_prices:
                continue

            entry_price = pred['current_price']
            current_price = current_prices[ticker]

            if entry_price == 0:
                continue

            pct_change = (current_price - entry_price) / entry_price * 100

            # Determine if prediction was correct
            signal = pred['signal']
            if signal in ['STRONG BUY', 'BUY']:
                is_correct = pct_change > 0
                expected_direction = 'UP'
            else:  # SELL, STRONG SELL
                is_correct = pct_change < 0
                expected_direction = 'DOWN'

            actual_direction = 'UP' if pct_change > 0 else 'DOWN'

            result = {
                'ticker': ticker,
                'signal': signal,
                'confidence': pred['confidence'],
                'entry_price': entry_price,
                'current_price': current_price,
                'pct_change': round(pct_change, 2),
                'expected_direction': expected_direction,
                'actual_direction': actual_direction,
                'is_correct': is_correct,
                'row_idx': row_idx,
            }
            results.append(result)

            total += 1
            if is_correct:
                correct += 1

        # Update Google Sheets with results
        logger.info("Updating Google Sheets with results...")
        self.sheets.update_results(prediction_date, results)

        # Calculate accuracy by signal type
        by_signal = {}
        for signal_type in ['STRONG BUY', 'BUY', 'SELL', 'STRONG SELL']:
            signal_results = [r for r in results if r['signal'] == signal_type]
            if signal_results:
                signal_correct = sum(1 for r in signal_results if r['is_correct'])
                by_signal[signal_type] = {
                    'total': len(signal_results),
                    'correct': signal_correct,
                    'accuracy': round(signal_correct / len(signal_results) * 100, 1)
                }

        # Create summary
        summary = {
            'prediction_date': prediction_date,
            'evaluation_date': datetime.now().strftime('%Y-%m-%d'),
            'horizon_days': horizon_days,
            'total_tracked': total,
            'correct': correct,
            'accuracy': round(correct / total * 100, 1) if total > 0 else 0,
            'by_signal': by_signal,
            'results': results
        }

        self._print_summary(summary)

        return summary

    def _print_summary(self, summary: Dict):
        """Print results summary to console."""
        print("\n" + "=" * 60)
        print(f"STOCK PREDICTION RESULTS - {summary['prediction_date']}")
        print("=" * 60)
        print(f"\nOverall Accuracy: {summary['correct']}/{summary['total_tracked']} ({summary['accuracy']}%)")

        print("\nBy Signal Type:")
        for signal, stats in summary.get('by_signal', {}).items():
            print(f"  {signal}: {stats['correct']}/{stats['total']} ({stats['accuracy']}%)")

        # Top winners and losers
        results = summary.get('results', [])
        if results:
            sorted_by_change = sorted(results, key=lambda x: x['pct_change'], reverse=True)

            print("\nTop Gainers:")
            for r in sorted_by_change[:3]:
                status = "HIT" if r['is_correct'] else "MISS"
                print(f"  {r['ticker']}: {r['pct_change']:+.1f}% ({r['signal']}) [{status}]")

            print("\nTop Losers:")
            for r in sorted_by_change[-3:]:
                status = "HIT" if r['is_correct'] else "MISS"
                print(f"  {r['ticker']}: {r['pct_change']:+.1f}% ({r['signal']}) [{status}]")

        print("=" * 60)

    def get_historical_performance(self) -> Dict[str, Any]:
        """Get aggregate performance from Google Sheets."""
        return self.sheets.get_performance_summary()


def main():
    """Main entry point for results tracking."""
    parser = argparse.ArgumentParser(description='Stock Prediction Results Tracker')
    parser.add_argument('date', nargs='?', help='Prediction date to track (YYYY-MM-DD)')
    parser.add_argument('--horizon', type=int, default=5, help='Trading days to wait (default: 5)')
    parser.add_argument('--historical', action='store_true', help='Show historical performance')
    args = parser.parse_args()

    tracker = StockResultsTracker()

    if args.historical:
        perf = tracker.get_historical_performance()
        if perf and perf.get('total', 0) > 0:
            print("\n" + "=" * 60)
            print("HISTORICAL STOCK PREDICTION PERFORMANCE")
            print("=" * 60)
            print(f"\nTotal Predictions: {perf['total']}")
            print(f"Overall Accuracy: {perf['hits']}/{perf['total']} ({perf['accuracy']}%)")
            print("\nBy Signal Type:")
            for signal, stats in perf.get('by_signal', {}).items():
                print(f"  {signal}: {stats['hits']}/{stats['total']} ({stats['accuracy']}%)")
            print("=" * 60)
        else:
            print("No historical results found")
        return

    # Track specific date
    if args.date:
        date = args.date
    else:
        # Default to 7 days ago (~5 trading days)
        target = datetime.now() - timedelta(days=7)
        date = target.strftime('%Y-%m-%d')

    tracker.track_results(date, horizon_days=args.horizon)


if __name__ == '__main__':
    main()

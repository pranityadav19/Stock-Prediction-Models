"""
Google Sheets Tracker for Stock Predictions
Exports predictions and results to Google Sheets for tracking.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockSheetsTracker:
    """Exports stock predictions to Google Sheets for tracking."""

    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    # Reuse credentials from NBA project
    CREDENTIALS_DIR = Path("/Users/pranityadav/Downloads/NBA-Betting/prizepicks_integration/credentials")

    def __init__(self, spreadsheet_name: str = "Stock Prediction Tracker"):
        """Initialize the tracker."""
        if not GSPREAD_AVAILABLE:
            raise ImportError("gspread not installed. Run: pip install gspread google-auth")

        self.spreadsheet_name = spreadsheet_name
        self.credentials_file = self.CREDENTIALS_DIR / "google_sheets_creds.json"
        self.authorized_user_file = self.CREDENTIALS_DIR / "authorized_user.json"
        self.client: Optional["gspread.Client"] = None
        self.spreadsheet: Optional["gspread.Spreadsheet"] = None

    def connect(self) -> bool:
        """Connect to Google Sheets API using OAuth2 or service account."""
        # Check for service account credentials from environment (GitHub Actions)
        import os
        service_account_json = os.environ.get('GOOGLE_CREDENTIALS')

        if service_account_json:
            # Use service account from environment variable
            try:
                import json
                creds_dict = json.loads(service_account_json)
                creds = Credentials.from_service_account_info(creds_dict, scopes=self.SCOPES)
                self.client = gspread.authorize(creds)
                logger.info("Connected to Google Sheets API via service account (env)")
                return True
            except Exception as e:
                logger.error(f"Service account auth failed: {e}")
                return False

        # Try OAuth2 first (local development - browser popup)
        try:
            self.client = gspread.oauth(
                scopes=self.SCOPES,
                credentials_filename=str(self.credentials_file) if self.credentials_file.exists() else None,
                authorized_user_filename=str(self.authorized_user_file)
            )
            logger.info("Connected to Google Sheets API via OAuth2")
            return True
        except Exception as e:
            logger.warning(f"OAuth2 failed: {e}, trying service account file...")

        # Fallback to service account file
        if self.credentials_file.exists():
            try:
                creds = Credentials.from_service_account_file(
                    str(self.credentials_file),
                    scopes=self.SCOPES
                )
                self.client = gspread.authorize(creds)
                logger.info("Connected to Google Sheets API via service account file")
                return True
            except Exception as e:
                logger.error(f"Service account file auth failed: {e}")

        logger.error("All authentication methods failed")
        return False

    def get_or_create_spreadsheet(self) -> Optional["gspread.Spreadsheet"]:
        """Get existing spreadsheet or create new one."""
        if not self.client:
            if not self.connect():
                return None

        try:
            self.spreadsheet = self.client.open(self.spreadsheet_name)
            logger.info(f"Opened existing spreadsheet: {self.spreadsheet_name}")
        except gspread.SpreadsheetNotFound:
            self.spreadsheet = self.client.create(self.spreadsheet_name)
            logger.info(f"Created new spreadsheet: {self.spreadsheet_name}")
            self._setup_spreadsheet()

        return self.spreadsheet

    def _setup_spreadsheet(self):
        """Set up spreadsheet with headers."""
        if not self.spreadsheet:
            return

        sheet = self.spreadsheet.sheet1
        headers = [
            'Date', 'Ticker', 'Signal', 'Confidence', 'Entry Price',
            'RSI', 'Trend', 'MACD', 'Current Price', 'Return %',
            'Direction', 'Result', 'Days Held'
        ]
        sheet.update('A1:M1', [headers])

        # Format header row
        sheet.format('A1:M1', {
            'backgroundColor': {'red': 0.2, 'green': 0.4, 'blue': 0.8},
            'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}},
            'horizontalAlignment': 'CENTER'
        })

    def export_predictions(self, predictions: List[Dict], date: str = None) -> bool:
        """
        Export predictions to Google Sheets.

        Args:
            predictions: List of prediction dicts (only BUY/SELL, not HOLD)
            date: Prediction date (defaults to today)
        """
        if not self.get_or_create_spreadsheet():
            return False

        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        sheet = self.spreadsheet.sheet1

        # Filter to actionable predictions
        actionable = [p for p in predictions if p.get('signal', 'HOLD') != 'HOLD']

        # Prepare rows
        rows = []
        for pred in actionable:
            row = [
                date,
                pred.get('ticker', ''),
                pred.get('signal', ''),
                round(pred.get('confidence', 0) * 100, 1),
                round(pred.get('current_price', 0), 2),
                round(pred.get('rsi', 50), 1),
                pred.get('trend', ''),
                pred.get('macd_signal', ''),
                '',  # Current price (filled later)
                '',  # Return % (filled later)
                '',  # Direction (filled later)
                '',  # Result (filled later)
                '',  # Days held (filled later)
            ]
            rows.append(row)

        if not rows:
            logger.info("No actionable predictions to export")
            return True

        # Find first empty row
        all_values = sheet.get_all_values()
        next_row = len(all_values) + 1

        # Append rows
        sheet.update(f'A{next_row}:M{next_row + len(rows) - 1}', rows)
        logger.info(f"Exported {len(rows)} predictions to Google Sheets")

        return True

    def update_results(self, date: str, results: List[Dict]) -> int:
        """
        Update results for predictions made on a specific date.

        Args:
            date: Original prediction date
            results: List of result dicts with ticker, current_price, pct_change, etc.

        Returns:
            Number of rows updated
        """
        if not self.get_or_create_spreadsheet():
            return 0

        sheet = self.spreadsheet.sheet1
        all_values = sheet.get_all_values()

        # Create lookup from results
        results_lookup = {r['ticker']: r for r in results}

        updated = 0
        for i, row in enumerate(all_values[1:], start=2):  # Skip header
            if len(row) < 2:
                continue

            row_date = row[0]
            ticker = row[1]

            if row_date == date and ticker in results_lookup:
                result = results_lookup[ticker]

                # Calculate days held
                try:
                    pred_date = datetime.strptime(date, '%Y-%m-%d')
                    days_held = (datetime.now() - pred_date).days
                except:
                    days_held = 0

                # Determine if prediction was correct
                pct_change = result.get('pct_change', 0)
                signal = row[2] if len(row) > 2 else ''

                if signal in ['STRONG BUY', 'BUY']:
                    is_correct = pct_change > 0
                else:
                    is_correct = pct_change < 0

                direction = 'UP' if pct_change > 0 else 'DOWN'
                result_str = 'HIT' if is_correct else 'MISS'

                # Update columns I-M (9-13)
                updates = [
                    round(result.get('current_price', 0), 2),
                    round(pct_change, 2),
                    direction,
                    result_str,
                    days_held
                ]
                sheet.update(f'I{i}:M{i}', [updates])
                updated += 1

                # Color the result cell
                color = {'red': 0.2, 'green': 0.8, 'blue': 0.2} if is_correct else {'red': 0.9, 'green': 0.3, 'blue': 0.3}
                sheet.format(f'L{i}', {'backgroundColor': color})

        logger.info(f"Updated {updated} results for {date}")
        return updated

    def get_spreadsheet_url(self) -> str:
        """Get the URL of the spreadsheet."""
        if self.spreadsheet:
            return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet.id}"
        return ""

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary from sheet."""
        if not self.get_or_create_spreadsheet():
            return {}

        sheet = self.spreadsheet.sheet1
        all_values = sheet.get_all_values()

        if len(all_values) <= 1:
            return {}

        total = 0
        hits = 0
        by_signal = {}

        for row in all_values[1:]:
            if len(row) < 12 or not row[11]:  # Result column
                continue

            total += 1
            result = row[11]
            signal = row[2]

            if result == 'HIT':
                hits += 1

            if signal not in by_signal:
                by_signal[signal] = {'total': 0, 'hits': 0}
            by_signal[signal]['total'] += 1
            if result == 'HIT':
                by_signal[signal]['hits'] += 1

        return {
            'total': total,
            'hits': hits,
            'accuracy': round(hits / total * 100, 1) if total > 0 else 0,
            'by_signal': {
                k: {
                    'total': v['total'],
                    'hits': v['hits'],
                    'accuracy': round(v['hits'] / v['total'] * 100, 1) if v['total'] > 0 else 0
                }
                for k, v in by_signal.items()
            }
        }

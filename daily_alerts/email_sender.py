"""
Email Sender for Stock Alerts using SendGrid
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
import yaml

from .predictor import PredictionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockEmailSender:
    """Sends daily stock alerts via SendGrid"""

    def __init__(self, config: Dict):
        self.config = config
        email_config = config.get('email', {})

        self.api_key = email_config.get('api_key') or os.environ.get('SENDGRID_API_KEY')
        self.sender = email_config.get('sender', 'alerts@example.com')
        self.sender_name = email_config.get('sender_name', 'Stock Prediction Alerts')
        self.recipients = email_config.get('recipients', [])

        if not self.api_key:
            logger.warning("SendGrid API key not configured!")

    def _format_price(self, price: float) -> str:
        """Format price for display"""
        if price >= 1000:
            return f"${price:,.0f}"
        elif price >= 100:
            return f"${price:.1f}"
        else:
            return f"${price:.2f}"

    def _format_market_cap(self, cap: float) -> str:
        """Format market cap for display"""
        if cap >= 1e12:
            return f"${cap/1e12:.1f}T"
        elif cap >= 1e9:
            return f"${cap/1e9:.1f}B"
        elif cap >= 1e6:
            return f"${cap/1e6:.0f}M"
        else:
            return "N/A"

    def _get_signal_emoji(self, signal: str) -> str:
        """Get emoji for signal type"""
        emojis = {
            'STRONG BUY': '🚀',
            'BUY': '📈',
            'HOLD': '➖',
            'SELL': '📉',
            'STRONG SELL': '🔻'
        }
        return emojis.get(signal, '')

    def _create_stock_row(self, result: PredictionResult) -> str:
        """Create HTML row for a stock"""
        confidence_color = '#28a745' if result.confidence > 0.65 else '#6c757d'
        rsi_color = '#dc3545' if result.rsi > 70 else ('#28a745' if result.rsi < 30 else '#6c757d')

        return f"""
        <tr style="border-bottom: 1px solid #dee2e6;">
            <td style="padding: 10px; font-weight: bold;">{result.ticker}</td>
            <td style="padding: 10px;">{self._format_price(result.current_price)}</td>
            <td style="padding: 10px; color: {confidence_color}; font-weight: bold;">{result.confidence:.0%}</td>
            <td style="padding: 10px; color: {rsi_color};">{result.rsi:.0f}</td>
            <td style="padding: 10px;">{result.macd_signal}</td>
            <td style="padding: 10px;">{result.trend}</td>
            <td style="padding: 10px; font-size: 12px;">{result.sector}</td>
        </tr>
        """

    def _create_signal_section(self, title: str, emoji: str, results: List[PredictionResult], color: str) -> str:
        """Create HTML section for a signal type"""
        if not results:
            return ""

        rows = ''.join([self._create_stock_row(r) for r in results[:10]])  # Max 10 per section

        return f"""
        <div style="margin-bottom: 30px;">
            <h2 style="color: {color}; border-bottom: 2px solid {color}; padding-bottom: 10px;">
                {emoji} {title} ({len(results)} stocks)
            </h2>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <thead>
                    <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                        <th style="padding: 10px; text-align: left;">Ticker</th>
                        <th style="padding: 10px; text-align: left;">Price</th>
                        <th style="padding: 10px; text-align: left;">Confidence</th>
                        <th style="padding: 10px; text-align: left;">RSI</th>
                        <th style="padding: 10px; text-align: left;">MACD</th>
                        <th style="padding: 10px; text-align: left;">Trend</th>
                        <th style="padding: 10px; text-align: left;">Sector</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """

    def _create_outlook_section(self, results: List[PredictionResult]) -> str:
        """Create multi-timeframe outlook section"""
        # Get top 5 for each timeframe
        short_term_bulls = sorted(results, key=lambda x: x.short_term_prediction, reverse=True)[:5]
        medium_term_bulls = sorted(results, key=lambda x: x.medium_term_prediction, reverse=True)[:5]
        long_term_bulls = sorted(results, key=lambda x: x.long_term_prediction, reverse=True)[:5]

        def create_outlook_list(stocks: List[PredictionResult], prob_key: str) -> str:
            items = ""
            for s in stocks:
                prob = getattr(s, prob_key)
                items += f"<li><strong>{s.ticker}</strong> - {prob:.0%} bullish probability ({self._format_price(s.current_price)})</li>"
            return f"<ul style='list-style-type: none; padding: 0;'>{items}</ul>"

        return f"""
        <div style="margin-top: 40px; background-color: #f8f9fa; padding: 20px; border-radius: 8px;">
            <h2 style="color: #343a40;">📊 Multi-Timeframe Outlook</h2>

            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 200px;">
                    <h3 style="color: #17a2b8;">📅 Daily (1-Day)</h3>
                    <p style="color: #6c757d; font-size: 12px;">Highest probability of gains tomorrow</p>
                    {create_outlook_list(short_term_bulls, 'short_term_prediction')}
                </div>

                <div style="flex: 1; min-width: 200px;">
                    <h3 style="color: #28a745;">📆 Weekly (5-Day)</h3>
                    <p style="color: #6c757d; font-size: 12px;">Best outlook for the week</p>
                    {create_outlook_list(medium_term_bulls, 'medium_term_prediction')}
                </div>

                <div style="flex: 1; min-width: 200px;">
                    <h3 style="color: #6610f2;">📈 Monthly (20-Day)</h3>
                    <p style="color: #6c757d; font-size: 12px;">Strongest long-term potential</p>
                    {create_outlook_list(long_term_bulls, 'long_term_prediction')}
                </div>
            </div>
        </div>
        """

    def create_email_content(self, signals: Dict[str, List[PredictionResult]],
                             all_results: List[PredictionResult]) -> tuple:
        """Create email subject and HTML content"""
        date_str = datetime.now().strftime("%B %d, %Y")
        total_signals = len(signals.get('STRONG BUY', [])) + len(signals.get('STRONG SELL', []))

        subject = f"📊 Stock Alerts - {date_str} - {total_signals} Strong Signals Found"

        # Summary stats
        buy_count = len(signals.get('STRONG BUY', [])) + len(signals.get('BUY', []))
        sell_count = len(signals.get('STRONG SELL', [])) + len(signals.get('SELL', []))

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Stock Prediction Alerts</title>
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; color: #333;">

            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px;">
                <h1 style="margin: 0; font-size: 28px;">📈 Daily Stock Prediction Report</h1>
                <p style="margin: 10px 0 0 0; opacity: 0.9;">{date_str} | S&P 500 Analysis</p>
            </div>

            <div style="background-color: #e7f3ff; padding: 20px; border-radius: 8px; margin-bottom: 30px;">
                <h3 style="margin-top: 0; color: #0056b3;">📋 Today's Summary</h3>
                <div style="display: flex; gap: 30px; flex-wrap: wrap;">
                    <div>
                        <span style="font-size: 24px; color: #28a745; font-weight: bold;">{buy_count}</span>
                        <span style="color: #6c757d;"> Buy Signals</span>
                    </div>
                    <div>
                        <span style="font-size: 24px; color: #dc3545; font-weight: bold;">{sell_count}</span>
                        <span style="color: #6c757d;"> Sell Signals</span>
                    </div>
                    <div>
                        <span style="font-size: 24px; color: #6c757d; font-weight: bold;">{len(all_results)}</span>
                        <span style="color: #6c757d;"> Stocks Analyzed</span>
                    </div>
                </div>
            </div>

            {self._create_signal_section("STRONG BUY Signals", "🚀", signals.get('STRONG BUY', []), "#28a745")}
            {self._create_signal_section("BUY Signals", "📈", signals.get('BUY', []), "#20c997")}
            {self._create_signal_section("STRONG SELL Signals", "🔻", signals.get('STRONG SELL', []), "#dc3545")}
            {self._create_signal_section("SELL Signals", "📉", signals.get('SELL', []), "#fd7e14")}

            {self._create_outlook_section(all_results)}

            <div style="margin-top: 40px; padding: 20px; background-color: #fff3cd; border-radius: 8px;">
                <h3 style="margin-top: 0; color: #856404;">⚠️ Disclaimer</h3>
                <p style="color: #856404; font-size: 12px; margin: 0;">
                    This is an automated analysis for informational purposes only. Past performance does not guarantee future results.
                    Always do your own research and consider consulting a financial advisor before making investment decisions.
                    The predictions are based on technical analysis and machine learning models, which have inherent limitations.
                </p>
            </div>

            <div style="margin-top: 30px; text-align: center; color: #6c757d; font-size: 12px;">
                <p>Generated by Stock Prediction System | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>

        </body>
        </html>
        """

        return subject, html_content

    def send_alert(self, signals: Dict[str, List[PredictionResult]],
                   all_results: List[PredictionResult]) -> bool:
        """Send alert email via SendGrid"""
        if not self.api_key:
            logger.error("Cannot send email: SendGrid API key not configured")
            return False

        if not self.recipients:
            logger.error("Cannot send email: No recipients configured")
            return False

        subject, html_content = self.create_email_content(signals, all_results)

        try:
            sg = SendGridAPIClient(self.api_key)

            for recipient in self.recipients:
                message = Mail(
                    from_email=Email(self.sender, self.sender_name),
                    to_emails=To(recipient),
                    subject=subject,
                    html_content=Content("text/html", html_content)
                )

                response = sg.send(message)
                logger.info(f"Email sent to {recipient}: Status {response.status_code}")

            return True

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False


if __name__ == "__main__":
    # Test email formatting
    from predictor import PredictionResult

    # Create sample results
    sample_results = [
        PredictionResult(
            ticker="AAPL", current_price=175.50, predicted_direction="UP",
            confidence=0.78, signal="STRONG BUY", short_term_prediction=0.72,
            medium_term_prediction=0.68, long_term_prediction=0.65,
            rsi=45, macd_signal="BULLISH", trend="UPTREND",
            sector="Technology", market_cap=2.8e12
        ),
        PredictionResult(
            ticker="TSLA", current_price=245.30, predicted_direction="DOWN",
            confidence=0.71, signal="STRONG SELL", short_term_prediction=0.28,
            medium_term_prediction=0.35, long_term_prediction=0.40,
            rsi=75, macd_signal="BEARISH", trend="DOWNTREND",
            sector="Consumer Cyclical", market_cap=780e9
        ),
    ]

    signals = {
        'STRONG BUY': [sample_results[0]],
        'BUY': [],
        'HOLD': [],
        'SELL': [],
        'STRONG SELL': [sample_results[1]]
    }

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    sender = StockEmailSender(config)
    subject, html = sender.create_email_content(signals, sample_results)

    # Save sample email for preview
    with open("sample_email.html", "w") as f:
        f.write(html)
    print(f"Sample email saved to sample_email.html")
    print(f"Subject: {subject}")


# ============================================================================
# DARK LOGIC ALGO - OPENALGO COMPATIBLE VERSION
# Automated Trading Strategy for OpenAlgo Platform
# Uses OpenAlgo Python SDK for Order Execution
# ============================================================================

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from openalgo import api

# ============================================================================
# SETUP LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('openalgo_dark_logic.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    '''Configuration class for strategy parameters'''

    # OpenAlgo Connection Settings
    OPENALGO_API_KEY = "e2dfd190d6a8ac7b201032bb10b699fdc2df1397ab2fa146273177b09d4f0a18"  # Get from OpenAlgo Profile
    OPENALGO_HOST = "http://127.0.0.1:5000"     # Default local host

    # Strategy Settings
    STRATEGY_NAME = "DarkLogicAlgo"
    SYMBOLS = ["NIFTY", "BANKNIFTY"]
    EXCHANGE = "NSE_INDEX"  # For spot price
    OPTIONS_EXCHANGE = "NFO"  # For options trading

    # Trading Parameters
    RISK_PER_TRADE = 1.0        # Percentage of capital
    STOP_LOSS_PCT = 2.0          # Percentage
    TAKE_PROFIT_PCT = 4.0        # Percentage
    TRAILING_SL_ENABLED = True
    MAX_TRADES_PER_DAY = 10

    # Technical Indicator Parameters
    STOCH_K_PERIOD = 14
    STOCH_D_PERIOD = 3
    ATR_PERIOD = 14
    EMA_PERIOD = 20
    RSI_PERIOD = 14

    # Trading Hours
    TRADING_START = "09:15"
    TRADING_END = "15:30"

    # Options Settings
    STRIKE_INTERVAL = 50
    OPTION_EXPIRY = "14NOV25"  # Update with current expiry
    PRODUCT_TYPE = "MIS"  # Intraday

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================
class TechnicalIndicators:
    '''Technical analysis indicators for signal generation'''

    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, 
                            d_period: int = 3) -> pd.DataFrame:
        '''Calculate Stochastic Oscillator'''
        df['lowest_low'] = df['low'].rolling(window=k_period).min()
        df['highest_high'] = df['high'].rolling(window=k_period).max()
        df['stoch_k'] = 100 * ((df['close'] - df['lowest_low']) / 
                              (df['highest_high'] - df['lowest_low']))
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        return df

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        '''Calculate Average True Range (ATR)'''
        df['h_l'] = df['high'] - df['low']
        df['h_pc'] = abs(df['high'] - df['close'].shift(1))
        df['l_pc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=period).mean()
        return df

    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        '''Calculate Exponential Moving Average'''
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        '''Calculate Relative Strength Index'''
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

# ============================================================================
# OPENALGO TRADING ENGINE
# ============================================================================
class OpenAlgoDarkLogic:
    '''
    Main Trading Engine using OpenAlgo API
    Implements Dark Logic Algo strategy with OpenAlgo integration
    '''

    def __init__(self, config: Config):
        self.config = config

        # Initialize OpenAlgo client
        self.client = api(
            api_key=config.OPENALGO_API_KEY,
            host=config.OPENALGO_HOST
        )

        # Strategy state
        self.in_position = False
        self.current_symbol = None
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trades_today = 0
        self.position_quantity = 0

        logger.info("OpenAlgo Dark Logic Strategy initialized")

    def get_historical_data(self, symbol: str, days: int = 5) -> pd.DataFrame:
        '''Fetch historical data from OpenAlgo'''
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            response = self.client.history(
                symbol=symbol,
                exchange=self.config.EXCHANGE,
                interval="5m",
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )

            if isinstance(response, pd.DataFrame):
                logger.info(f"Fetched {len(response)} candles for {symbol}")
                return response
            else:
                logger.error(f"Invalid response format: {response}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def get_option_symbol(self, underlying: str, option_type: str, 
                         offset: str = "ATM") -> Optional[Dict]:
        '''Get option symbol using OpenAlgo's optionsymbol function'''
        try:
            response = self.client.optionsymbol(
                underlying=underlying,
                exchange=self.config.EXCHANGE,
                expiry_date=self.config.OPTION_EXPIRY,
                strike_int=self.config.STRIKE_INTERVAL,
                offset=offset,
                option_type=option_type
            )

            if response.get('status') == 'success':
                logger.info(f"Option symbol: {response.get('symbol')}")
                return response
            else:
                logger.error(f"Failed to get option symbol: {response}")
                return None

        except Exception as e:
            logger.error(f"Error getting option symbol: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Calculate all technical indicators'''
        if len(df) < 50:
            logger.warning("Insufficient data for indicator calculation")
            return df

        df = TechnicalIndicators.calculate_stochastic(
            df, self.config.STOCH_K_PERIOD, self.config.STOCH_D_PERIOD
        )
        df = TechnicalIndicators.calculate_atr(df, self.config.ATR_PERIOD)
        df = TechnicalIndicators.calculate_ema(df, self.config.EMA_PERIOD)
        df = TechnicalIndicators.calculate_rsi(df, self.config.RSI_PERIOD)

        return df

    def generate_signal(self, df: pd.DataFrame) -> Dict:
        '''Generate trading signal based on technical indicators'''
        if len(df) < 50:
            return {'signal': 'NEUTRAL', 'strength': 0.0}

        # Get latest values
        current_price = df['close'].iloc[-1]
        stoch_k = df['stoch_k'].iloc[-1]
        stoch_d = df['stoch_d'].iloc[-1]
        prev_stoch_k = df['stoch_k'].iloc[-2]
        prev_stoch_d = df['stoch_d'].iloc[-2]
        atr = df['atr'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        ema_20 = df[f'ema_{self.config.EMA_PERIOD}'].iloc[-1]

        # Buy Signal Conditions
        buy_conditions = [
            (prev_stoch_k < prev_stoch_d) and (stoch_k > stoch_d),  # Crossover
            stoch_k < 30,  # Oversold
            rsi < 70,  # Not overbought
            current_price > ema_20,  # Above EMA
            atr > df['atr'].iloc[-10:].mean()  # Volatility
        ]

        # Sell Signal Conditions
        sell_conditions = [
            (prev_stoch_k > prev_stoch_d) and (stoch_k < stoch_d),  # Crossover
            stoch_k > 70,  # Overbought
            rsi > 30,  # Not oversold
            current_price < ema_20,  # Below EMA
            atr > df['atr'].iloc[-10:].mean()  # Volatility
        ]

        buy_strength = sum(buy_conditions) / len(buy_conditions)
        sell_strength = sum(sell_conditions) / len(sell_conditions)

        signal = {'signal': 'NEUTRAL', 'strength': 0.0, 'price': current_price}

        if buy_strength > 0.6 and not self.in_position:
            signal['signal'] = 'BUY'
            signal['strength'] = buy_strength
            signal['stop_loss'] = current_price * (1 - self.config.STOP_LOSS_PCT/100)
            signal['take_profit'] = current_price * (1 + self.config.TAKE_PROFIT_PCT/100)

        elif sell_strength > 0.6 and not self.in_position:
            signal['signal'] = 'SELL'
            signal['strength'] = sell_strength
            signal['stop_loss'] = current_price * (1 + self.config.STOP_LOSS_PCT/100)
            signal['take_profit'] = current_price * (1 - self.config.TAKE_PROFIT_PCT/100)

        return signal

    def place_options_order(self, signal: Dict, underlying: str):
        '''Place options order using OpenAlgo'''
        if self.trades_today >= self.config.MAX_TRADES_PER_DAY:
            logger.warning("Max trades per day reached")
            return False

        try:
            # Determine option type based on signal
            option_type = "CE" if signal['signal'] == 'BUY' else "PE"
            action = "BUY"  # We buy options for both calls and puts

            # Get option symbol
            option_info = self.get_option_symbol(underlying, option_type, "ATM")
            if not option_info:
                logger.error("Failed to get option symbol")
                return False

            symbol = option_info['symbol']
            lot_size = option_info['lotsize']

            # Place order using OpenAlgo
            response = self.client.placeorder(
                strategy=self.config.STRATEGY_NAME,
                symbol=symbol,
                action=action,
                exchange=self.config.OPTIONS_EXCHANGE,
                price_type="MARKET",
                product=self.config.PRODUCT_TYPE,
                quantity=lot_size
            )

            if response.get('status') == 'success':
                order_id = response.get('orderid')
                logger.info(f"✓ Order placed successfully: {order_id}")
                logger.info(f"  Symbol: {symbol}, Type: {option_type}, Qty: {lot_size}")

                self.in_position = True
                self.current_symbol = symbol
                self.entry_price = signal['price']
                self.stop_loss = signal.get('stop_loss', 0)
                self.take_profit = signal.get('take_profit', 0)
                self.position_quantity = lot_size
                self.trades_today += 1

                return True
            else:
                logger.error(f"Order failed: {response}")
                return False

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False

    def check_position_status(self):
        '''Check and manage open positions'''
        if not self.in_position or not self.current_symbol:
            return

        try:
            # Get current position
            position_response = self.client.openposition(
                strategy=self.config.STRATEGY_NAME,
                symbol=self.current_symbol,
                exchange=self.config.OPTIONS_EXCHANGE,
                product=self.config.PRODUCT_TYPE
            )

            if position_response.get('status') == 'success':
                quantity = int(position_response.get('quantity', 0))

                if quantity == 0:
                    logger.info("Position already closed")
                    self.in_position = False
                    self.current_symbol = None
                    return

                # Get current LTP
                quotes_response = self.client.quotes(
                    symbol=self.current_symbol,
                    exchange=self.config.OPTIONS_EXCHANGE
                )

                if quotes_response.get('status') == 'success':
                    current_price = quotes_response['data']['ltp']

                    # Check stop loss
                    if current_price <= self.stop_loss:
                        logger.info(f"Stop Loss hit at {current_price}")
                        self.close_position()
                        return

                    # Check take profit
                    if current_price >= self.take_profit:
                        logger.info(f"Take Profit hit at {current_price}")
                        self.close_position()
                        return

                    # Update trailing stop loss
                    if self.config.TRAILING_SL_ENABLED:
                        profit_pct = ((current_price - self.entry_price) / self.entry_price) * 100
                        if profit_pct > 1.0:
                            new_sl = current_price * (1 - self.config.STOP_LOSS_PCT/100)
                            if new_sl > self.stop_loss:
                                self.stop_loss = new_sl
                                logger.info(f"Trailing SL updated to {self.stop_loss:.2f}")

        except Exception as e:
            logger.error(f"Error checking position: {e}")

    def close_position(self):
        '''Close current position using OpenAlgo'''
        if not self.in_position or not self.current_symbol:
            return

        try:
            response = self.client.closeposition(
                strategy=self.config.STRATEGY_NAME
            )

            if response.get('status') == 'success':
                logger.info(f"✓ Position closed: {self.current_symbol}")
                self.in_position = False
                self.current_symbol = None
                self.entry_price = 0.0
                self.stop_loss = 0.0
                self.take_profit = 0.0
                self.position_quantity = 0
            else:
                logger.error(f"Failed to close position: {response}")

        except Exception as e:
            logger.error(f"Error closing position: {e}")

    def is_trading_hours(self) -> bool:
        '''Check if current time is within trading hours'''
        now = datetime.now().time()
        start_time = datetime.strptime(self.config.TRADING_START, '%H:%M').time()
        end_time = datetime.strptime(self.config.TRADING_END, '%H:%M').time()
        return start_time <= now <= end_time

    def run(self):
        '''Main strategy loop'''
        logger.info("="*80)
        logger.info("Dark Logic Algo Strategy - OpenAlgo Version Started")
        logger.info("="*80)

        # Test OpenAlgo connection
        try:
            funds = self.client.funds()
            logger.info(f"Connected to OpenAlgo - Available funds: {funds}")
        except Exception as e:
            logger.error(f"Failed to connect to OpenAlgo: {e}")
            return

        while True:
            try:
                # Check trading hours
                if not self.is_trading_hours():
                    logger.info("Outside trading hours. Sleeping...")
                    time.sleep(60)
                    continue

                # Check existing positions
                self.check_position_status()

                # If already in position, skip signal generation
                if self.in_position:
                    time.sleep(30)
                    continue

                # Process each symbol
                for symbol in self.config.SYMBOLS:
                    logger.info(f"\nAnalyzing {symbol}...")

                    # Get historical data
                    df = self.get_historical_data(symbol)
                    if df.empty:
                        continue

                    # Calculate indicators
                    df = self.calculate_indicators(df)

                    # Generate signal
                    signal = self.generate_signal(df)

                    logger.info(f"Signal: {signal['signal']}, Strength: {signal['strength']:.2f}")

                    # Execute trade if signal is strong
                    if signal['signal'] in ['BUY', 'SELL'] and signal['strength'] > 0.6:
                        logger.info(f"🚀 {signal['signal']} Signal Generated!")
                        self.place_options_order(signal, symbol)
                        break  # Only one trade at a time

                # Sleep before next iteration
                time.sleep(60)

            except KeyboardInterrupt:
                logger.info("\nStrategy stopped by user")
                # Close any open positions
                if self.in_position:
                    self.close_position()
                break

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Create configuration
    config = Config()

    # Important: Update these before running
    print("="*80)
    print("OPENALGO DARK LOGIC ALGO STRATEGY")
    print("="*80)
    print("\nBefore running, please update:")
    print("1. OPENALGO_API_KEY in Config class")
    print("2. OPTION_EXPIRY with current expiry date")
    print("3. Review other parameters as needed")
    print("="*80)

    # Prompt user to continue
    response = input("\nHave you updated the configuration? (yes/no): ")
    if response.lower() != 'yes':
        print("Please update configuration before running.")
        exit()

    # Initialize and run strategy
    strategy = OpenAlgoDarkLogic(config)
    strategy.run()

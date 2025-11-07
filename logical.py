#!/usr/bin/env python3
"""
🔁 OpenAlgo Python Bot is running.
Requires:
  pip install openalgo apscheduler pytz pandas numpy
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dtime
import pytz
from apscheduler.schedulers.background import BackgroundScheduler

# OpenAlgo client (local)
from openalgo import api, ta  # using OpenAlgo's api + ta modules as default

# --------------------
# Config / constants
# --------------------
EXCHANGE = "NFO"                # options on NSE futures & options
UNDERLYING_EXCHANGE = "NSE_INDEX"
PRODUCT = "MIS"                 # intraday by default
PRICE_TYPE = "MARKET"
NIFTY_LOT_SIZE = 75             # NIFTY lot size
MAX_SPEND = 100000              # fallback available cash if account queries fail
IST = pytz.timezone("Asia/Kolkata")
API_EXPECTS_LOTS = True         # set False if your broker expects contracts

print("🔁 OpenAlgo Python Bot is running.")  # required startup print

# --------------------
# Month names for OpenAlgo symbology
# --------------------
MONTHS = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"
}

def expiry_str_from_date(dt: datetime):
    """Format expiry date for OpenAlgo symbol: DDMMMYY -> 07OCT25"""
    day = dt.day
    mon = MONTHS[dt.month]
    yy = str(dt.year)[-2:]
    return f"{day:02d}{mon}{yy}"

def build_option_symbol(base_symbol: str, expiry_dt: datetime, strike: int, opt_type: str):
    ex = expiry_str_from_date(expiry_dt)
    return f"{base_symbol}{ex}{strike}{opt_type}"

def nearest_strike(spot_price, strike_step=50):
    return int(round(spot_price / strike_step) * strike_step)

# --------------------
# Core engine
# --------------------
class OptionBuyingEngine:
    def __init__(self, api_key, base_symbol="NIFTY", timeframe="1m",
                 lorentz_threshold=2.0, strike_step=50,
                 max_spend_fallback=MAX_SPEND, hold_hours=2):
        self.client = api(api_key=api_key, host="http://127.0.0.1:5000")
        self.base_symbol = base_symbol
        self.timeframe = timeframe
        self.lorentz_threshold = lorentz_threshold
        self.strike_step = strike_step
        self.max_spend_fallback = max_spend_fallback
        self.hold_hours = hold_hours

        self.positions = {}
        self.last_signal = {}
        self.current_target_strike = None
        self.current_target_expiry = None
        self.current_target_timestamp = None
        self.market_started = False

        # Scheduler for EOD & expiry close
        self.scheduler = BackgroundScheduler(timezone=IST)
        self.scheduler.add_job(self.force_close_all, 'cron', hour=14, minute=58)
        self.scheduler.add_job(self.force_close_all_on_expiry, 'cron', day_of_week='tue', hour=12, minute=0)
        self.scheduler.start()

    # -------------------- Market Data --------------------
    def fetch_spot(self):
        try:
            q = self.client.quotes(symbol=self.base_symbol, exchange=UNDERLYING_EXCHANGE)
            print(f"[QUOTE] {UNDERLYING_EXCHANGE}:{self.base_symbol} -> {q}")
            if isinstance(q, dict) and "data" in q and isinstance(q["data"], dict):
                ltp = q["data"].get("ltp") or q["data"].get("close")
                return float(ltp)
            return None
        except Exception as e:
            print("Error fetching spot:", e)
            return None

    def current_weekly_tuesday(self, ref_dt: datetime):
        if ref_dt.tzinfo is None:
            ref_dt = IST.localize(ref_dt)
        weekday = ref_dt.weekday()
        days_ahead = (1 - weekday) % 7  # Tuesday
        candidate = ref_dt + timedelta(days=days_ahead)
        expiry_cutoff = candidate.replace(hour=15, minute=30)
        if ref_dt >= expiry_cutoff:
            candidate += timedelta(days=7)
        return candidate.astimezone(IST)

    def get_atm_weekly_option_symbols_from_spot(self, spot):
        strike = nearest_strike(spot, strike_step=self.strike_step)
        expiry_dt = self.current_weekly_tuesday(datetime.now(IST)).replace(hour=15, minute=30)
        ce = build_option_symbol(self.base_symbol, expiry_dt, strike, "CE")
        pe = build_option_symbol(self.base_symbol, expiry_dt, strike, "PE")
        return ce, pe, expiry_dt, strike

    def get_current_target_symbols(self):
        if self.current_target_strike and self.current_target_expiry:
            ce = build_option_symbol(self.base_symbol, self.current_target_expiry, self.current_target_strike, "CE")
            pe = build_option_symbol(self.base_symbol, self.current_target_expiry, self.current_target_strike, "PE")
            return ce, pe, self.current_target_expiry, self.current_target_strike

        spot = self.fetch_spot()
        ce, pe, expiry_dt, strike = self.get_atm_weekly_option_symbols_from_spot(spot)
        self.current_target_strike = strike
        self.current_target_expiry = expiry_dt
        self.current_target_timestamp = datetime.now(IST)
        print(f"[TARGET-CAPTURE] Captured ATM strike {strike} at {self.current_target_timestamp}")
        return ce, pe, expiry_dt, strike

    # -------------------- Historical Data --------------------
    def fetch_history_for_symbol(self, symbol, exchange=EXCHANGE, start_date=None, end_date=None):
        end = datetime.now(IST)
        start = end - timedelta(days=2)
        start_date = start.strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")
        try:
            df = self.client.history(symbol=symbol, exchange=exchange, interval=self.timeframe,
                                     start_date=start_date, end_date=end_date)
            if isinstance(df, dict):
                df = pd.DataFrame(df.get("data") or df)
            if "timestamp" in df.columns:
                df.index = pd.to_datetime(df["timestamp"])
            print(f"[HISTORY] {exchange}:{symbol} -> {len(df)} rows ({start_date} -> {end_date})")
            return df
        except Exception as e:
            print("Error fetching history:", e)
            return pd.DataFrame()

    # -------------------- Account Sizing --------------------
    def get_available_cash(self):
        try:
            res = self.client.getfunds()
            if isinstance(res, dict) and "available" in res:
                return float(res["available"])
        except Exception:
            pass
        print(f"[ACCOUNT] Using fallback cash {self.max_spend_fallback}")
        return float(self.max_spend_fallback)

    def compute_max_lots_for_buy(self, premium_per_contract, lot_size=NIFTY_LOT_SIZE):
        available = self.get_available_cash()
        if premium_per_contract <= 0:
            return 0
        per_lot_cost = premium_per_contract * lot_size
        max_lots = int(available // per_lot_cost)
        if max_lots <= 0:
            print(f"[SIZING] Not enough funds for 1 lot. available={available}, cost={per_lot_cost}")
            return 0
        return max_lots

    # -------------------- Indicator Feature Builder --------------------
    def calc_rsi(self, series, period):
        series = np.asarray(series, dtype=float)
        delta = np.diff(series, prepend=series[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(period).mean().to_numpy()
        avg_loss = pd.Series(loss).rolling(period).mean().to_numpy()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_features_from_df(self, df):
        close = df['close'].to_numpy()
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        hl3 = (high + low + close) / 3

        f1 = ta.rsi(df['close'], period=14)
        f2 = ta.rsi(pd.Series(hl3), period=10)

        tp = (high + low + close) / 3
        ma = pd.Series(tp).rolling(20).mean().to_numpy()
        md = pd.Series(np.abs(tp - ma)).rolling(20).mean().to_numpy()
        f3 = (tp - ma) / (0.015 * md + 1e-8)

        _, _, adx = ta.adx(df['high'], df['low'], df['close'], period=20)
        f4 = np.asarray(adx)
        f5 = ta.rsi(df['close'], period=9)

        features_matrix = np.vstack([f1, f2, f3, f4, f5])
        # FIX: Replace NaN / inf with safe values
        features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return features_matrix

    def lorentzian_score(self, features_matrix, idx):
        n_features, n_bars = features_matrix.shape
        current = np.nan_to_num(features_matrix[:, idx])
        sample_start = max(0, idx - 200)
        distances = []
        for i in range(sample_start, idx):
            if (i % 4) != 0:
                continue
            d = 0.0
            for f in range(n_features):
                d += np.log(1 + abs(current[f] - features_matrix[f, i]))
            distances.append(d)
        if not distances:
            return 0.0
        mean_d = np.mean(distances)
        if np.isnan(mean_d) or mean_d == 0:
            return 0.0
        return 1.0 / mean_d

    # -------------------- Trade Decision --------------------
    def decide_and_trade(self, df, symbol):
        if df.empty or len(df) < 30:
            return

        features = self.compute_features_from_df(df)
        n = features.shape[1] - 1
        score = self.lorentzian_score(features, n)
        green_line = pd.Series(features[1, :]).ewm(span=3).mean().to_numpy()
        green_now = green_line[n]
        green_prev = green_line[n - 1] if n - 1 >= 0 else 0.0

        last = self.last_signal.get(symbol, 0.0)
        up_arrow = (score > self.lorentz_threshold) and (last <= self.lorentz_threshold)
        exit_signal = (green_prev > 0) and (green_now <= 0)

        print(f"[ANALYSE] {symbol} score={score:.4f} green_now={green_now:.4f} up_arrow={up_arrow} exit={exit_signal}")

        # Fetch option premium
        q = self.client.quotes(symbol=symbol, exchange=EXCHANGE)
        premium = float(q.get("data", {}).get("ltp", 0.0))

        # ---------------- Entry ----------------
        if up_arrow and symbol not in self.positions:
            lots = self.compute_max_lots_for_buy(premium)
            if lots > 0:
                qty = lots if API_EXPECTS_LOTS else lots * NIFTY_LOT_SIZE
                print(f"[ENTRY] BUY {symbol} at {premium} x {lots} lots")
                resp = self.client.placeorder(
                    strategy="Python",
                    symbol=symbol,
                    action="BUY",
                    exchange=EXCHANGE,
                    price_type=PRICE_TYPE,
                    product=PRODUCT,
                    quantity=qty
                )
                print(f"[ORDER-BUY] {resp}")
                self.positions[symbol] = {"entry_time": datetime.now(IST), "lots": lots, "qty": qty}
        # ---------------- Exit ----------------
        if exit_signal and symbol in self.positions:
            qty = self.positions[symbol]["qty"]
            print(f"[EXIT] SELL {symbol} qty={qty}")
            resp = self.client.placeorder(
                strategy="Python",
                symbol=symbol,
                action="SELL",
                exchange=EXCHANGE,
                price_type=PRICE_TYPE,
                product=PRODUCT,
                quantity=qty
            )
            print(f"[ORDER-EXIT] {resp}")
            self.positions.pop(symbol, None)

        self.last_signal[symbol] = score

    # -------------------- Tick Loop --------------------
    def tick(self):
        ce, pe, _, _ = self.get_current_target_symbols()
        end = datetime.now(IST)
        start = end - timedelta(days=2)
        sdate, edate = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        df_ce = self.fetch_history_for_symbol(ce, start_date=sdate, end_date=edate)
        df_pe = self.fetch_history_for_symbol(pe, start_date=sdate, end_date=edate)
        self.decide_and_trade(df_ce, ce)
        self.decide_and_trade(df_pe, pe)

    # -------------------- Force Close --------------------
    def force_close_all(self):
        print("[FORCE-CLOSE] Closing all positions.")
        for sym, pos in list(self.positions.items()):
            qty = pos["qty"]
            resp = self.client.placeorder(
                strategy="Python",
                symbol=sym,
                action="SELL",
                exchange=EXCHANGE,
                price_type=PRICE_TYPE,
                product=PRODUCT,
                quantity=qty
            )
            print(f"[FORCE-SELL] {sym} -> {resp}")
        self.positions.clear()

    def force_close_all_on_expiry(self):
        now = datetime.now(IST)
        expiry = self.current_weekly_tuesday(now).date()
        if now.date() == expiry:
            print("[EXPIRY] Closing all positions on expiry Tuesday.")
            self.force_close_all()

    # -------------------- Run Loop --------------------
    def run_loop(self, interval_seconds=15):
        try:
            while True:
                now = datetime.now(IST)
                if not self.market_started:
                    self.get_current_target_symbols()
                    self.market_started = True
                    print(f"[MARKET-START] ATM set for {self.base_symbol}")

                if now.time() >= dtime(15, 0):
                    print("Market closed. Exiting loop.")
                    self.force_close_all()
                    break

                self.tick()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            self.force_close_all()

# -------------------- Entrypoint --------------------
if __name__ == "__main__":
    API_KEY = "78a73982b9ea8e992cb182205107d74478d8b7edcadf7b249e061ed912aeb73b"
    engine = OptionBuyingEngine(api_key=API_KEY, hold_hours=2)
    engine.run_loop(interval_seconds=15)

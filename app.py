# -*- coding: utf-8 -*-
import streamlit as st
import asyncio
import aiohttp
import numpy as np
from collections import deque
from datetime import datetime
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Streamlit 페이지 설정
st.set_page_config(
    page_title="실시간 암호화폐 분석기",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 설정값들
CANDLE_HISTORY_SIZE = 200
MA_PERIODS = [9, 25, 99, 200]
BOLLINGER_PERIOD = 20
BOLLINGER_STD_DEV = 2
RSI_PERIOD = 14
VOLUME_MA_PERIOD = 20
INACTIVE_TRADE_VALUE_THRESHOLD = 100_000_000
SPREAD_RATE_WARN_THRESHOLD = 0.1

# 신규 지표 기준값
TURNOVER_HIGH_THRESHOLD = 100.0
TURNOVER_LOW_THRESHOLD = 30.0
VOLUME_CHANGE_HIGH_THRESHOLD = 20.0
TRADE_FREQ_HIGH_THRESHOLD = 5.0
TRADE_FREQ_LOW_THRESHOLD = 1.0
DEPTH_LARGE_THRESHOLD = 10_000_000_000
DEPTH_SMALL_THRESHOLD = 1_000_000_000

# 신호 분석 기준값
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
BTC_STRONG_OUTPERFORM = 3.0
BTC_STRONG_UNDERPERFORM = -3.0

class BithumbApi:
    BASE_URL = "https://api.bithumb.com/v1"

    def __init__(self):
        self.session = None

    async def create_session(self):
        timeout = aiohttp.ClientTimeout(total=15)
        self.session = aiohttp.ClientSession(timeout=timeout)

    async def close_session(self):
        if self.session:
            await self.session.close()

    async def fetch(self, endpoint: str) -> dict | None:
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            headers = {"accept": "application/json"}
            async with self.session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    st.error(f"API Error: Status {resp.status}")
                    return None
                data = await resp.json()
                return data
        except Exception as e:
            st.error(f"Network Error: {e}")
            return None

    async def get_ticker(self, symbol: str):
        """현재가 정보 조회"""
        return await self.fetch(f"ticker?markets=KRW-{symbol.upper()}")

    async def get_all_tickers(self):
        """전체 마켓 티커 정보"""
        return await self.fetch("ticker")

    async def get_orderbook(self, symbol: str):
        """호가 정보 조회"""
        return await self.fetch(f"orderbook?markets=KRW-{symbol.upper()}")

    async def get_transaction_history(self, symbol: str):
        """최근 체결 내역"""
        return await self.fetch(f"trades/ticks?market=KRW-{symbol.upper()}&count=200")

    async def get_candlestick(self, symbol: str, interval: str = "days"):
        """캔들스틱 데이터 조회"""
        if interval == "24h" or interval == "days":
            return await self.fetch(f"candles/days?market=KRW-{symbol.upper()}&count={CANDLE_HISTORY_SIZE}")
        elif interval == "1m":
            return await self.fetch(f"candles/minutes/1?market=KRW-{symbol.upper()}&count=1")
        else:
            return await self.fetch(f"candles/days?market=KRW-{symbol.upper()}&count={CANDLE_HISTORY_SIZE}")

    async def get_market_all(self):
        """마켓 코드 조회"""
        return await self.fetch("market/all?isDetails=false")

class DataProcessor:
    @staticmethod
    def calculate_ma(prices: list, period: int) -> float | None:
        if len(prices) < period: 
            return None
        return np.mean(prices[-period:])

    @staticmethod
    def calculate_bollinger_bands(prices: list, period: int, std_dev: int):
        if len(prices) < period: 
            return None, None, None
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        return sma + (std * std_dev), sma, sma - (std * std_dev)

    @staticmethod
    def calculate_rsi(prices: list, period: int) -> float | None:
        if len(prices) < period + 1: 
            return None
        deltas = np.diff(prices)
        gains = deltas[deltas >= 0]
        losses = -deltas[deltas < 0]
        avg_gain = np.mean(gains[-period:]) if len(gains) > 0 else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) > 0 else 1
        if avg_loss == 0: 
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_turnover_rate(trade_value_24h: float, market_cap: float) -> float | None:
        if market_cap and market_cap > 0:
            return (trade_value_24h / market_cap) * 100
        return None

    @staticmethod
    def calculate_volume_change_rate(current_volume: float, volume_ma: float) -> float | None:
        if volume_ma and volume_ma > 0:
            return ((current_volume - volume_ma) / volume_ma) * 100
        return None

    @staticmethod
    def calculate_orderbook_depth(orderbook: list):
        try:
            if not orderbook or len(orderbook) == 0:
                return None, None, None
                
            orderbook_data = orderbook[0]
            orderbook_units = orderbook_data.get("orderbook_units", [])
            if not orderbook_units:
                return None, None, None
            
            total_bid_amount = 0
            total_ask_amount = 0
            
            for unit in orderbook_units:
                bid_price = float(unit.get("bid_price", 0))
                bid_size = float(unit.get("bid_size", 0))
                ask_price = float(unit.get("ask_price", 0))
                ask_size = float(unit.get("ask_size", 0))
                
                total_bid_amount += bid_price * bid_size
                total_ask_amount += ask_price * ask_size
            
            total_depth = total_bid_amount + total_ask_amount
            return total_bid_amount, total_ask_amount, total_depth
            
        except (KeyError, ValueError, TypeError, IndexError):
            return None, None, None

    @staticmethod
    def calculate_real_trade_frequency(transactions: list) -> dict:
        if not transactions or len(transactions) == 0:
            return {
                'trades_per_minute': 0,
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'avg_trade_size': 0,
                'status': '거래 데이터 없음',
                'is_real_data': False,
                'time_range_minutes': 0
            }
        
        try:
            timestamps = []
            for transaction in transactions:
                try:
                    trade_date = transaction.get('trade_date_utc')
                    trade_time = transaction.get('trade_time_utc')
                    
                    if trade_date and trade_time:
                        datetime_str = f"{trade_date} {trade_time}"
                        dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
                        timestamp_ms = int(dt.timestamp() * 1000)
                        timestamps.append(timestamp_ms)
                    elif transaction.get('timestamp'):
                        timestamp_value = transaction.get('timestamp')
                        if isinstance(timestamp_value, (int, float)):
                            timestamps.append(int(timestamp_value))
                        elif isinstance(timestamp_value, str):
                            timestamps.append(int(timestamp_value))
                except (ValueError, TypeError):
                    continue
            
            if len(timestamps) >= 2:
                timestamps.sort()
                time_diff_ms = timestamps[-1] - timestamps[0]
                time_range_minutes = max(time_diff_ms / (1000 * 60), 1.0)
            else:
                time_range_minutes = 10.0
            
            total_trades = len(transactions)
            buy_trades = 0
            sell_trades = 0
            trade_sizes = []
            
            for transaction in transactions:
                try:
                    trade_type = transaction.get('ask_bid', '').upper()
                    
                    if trade_type == 'BID':
                        buy_trades += 1
                    elif trade_type == 'ASK':
                        sell_trades += 1
                    
                    trade_volume = float(transaction.get('trade_volume', 0))
                    trade_price = float(transaction.get('trade_price', 0))
                    trade_size = trade_volume * trade_price
                    
                    if trade_size > 0:
                        trade_sizes.append(trade_size)
                        
                except (ValueError, TypeError, KeyError):
                    continue
            
            avg_trade_size = np.mean(trade_sizes) if trade_sizes else 0
            trades_per_minute = total_trades / time_range_minutes
            
            if time_range_minutes <= 5:
                if trades_per_minute >= 30:
                    status = "극도로 활발"
                elif trades_per_minute >= 20:
                    status = "매우 활발"
                else:
                    status = "활발"
            elif time_range_minutes <= 60:
                if trades_per_minute >= 5:
                    status = "매우 활발"
                elif trades_per_minute >= 2:
                    status = "활발"
                else:
                    status = "보통"
            elif time_range_minutes <= 360:
                if trades_per_minute >= 1:
                    status = "활발"
                elif trades_per_minute >= 0.5:
                    status = "보통"
                else:
                    status = "저조"
            else:
                if trades_per_minute >= 0.5:
                    status = "보통"
                elif trades_per_minute >= 0.1:
                    status = "저조"
                else:
                    status = "매우 저조"
            
            return {
                'trades_per_minute': round(trades_per_minute, 1),
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'avg_trade_size': avg_trade_size,
                'status': status,
                'is_real_data': True,
                'time_range_minutes': round(time_range_minutes, 1)
            }
            
        except Exception as e:
            return {
                'trades_per_minute': 0,
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'avg_trade_size': 0,
                'status': '계산 오류',
                'is_real_data': False,
                'time_range_minutes': 0
            }

    @staticmethod
    def calculate_btc_relative_strength(coin_change_rate: float, btc_change_rate: float) -> float | None:
        if btc_change_rate is not None:
            return coin_change_rate - btc_change_rate
        return None

class SignalAnalyzer:
    def calculate_signal_score(self, data: dict):
        score = 0
        max_score = 10
        positive_signals = []
        negative_signals = []

        # 1. 회전율 평가 (0-2점)
        turnover = data.get('turnover_rate')
        if turnover is not None:
            if turnover >= TURNOVER_HIGH_THRESHOLD:
                score += 2
                positive_signals.append(f"회전율 {turnover:.1f}% - 매우 활발한 거래")
            elif turnover >= TURNOVER_LOW_THRESHOLD:
                score += 1
                positive_signals.append(f"회전율 {turnover:.1f}% - 적당한 거래")
            else:
                negative_signals.append(f"회전율 {turnover:.1f}% - 거래 저조")

        # 2. 거래량 증가율 평가 (0-2점)
        volume_change = data.get('volume_change_rate')
        if volume_change is not None:
            if volume_change >= VOLUME_CHANGE_HIGH_THRESHOLD:
                score += 2
                positive_signals.append(f"거래량 {volume_change:+.1f}% - 관심도 급증")
            elif volume_change >= 5.0:
                score += 1
                positive_signals.append(f"거래량 {volume_change:+.1f}% - 거래 증가")
            elif volume_change <= -VOLUME_CHANGE_HIGH_THRESHOLD:
                negative_signals.append(f"거래량 {volume_change:+.1f}% - 관심도 급감")

        # 3. BTC 대비 강도 평가 (0-2점)
        btc_strength = data.get('btc_relative_strength')
        if btc_strength is not None:
            if btc_strength >= BTC_STRONG_OUTPERFORM:
                score += 2
                positive_signals.append(f"BTC 대비 {btc_strength:+.1f}% - 독립적 강세")
            elif btc_strength >= 1.0:
                score += 1
                positive_signals.append(f"BTC 대비 {btc_strength:+.1f}% - 상대적 강세")
            elif btc_strength <= BTC_STRONG_UNDERPERFORM:
                negative_signals.append(f"BTC 대비 {btc_strength:+.1f}% - 독립적 약세")

        # 4. 기술적 지표 평가 (0-2점)
        rsi = data.get('rsi')
        mas = data.get('mas', {})
        price = data.get('price', 0)

        tech_score = 0
        if rsi is not None:
            if 40 <= rsi <= 70:
                tech_score += 1
                positive_signals.append(f"RSI {rsi:.1f} - 건전한 모멘텀")
            elif rsi > RSI_OVERBOUGHT:
                negative_signals.append(f"RSI {rsi:.1f} - 과매수 위험")
            elif rsi < RSI_OVERSOLD:
                positive_signals.append(f"RSI {rsi:.1f} - 과매도 반등 기대")

        ma_above_count = 0
        for period, ma_value in mas.items():
            if ma_value is not None and price > ma_value:
                ma_above_count += 1

        if ma_above_count == len(mas) and len(mas) > 0:
            tech_score += 1
            positive_signals.append("모든 이동평균선 상회 - 상승 추세")

        score += tech_score

        # 5. 실제 거래 활성도 평가 (0-2점)
        total_depth = data.get('total_depth')
        trade_freq_data = data.get('trade_frequency_data', {})

        liquidity_score = 0
        if total_depth is not None:
            if total_depth >= DEPTH_LARGE_THRESHOLD:
                liquidity_score += 1
                positive_signals.append("대형 호가창 - 안정적 유동성")
            elif total_depth < DEPTH_SMALL_THRESHOLD:
                negative_signals.append("소형 호가창 - 슬리피지 위험")

        if trade_freq_data.get('is_real_data', False):
            trades_per_minute = trade_freq_data.get('trades_per_minute', 0)
            if trades_per_minute >= TRADE_FREQ_HIGH_THRESHOLD:
                liquidity_score += 1
                positive_signals.append(f"실제 거래빈도 {trades_per_minute:.1f}건/분 - 활발한 거래")
            elif trades_per_minute < TRADE_FREQ_LOW_THRESHOLD:
                negative_signals.append(f"실제 거래빈도 {trades_per_minute:.1f}건/분 - 거래 부진")

        score += liquidity_score
        return score, max_score, positive_signals, negative_signals

    def get_investment_signal(self, score: int, max_score: int):
        percentage = (score / max_score) * 100
        if percentage >= 70:
            return "🟢", "강한 매수", "매수"
        elif percentage >= 50:
            return "🟡", "약한 매수", "관심"
        elif percentage >= 30:
            return "⚪", "중립", "관망"
        else:
            return "🔴", "약세", "주의"

    def get_risk_assessment(self, data: dict):
        risk_factors = []
        rsi = data.get('rsi')
        if rsi is not None and rsi > RSI_OVERBOUGHT:
            risk_factors.append("RSI 과매수 - 단기 조정 가능")

        total_depth = data.get('total_depth')
        if total_depth is not None and total_depth < DEPTH_SMALL_THRESHOLD:
            risk_factors.append("유동성 부족 - 대량 거래 시 가격 영향")

        spread_rate = data.get('spread_rate', 0)
        if spread_rate > SPREAD_RATE_WARN_THRESHOLD:
            risk_factors.append(f"호가 스프레드 {spread_rate:.3f}% - 거래비용 높음")

        if data.get('is_inactive', False):
            risk_factors.append("거래 비활성 - 유동성 극히 부족")

        if len(risk_factors) >= 3:
            risk_level = "🔴 높음"
        elif len(risk_factors) >= 1:
            risk_level = "🟡 중간"
        else:
            risk_level = "🟢 낮음"

        return risk_level, risk_factors

class CoinAnalyzer:
    def __init__(self, symbol: str, api: BithumbApi):
        self.symbol = symbol
        self.api = api
        self.processor = DataProcessor()
        self.signal_analyzer = SignalAnalyzer()
        self.candle_history = deque(maxlen=CANDLE_HISTORY_SIZE)
        self.volume_history = deque(maxlen=CANDLE_HISTORY_SIZE)

    async def initialize_history(self):
        initial_candles = await self.api.get_candlestick(self.symbol, "days")
        if initial_candles and isinstance(initial_candles, list):
            for candle in initial_candles[-CANDLE_HISTORY_SIZE:]:
                if isinstance(candle, dict):
                    self.candle_history.append(float(candle.get('trade_price', 0)))
                    self.volume_history.append(float(candle.get('candle_acc_trade_volume', 0)))

    async def calculate_strength(self) -> float | None:
        transactions = await self.api.get_transaction_history(self.symbol)
        if not transactions or not isinstance(transactions, list): 
            return None
        
        buy_volume = 0
        sell_volume = 0
        
        for t in transactions:
            if isinstance(t, dict):
                volume = float(t.get("trade_volume", 0))
                trade_type = t.get("ask_bid", "").upper()
                
                if trade_type == "BID":
                    buy_volume += volume
                elif trade_type == "ASK":
                    sell_volume += volume
        
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            return (buy_volume / total_volume) * 100
        return None

    async def run_analysis(self):
        # 먼저 마켓 코드 확인
        market_check = await self.api.get_ticker(self.symbol)
        if not market_check or isinstance(market_check, Exception):
            return {"error": f"'{self.symbol}' 코인을 찾을 수 없습니다. 올바른 심볼인지 확인해주세요."}
        
        try:
            results = await asyncio.gather(
                self.api.get_ticker(self.symbol),
                self.api.get_orderbook(self.symbol),
                self.api.get_all_tickers(),
                self.api.get_transaction_history(self.symbol),
                return_exceptions=True
            )
            ticker, orderbook, all_tickers, transactions = results

        except Exception as e:
            return {"error": f"API 요청 실패: {e}"}

        if not ticker or isinstance(ticker, Exception) or not orderbook or isinstance(orderbook, Exception):
            return {"error": "필수 데이터 수신 실패"}

        try:
            if isinstance(ticker, list) and len(ticker) > 0:
                ticker_data = ticker[0]
            else:
                return {"error": "티커 데이터 형식 오류"}
            
            closing_price = float(ticker_data.get("trade_price", 0))
            change_rate = float(ticker_data.get("signed_change_rate", 0)) * 100
            acc_trade_price_24h = float(ticker_data.get("acc_trade_price_24h", 0))
            acc_trade_volume_24h = float(ticker_data.get("acc_trade_volume_24h", 0))
            
            is_inactive = acc_trade_price_24h < INACTIVE_TRADE_VALUE_THRESHOLD
            
        except (ValueError, KeyError, TypeError) as e:
            return {"error": f"티커 데이터 파싱 오류: {e}"}

        # BTC 데이터 및 거래량 순위 찾기
        btc_change_rate = None
        volume_rank, total_coins = None, None
        
        if all_tickers and not isinstance(all_tickers, Exception):
            try:
                # 빗썸 API는 항상 list 형태로 응답
                if isinstance(all_tickers, list) and len(all_tickers) > 0:
                    # BTC 데이터 찾기
                    for ticker_item in all_tickers:
                        if isinstance(ticker_item, dict):
                            market = ticker_item.get("market", "")
                            if market == "KRW-BTC":
                                try:
                                    signed_change = ticker_item.get("signed_change_rate", 0)
                                    btc_change_rate = float(signed_change) * 100
                                except (ValueError, TypeError):
                                    pass
                                break
                    
                    # 거래량으로 정렬하여 순위 계산
                    try:
                        # acc_trade_price_24h로 정렬
                        sorted_tickers = sorted(all_tickers, 
                                              key=lambda item: float(item.get('acc_trade_price_24h', 0)), 
                                              reverse=True)
                        total_coins = len(sorted_tickers)
                        
                        # 현재 코인의 순위 찾기
                        current_market = f"KRW-{self.symbol}"
                        for rank, ticker_item in enumerate(sorted_tickers, 1):
                            market = ticker_item.get("market", "")
                            if market == current_market:
                                volume_rank = rank
                                break
                    except Exception:
                        pass
                        
            except Exception:
                pass
        
        # 전체 티커가 없거나 BTC 데이터가 없으면 직접 요청
        if btc_change_rate is None:
            try:
                btc_ticker = await self.api.get_ticker("BTC")
                if btc_ticker and isinstance(btc_ticker, list) and len(btc_ticker) > 0:
                    btc_data = btc_ticker[0]
                    btc_signed_change = btc_data.get("signed_change_rate", 0)
                    btc_change_rate = float(btc_signed_change) * 100
            except Exception:
                pass

        strength = await self.calculate_strength() if not is_inactive else None

        # 기술적 지표 계산
        prices, volumes = list(self.candle_history), list(self.volume_history)
        mas = {p: self.processor.calculate_ma(prices, p) for p in MA_PERIODS}
        bb_upper, bb_mid, bb_lower = self.processor.calculate_bollinger_bands(prices, BOLLINGER_PERIOD, BOLLINGER_STD_DEV)
        rsi = self.processor.calculate_rsi(prices, RSI_PERIOD)
        volume_ma = self.processor.calculate_ma(volumes, VOLUME_MA_PERIOD)

        # 신규 지표 계산
        market_cap = closing_price * acc_trade_volume_24h if acc_trade_volume_24h > 0 else None
        turnover_rate = self.processor.calculate_turnover_rate(acc_trade_price_24h, market_cap)
        current_volume = volumes[-1] if volumes else 0
        volume_change_rate = self.processor.calculate_volume_change_rate(current_volume, volume_ma)
        bid_depth, ask_depth, total_depth = self.processor.calculate_orderbook_depth(orderbook)
        
        trade_frequency_data = self.processor.calculate_real_trade_frequency(
            transactions if not isinstance(transactions, Exception) else None
        )
        
        btc_relative_strength = self.processor.calculate_btc_relative_strength(change_rate, btc_change_rate)

        # 호가창 분석
        try:
            if isinstance(orderbook, list) and len(orderbook) > 0:
                orderbook_data = orderbook[0]
                orderbook_units = orderbook_data.get("orderbook_units", [])
                
                if orderbook_units:
                    total_bid_qty = sum(float(unit.get("bid_size", 0)) for unit in orderbook_units)
                    total_ask_qty = sum(float(unit.get("ask_size", 0)) for unit in orderbook_units)
                    total_order_qty = total_bid_qty + total_ask_qty
                    bid_ratio = (total_bid_qty / total_order_qty) * 100 if total_order_qty else 0
                    
                    best_bid = float(orderbook_units[0].get("bid_price", 0))
                    best_ask = float(orderbook_units[0].get("ask_price", 0))
                    spread = best_ask - best_bid
                    spread_rate = (spread / closing_price) * 100 if closing_price else 0
                else:
                    bid_ratio, spread, spread_rate = 50.0, 0.0, 0.0
            else:
                bid_ratio, spread, spread_rate = 50.0, 0.0, 0.0
        except (ValueError, KeyError, IndexError, TypeError):
            bid_ratio, spread, spread_rate = 50.0, 0.0, 0.0

        # 분석 데이터 구성
        analysis_data = {
            "symbol": self.symbol, "timestamp": datetime.now(), "price": closing_price,
            "change_rate": change_rate, "value_24h": acc_trade_price_24h, "is_inactive": is_inactive,
            "strength": strength, "mas": mas, "bollinger_bands": {"upper": bb_upper, "middle": bb_mid, "lower": bb_lower},
            "rsi": rsi, "volume": current_volume, "volume_ma": volume_ma,
            "bid_ratio": bid_ratio, "spread": spread, "spread_rate": spread_rate,
            "volume_rank": volume_rank, "total_coins": total_coins,
            "turnover_rate": turnover_rate, "volume_change_rate": volume_change_rate,
            "bid_depth": bid_depth, "ask_depth": ask_depth, "total_depth": total_depth,
            "trade_frequency_data": trade_frequency_data, "btc_relative_strength": btc_relative_strength,
            "btc_change_rate": btc_change_rate,
        }

        # 신호 분석
        signal_score, max_score, positive_signals, negative_signals = self.signal_analyzer.calculate_signal_score(analysis_data)
        signal_color, signal_text, signal_action = self.signal_analyzer.get_investment_signal(signal_score, max_score)
        risk_level, risk_factors = self.signal_analyzer.get_risk_assessment(analysis_data)

        analysis_data.update({
            "signal_score": signal_score, "signal_max_score": max_score,
            "signal_color": signal_color, "signal_text": signal_text,
            "positive_signals": positive_signals, "negative_signals": negative_signals,
            "risk_level": risk_level, "risk_factors": risk_factors,
        })

        return analysis_data

# Streamlit UI 구성
def main():
    st.title("🪙 실시간 암호화폐 분석기")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.title("설정")
    symbol = st.sidebar.text_input("코인 심볼", value="BTC", help="예: BTC, ETH, GMT").upper()
    auto_refresh = st.sidebar.checkbox("자동 새로고침 (60초)", value=True)
    
    if st.sidebar.button("분석 시작") or auto_refresh:
        # 세션 상태 초기화
        if 'api' not in st.session_state:
            st.session_state.api = None
        
        # API 초기화
        @st.cache_resource
        def get_api():
            return BithumbApi()
        
        api = get_api()
        
        # 분석 실행
        async def run_analysis_async():
            await api.create_session()
            analyzer = CoinAnalyzer(symbol, api)
            await analyzer.initialize_history()
            result = await analyzer.run_analysis()
            await api.close_session()
            return result
        
        # asyncio 실행
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysis_data = loop.run_until_complete(run_analysis_async())
            loop.close()
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
            return
        
        if analysis_data.get("error"):
            st.error(f"분석 오류: {analysis_data['error']}")
            return
        
        # 결과 표시
        display_analysis_results(analysis_data)
        
        # 자동 새로고침
        if auto_refresh:
            time.sleep(1)
            st.rerun()

def display_analysis_results(data: dict):
    """분석 결과를 Streamlit UI로 표시"""
    
    # 메인 정보 헤더
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"{data['symbol']} 현재가",
            value=f"{data['price']:,.0f} KRW",
            delta=f"{data['change_rate']:+.2f}%"
        )
    
    with col2:
        signal_color = data.get('signal_color', '⚪')
        signal_text = data.get('signal_text', '중립')
        st.metric(
            label="투자 신호",
            value=f"{signal_color} {signal_text}",
            delta=f"{data.get('signal_score', 0)}/{data.get('signal_max_score', 10)}점"
        )
    
    with col3:
        volume_rank = data.get('volume_rank')
        total_coins = data.get('total_coins')
        if volume_rank and total_coins:
            st.metric(
                label="거래량 순위",
                value=f"{volume_rank}위",
                delta=f"총 {total_coins}개"
            )
        else:
            st.metric(label="거래량 순위", value="N/A")
    
    with col4:
        risk_level = data.get('risk_level', '🟡 중간')
        st.metric(
            label="위험도",
            value=risk_level.split(' ')[1] if ' ' in risk_level else risk_level,
            delta=risk_level.split(' ')[0] if ' ' in risk_level else ''
        )
    
    st.markdown("---")
    
    # 탭으로 구분
    tab1, tab2, tab3, tab4 = st.tabs(["📊 시장 분석", "📈 기술적 지표", "🔍 유동성 분석", "🎯 투자 신호"])
    
    with tab1:
        display_market_analysis(data)
    
    with tab2:
        display_technical_analysis(data)
    
    with tab3:
        display_liquidity_analysis(data)
    
    with tab4:
        display_signal_analysis(data)

def display_market_analysis(data: dict):
    """시장 분석 탭"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("기본 정보")
        
        # 기본 정보 테이블
        basic_info = {
            "현재가": f"{data['price']:,.0f} KRW",
            "24시간 변동": f"{data['change_rate']:+.2f}%",
            "24시간 거래대금": f"{data['value_24h']:,.0f} KRW",
            "거래 상태": "비활성" if data['is_inactive'] else "활성"
        }
        
        for key, value in basic_info.items():
            st.write(f"**{key}**: {value}")
    
    with col2:
        st.subheader("매수/매도 압력")
        
        strength = data.get('strength')
        if strength is not None:
            st.write(f"**체결강도**: {strength:.2f}% (매수체결비율)")
        else:
            st.write("**체결강도**: N/A")
        
        bid_ratio = data.get('bid_ratio', 50)
        st.write(f"**호가비율**: 매수 {bid_ratio:.1f}% | 매도 {100-bid_ratio:.1f}%")
        
        # 호가 비율 차트
        fig_bid = go.Figure(data=[
            go.Bar(name='매수', x=['호가'], y=[bid_ratio], marker_color='green'),
            go.Bar(name='매도', x=['호가'], y=[100-bid_ratio], marker_color='red')
        ])
        fig_bid.update_layout(
            title="호가창 매수/매도 비율",
            yaxis_title="비율 (%)",
            barmode='stack',
            height=300
        )
        st.plotly_chart(fig_bid, use_container_width=True)
    
    # BTC 대비 성과
    if data.get('btc_relative_strength') is not None and data.get('btc_change_rate') is not None:
        st.subheader("BTC 대비 성과")
        btc_strength = data['btc_relative_strength']
        btc_rate = data['btc_change_rate']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BTC 변동률", f"{btc_rate:+.2f}%")
        with col2:
            st.metric(f"{data['symbol']} 변동률", f"{data['change_rate']:+.2f}%")
        with col3:
            st.metric("상대적 강도", f"{btc_strength:+.2f}%")

def display_technical_analysis(data: dict):
    """기술적 지표 탭"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("이동평균선")
        mas = data.get('mas', {})
        price = data.get('price', 0)
        
        ma_data = []
        for period in MA_PERIODS:
            ma_value = mas.get(period)
            if ma_value is not None:
                position = "상회" if price > ma_value else "하회"
                deviation = ((price - ma_value) / ma_value) * 100
                ma_data.append({
                    "기간": f"MA({period})",
                    "값": f"{ma_value:,.0f}",
                    "위치": position,
                    "편차": f"{deviation:+.1f}%"
                })
        
        if ma_data:
            df_ma = pd.DataFrame(ma_data)
            st.dataframe(df_ma, use_container_width=True)
        
        # RSI
        rsi = data.get('rsi')
        if rsi is not None:
            st.subheader("RSI (상대강도지수)")
            
            # RSI 게이지 차트
            fig_rsi = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = rsi,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"RSI({RSI_PERIOD})"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_rsi.update_layout(height=300)
            st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col2:
        st.subheader("볼린저 밴드")
        bb = data.get('bollinger_bands', {})
        if all(bb.values()):
            position = "상단 돌파" if price > bb['upper'] else "하단 이탈" if price < bb['lower'] else "밴드 내"
            
            bb_data = {
                "상단": f"{bb['upper']:,.0f}",
                "중간": f"{bb['middle']:,.0f}",
                "하단": f"{bb['lower']:,.0f}",
                "현재가 위치": position
            }
            
            for key, value in bb_data.items():
                st.write(f"**{key}**: {value}")
        
        st.subheader("거래량 분석")
        volume = data.get('volume', 0)
        volume_ma = data.get('volume_ma')
        
        if volume_ma is not None:
            volume_status = "평균 상회" if volume > volume_ma else "평균 하회"
            st.write(f"**현재 거래량**: {volume:,.0f}")
            st.write(f"**평균 대비**: {volume_status}")
            
            # 거래량 차트
            volume_data = pd.DataFrame({
                '구분': ['현재 거래량', f'{VOLUME_MA_PERIOD}일 평균'],
                '거래량': [volume, volume_ma]
            })
            
            fig_volume = px.bar(volume_data, x='구분', y='거래량', 
                              title="거래량 비교")
            st.plotly_chart(fig_volume, use_container_width=True)

def display_liquidity_analysis(data: dict):
    """유동성 분석 탭"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("회전율 및 거래 활성도")
        
        turnover_rate = data.get('turnover_rate')
        if turnover_rate is not None:
            st.metric("회전율 (시총대비)", f"{turnover_rate:.1f}%")
            
            # 회전율 상태
            if turnover_rate >= TURNOVER_HIGH_THRESHOLD:
                st.success("🔥 매우 활발한 거래")
            elif turnover_rate <= TURNOVER_LOW_THRESHOLD:
                st.warning("⚠️ 거래 저조")
            else:
                st.info("📊 보통 수준")
        
        volume_change = data.get('volume_change_rate')
        if volume_change is not None:
            st.metric("거래량 증가율 (24H)", f"{volume_change:+.1f}%")
            
            if volume_change >= VOLUME_CHANGE_HIGH_THRESHOLD:
                st.success("⬆️⬆️ 관심도 급증")
            elif volume_change <= -VOLUME_CHANGE_HIGH_THRESHOLD:
                st.error("⬇️⬇️ 관심도 급감")
    
    with col2:
        st.subheader("호가창 깊이")
        
        total_depth = data.get('total_depth')
        bid_depth = data.get('bid_depth')
        ask_depth = data.get('ask_depth')
        
        if all([total_depth, bid_depth, ask_depth]):
            st.metric("총 호가창 깊이", f"{total_depth:,.0f} 원")
            
            # 호가창 깊이 차트
            depth_data = pd.DataFrame({
                '구분': ['매수 호가', '매도 호가'],
                '금액': [bid_depth, ask_depth]
            })
            
            fig_depth = px.pie(depth_data, values='금액', names='구분',
                             title="호가창 구성 비율")
            st.plotly_chart(fig_depth, use_container_width=True)
            
            # 깊이 상태
            if total_depth >= DEPTH_LARGE_THRESHOLD:
                st.success("💎 대형 호가창 - 안정적 유동성")
            elif total_depth >= DEPTH_SMALL_THRESHOLD:
                st.info("💰 중형 호가창")
            else:
                st.warning("📊 소형 호가창 - 슬리피지 위험")
    
    # 실제 거래 빈도
    st.subheader("실제 거래 빈도")
    trade_freq_data = data.get('trade_frequency_data', {})
    
    if trade_freq_data.get('is_real_data', False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("분당 거래건수", f"{trade_freq_data.get('trades_per_minute', 0):.1f}건")
        
        with col2:
            st.metric("총 거래건수", f"{trade_freq_data.get('total_trades', 0)}건")
        
        with col3:
            st.metric("매수 거래", f"{trade_freq_data.get('buy_trades', 0)}건")
        
        with col4:
            st.metric("매도 거래", f"{trade_freq_data.get('sell_trades', 0)}건")
        
        status = trade_freq_data.get('status', '')
        if '매우 활발' in status:
            st.success(f"⚡⚡ {status}")
        elif '활발' in status:
            st.success(f"⚡ {status}")
        elif '저조' in status:
            st.warning(f"💤 {status}")
        else:
            st.info(f"➡️ {status}")

def display_signal_analysis(data: dict):
    """투자 신호 탭"""
    
    # 종합 신호
    st.subheader("🚀 종합 투자 신호")
    
    score = data.get('signal_score', 0)
    max_score = data.get('signal_max_score', 10)
    signal_color = data.get('signal_color', '⚪')
    signal_text = data.get('signal_text', '중립')
    
    # 신호 점수 프로그레스 바
    progress_value = score / max_score if max_score > 0 else 0
    st.progress(progress_value)
    st.write(f"**종합 신호**: {signal_color} {signal_text} ({score}/{max_score}점)")
    
    # 위험도
    risk_level = data.get('risk_level', '🟡 중간')
    st.write(f"**위험도 평가**: {risk_level}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 긍정적 요소")
        positive_signals = data.get('positive_signals', [])
        if positive_signals:
            for signal in positive_signals:
                st.success(f"✅ {signal}")
        else:
            st.info("현재 특별한 긍정적 요소가 없습니다.")
    
    with col2:
        st.subheader("⚠️ 주의 요소")
        negative_signals = data.get('negative_signals', [])
        risk_factors = data.get('risk_factors', [])
        
        warning_items = negative_signals + risk_factors
        if warning_items:
            for warning in warning_items:
                st.warning(f"🔸 {warning}")
        else:
            st.success("현재 특별한 위험 요소가 없습니다.")
    
    # 거래 전략 (간단한 추천)
    st.subheader("💡 거래 가이드")
    
    if "강한 매수" in signal_text:
        st.success("🔵 적극 매수 고려 (2-3회 분할)")
        st.info("🎯 목표: +15~25% | 🛑 손절: -8%")
        st.info("⏰ 투자기간: 1-3주 (단기 스윙)")
    elif "약한 매수" in signal_text:
        st.info("🟡 신중 매수 고려 (3-4회 분할)")
        st.info("🎯 목표: +8~15% | 🛑 손절: -6%")
        st.info("⏰ 투자기간: 2-4주 (중기)")
    elif "중립" in signal_text:
        st.warning("⚪ 관망 또는 DCA 고려")
        st.info("🎯 변동성 거래 | 🛑 -5% 엄격 준수")
    else:
        st.error("🔴 진입 금지")
        st.info("🎯 매도 고려 | 🛑 손절 우선")

if __name__ == "__main__":
    main()

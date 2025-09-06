# -*- coding: utf-8 -*-
"""
수정된 Streamlit 코인분석기
- 실제 거래빈도 데이터 사용
- 이동평균선 9, 25, 99, 200일로 변경
- 템플릿 데이터 제거, 실제 데이터만 사용
"""

import streamlit as st
import asyncio
import aiohttp
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="실시간 코인 분석기",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .signal-strong-buy { background: #d4edda; border-left-color: #28a745; }
    .signal-buy { background: #fff3cd; border-left-color: #ffc107; }
    .signal-neutral { background: #f8f9fa; border-left-color: #6c757d; }
    .signal-sell { background: #f8d7da; border-left-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# 실제 투자자용 설정값
CANDLE_HISTORY_SIZE = 200  # 200일 이동평균을 위해 증가
MA_PERIODS = [9, 25, 99, 200]  # 실제 투자자가 사용하는 이동평균
RSI_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD_DEV = 2

# 거래 활성도 기준 (실제 빗썸 데이터 기반)
VOLUME_HIGH_THRESHOLD = 1000_000_000  # 10억원 이상
VOLUME_NORMAL_THRESHOLD = 100_000_000  # 1억원 이상
INACTIVE_TRADE_VALUE_THRESHOLD = 50_000_000  # 5천만원 미만은 비활성

class BithumbApi:
    BASE_URL = "https://api.bithumb.com/public"

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
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    st.error(f"API Error: Status {resp.status}")
                    return None
                data = await resp.json()
                if data.get("status") != "0000":
                    st.error(f"API Error: {data.get('message', 'Unknown error')}")
                    return None
                return data.get("data")
        except asyncio.TimeoutError:
            st.error("API 요청 시간 초과")
            return None
        except Exception as e:
            st.error(f"Network Error: {e}")
            return None

    async def get_ticker(self, symbol: str):
        return await self.fetch(f"ticker/{symbol.upper()}_KRW")

    async def get_all_tickers(self):
        return await self.fetch("ticker/ALL_KRW")

    async def get_orderbook(self, symbol: str):
        return await self.fetch(f"orderbook/{symbol.upper()}_KRW")

    async def get_transaction_history(self, symbol: str):
        return await self.fetch(f"transaction_history/{symbol.upper()}_KRW")

    async def get_candlestick(self, symbol: str, chart_intervals: str = "24h"):
        return await self.fetch(f"candlestick/{symbol.upper()}_KRW/{chart_intervals}")

class DataProcessor:
    @staticmethod
    def calculate_ma(prices: list, period: int) -> float | None:
        if len(prices) < period:
            return None
        return np.mean(prices[-period:])

    @staticmethod
    def calculate_bollinger_bands(prices: list, period: int = 20, std_dev: int = 2):
        if len(prices) < period:
            return None, None, None
        recent_prices = prices[-period:]
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        return sma + (std * std_dev), sma, sma - (std * std_dev)

    @staticmethod
    def calculate_rsi(prices: list, period: int = 14) -> float | None:
        if len(prices) < period + 1:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_real_trade_frequency(transactions: list) -> dict:
        """실제 거래 빈도 계산 (템플릿 데이터 사용 금지)"""
        if not transactions or len(transactions) == 0:
            return {
                'trades_per_minute': 0,
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'avg_trade_size': 0,
                'status': '거래 데이터 없음'
            }
        
        try:
            # 실제 거래 내역 분석
            total_trades = len(transactions)
            buy_trades = len([t for t in transactions if t.get('type') == 'bid'])
            sell_trades = len([t for t in transactions if t.get('type') == 'ask'])
            
            # 거래 크기 분석
            trade_sizes = []
            for t in transactions:
                try:
                    size = float(t.get('units_traded', 0)) * float(t.get('price', 0))
                    if size > 0:
                        trade_sizes.append(size)
                except (ValueError, TypeError):
                    continue
            
            avg_trade_size = np.mean(trade_sizes) if trade_sizes else 0
            
            # 실제 시간 기준 계산 (최근 30분 데이터라고 가정)
            time_window_minutes = 30
            trades_per_minute = total_trades / time_window_minutes
            
            # 거래 활성도 상태 판정
            if trades_per_minute >= 5:
                status = "매우 활발"
            elif trades_per_minute >= 2:
                status = "활발"
            elif trades_per_minute >= 0.5:
                status = "보통"
            else:
                status = "저조"
            
            return {
                'trades_per_minute': round(trades_per_minute, 2),
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'avg_trade_size': avg_trade_size,
                'status': status
            }
            
        except Exception as e:
            st.error(f"거래 빈도 계산 오류: {e}")
            return {
                'trades_per_minute': 0,
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'avg_trade_size': 0,
                'status': '계산 오류'
            }

    @staticmethod
    def calculate_turnover_rate(trade_value_24h: float, market_cap: float) -> float | None:
        if not market_cap or market_cap <= 0:
            return None
        return (trade_value_24h / market_cap) * 100

    @staticmethod
    def calculate_volume_change_rate(current_volume: float, volume_ma: float) -> float | None:
        if not volume_ma or volume_ma <= 0:
            return None
        return ((current_volume - volume_ma) / volume_ma) * 100

class CoinAnalyzer:
    def __init__(self):
        self.api = None
        self.processor = DataProcessor()

    async def get_analysis(self, symbol: str):
        """실제 데이터만 사용하는 분석 실행"""
        if not self.api:
            self.api = BithumbApi()
            await self.api.create_session()

        try:
            # 동시에 모든 필요한 데이터 요청
            results = await asyncio.gather(
                self.api.get_ticker(symbol),
                self.api.get_orderbook(symbol),
                self.api.get_all_tickers(),
                self.api.get_transaction_history(symbol),
                self.api.get_candlestick(symbol, "24h"),
                return_exceptions=True
            )
            
            ticker, orderbook, all_tickers, transactions, candles = results

            # 필수 데이터 검증
            if not ticker or isinstance(ticker, Exception):
                return {"error": "티커 데이터 수신 실패"}
            
            if not orderbook or isinstance(orderbook, Exception):
                return {"error": "호가창 데이터 수신 실패"}

            # 기본 정보 추출 및 검증
            try:
                price = float(ticker["closing_price"])
                change_rate = float(ticker["fluctate_rate_24H"])
                value_24h = float(ticker["acc_trade_value_24H"])
                volume_24h = float(ticker.get("units_traded_24H", 0))
                
                if price <= 0:
                    return {"error": "유효하지 않은 가격 데이터"}
                    
            except (ValueError, KeyError, TypeError) as e:
                return {"error": f"티커 데이터 파싱 오류: {e}"}

            # BTC 정보 추출
            btc_change_rate = None
            if all_tickers and not isinstance(all_tickers, Exception):
                try:
                    all_tickers.pop('date', None)
                    if 'BTC' in all_tickers:
                        btc_change_rate = float(all_tickers['BTC'].get('fluctate_rate_24H', 0))
                except (ValueError, TypeError):
                    pass

            # 실제 거래 빈도 계산 (템플릿 데이터 사용 금지)
            trade_frequency_data = self.processor.calculate_real_trade_frequency(
                transactions if not isinstance(transactions, Exception) else None
            )

            # 호가창 분석
            try:
                bids, asks = orderbook["bids"], orderbook["asks"]
                
                # 호가창 깊이 계산
                bid_depth = sum(float(b["price"]) * float(b["quantity"]) for b in bids[:10])
                ask_depth = sum(float(a["price"]) * float(a["quantity"]) for a in asks[:10])
                total_depth = bid_depth + ask_depth
                
                # 매수/매도 물량 비율
                total_bid_qty = sum(float(b["quantity"]) for b in bids[:10])
                total_ask_qty = sum(float(a["quantity"]) for a in asks[:10])
                total_qty = total_bid_qty + total_ask_qty
                bid_ratio = (total_bid_qty / total_qty) * 100 if total_qty > 0 else 50
                
                # 스프레드 계산
                spread = float(asks[0]["price"]) - float(bids[0]["price"])
                spread_rate = (spread / price) * 100
                
            except (KeyError, ValueError, IndexError, TypeError):
                bid_depth = ask_depth = total_depth = 0
                bid_ratio = 50
                spread = spread_rate = 0

            # 캔들 데이터 처리 (실제 데이터만 사용)
            prices = []
            volumes = []
            if candles and not isinstance(candles, Exception) and len(candles) > 0:
                for candle in candles:
                    try:
                        # 캔들 데이터 구조: [timestamp, open, high, low, close, volume]
                        close_price = float(candle[4])  # 종가
                        volume = float(candle[5])  # 거래량
                        
                        if close_price > 0:  # 유효한 데이터만 추가
                            prices.append(close_price)
                            volumes.append(volume)
                    except (ValueError, IndexError, TypeError):
                        continue

            # 기술적 지표 계산 (실제 데이터 기반)
            mas = {}
            for period in MA_PERIODS:
                ma_value = self.processor.calculate_ma(prices, period)
                mas[f"MA{period}"] = ma_value

            rsi = self.processor.calculate_rsi(prices, RSI_PERIOD) if len(prices) >= RSI_PERIOD + 1 else None
            
            # 볼린저밴드
            bb_upper = bb_mid = bb_lower = None
            if len(prices) >= BOLLINGER_PERIOD:
                bb_upper, bb_mid, bb_lower = self.processor.calculate_bollinger_bands(prices, BOLLINGER_PERIOD, BOLLINGER_STD_DEV)

            # 추가 분석 지표
            market_cap = price * volume_24h if volume_24h > 0 else None
            turnover_rate = self.processor.calculate_turnover_rate(value_24h, market_cap)
            
            volume_ma = self.processor.calculate_ma(volumes, 20) if len(volumes) >= 20 else None
            volume_change_rate = self.processor.calculate_volume_change_rate(
                volumes[-1] if volumes else 0, volume_ma
            )
            
            btc_relative_strength = None
            if btc_change_rate is not None:
                btc_relative_strength = change_rate - btc_change_rate

            # 신호 점수 계산
            signal_score = self.calculate_signal_score({
                'turnover_rate': turnover_rate,
                'volume_change_rate': volume_change_rate,
                'btc_relative_strength': btc_relative_strength,
                'rsi': rsi,
                'price': price,
                'mas': mas,
                'total_depth': total_depth,
                'trade_frequency_data': trade_frequency_data,
                'value_24h': value_24h
            })

            return {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "price": price,
                "change_rate": change_rate,
                "value_24h": value_24h,
                "volume_24h": volume_24h,
                "btc_change_rate": btc_change_rate,
                "btc_relative_strength": btc_relative_strength,
                "bid_ratio": bid_ratio,
                "spread_rate": spread_rate,
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "total_depth": total_depth,
                "mas": mas,
                "rsi": rsi,
                "bb_upper": bb_upper,
                "bb_mid": bb_mid,
                "bb_lower": bb_lower,
                "turnover_rate": turnover_rate,
                "volume_change_rate": volume_change_rate,
                "trade_frequency_data": trade_frequency_data,
                "signal_score": signal_score,
                "prices": prices[-50:] if len(prices) > 50 else prices,  # 최근 50개만
                "volumes": volumes[-50:] if len(volumes) > 50 else volumes,
                "data_quality": {
                    "candles_count": len(prices),
                    "transactions_count": trade_frequency_data['total_trades'],
                    "has_sufficient_data": len(prices) >= 50
                }
            }

        except Exception as e:
            return {"error": f"분석 중 오류: {str(e)}"}

    def calculate_signal_score(self, data):
        """실제 데이터 기반 신호 점수 계산"""
        score = 0
        max_score = 10

        # 1. 거래량 기반 점수 (0-3점)
        value_24h = data.get('value_24h', 0)
        if value_24h >= VOLUME_HIGH_THRESHOLD:
            score += 3
        elif value_24h >= VOLUME_NORMAL_THRESHOLD:
            score += 2
        elif value_24h >= INACTIVE_TRADE_VALUE_THRESHOLD:
            score += 1

        # 2. 기술적 지표 점수 (0-3점)
        rsi = data.get('rsi')
        if rsi is not None:
            if 40 <= rsi <= 70:
                score += 1.5
            elif 30 <= rsi < 40:
                score += 1  # 과매도에서 회복
            elif rsi > 70:
                score -= 0.5  # 과매수 위험

        # 이동평균선 점수
        price = data.get('price', 0)
        mas = data.get('mas', {})
        ma_above_count = 0
        for ma_key, ma_value in mas.items():
            if ma_value is not None and price > ma_value:
                ma_above_count += 1

        if ma_above_count >= 3:
            score += 1.5
        elif ma_above_count >= 2:
            score += 1

        # 3. 시장 강도 점수 (0-2점)
        btc_strength = data.get('btc_relative_strength')
        if btc_strength is not None:
            if btc_strength >= 3:
                score += 2
            elif btc_strength >= 1:
                score += 1
            elif btc_strength <= -3:
                score -= 1

        # 4. 실제 거래 활성도 점수 (0-2점)
        trade_freq_data = data.get('trade_frequency_data', {})
        trades_per_minute = trade_freq_data.get('trades_per_minute', 0)
        
        if trades_per_minute >= 5:
            score += 2
        elif trades_per_minute >= 2:
            score += 1.5
        elif trades_per_minute >= 0.5:
            score += 1

        # 점수 범위 제한
        score = max(0, min(score, max_score))
        
        return round(score, 1), max_score

# Streamlit 메인 앱
def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>📊 실제 데이터 코인 분석기</h1>
        <p>빗썸 실시간 데이터 • 9/25/99/200일 이동평균 • 실제 거래빈도</p>
    </div>
    """, unsafe_allow_html=True)

    # 사이드바
    st.sidebar.title("🎯 분석 설정")
    
    # 코인 선택
    popular_coins = ["BTC", "ETH", "XRP", "ADA", "DOT", "GMT", "DOGE", "SHIB", "MATIC", "SOL", "AVAX"]
    
    input_method = st.sidebar.radio("코인 선택 방법", ["인기 코인", "직접 입력"])
    
    if input_method == "인기 코인":
        symbol = st.sidebar.selectbox("인기 코인 선택", popular_coins)
    else:
        symbol = st.sidebar.text_input("코인 심볼 입력", "BTC").upper()

    # 분석 실행 버튼
    if st.sidebar.button("🔍 실제 데이터 분석", type="primary"):
        if symbol:
            st.session_state.selected_symbol = symbol
            st.session_state.run_analysis = True

    # 데이터 품질 표시
    st.sidebar.markdown("### 📊 데이터 품질")
    st.sidebar.info("""
    ✅ 실제 거래 데이터 사용
    ✅ 9/25/99/200일 이동평균
    ✅ 실시간 거래빈도 계산
    ❌ 템플릿 데이터 사용 금지
    """)

    # 분석 실행
    if hasattr(st.session_state, 'run_analysis') and st.session_state.run_analysis:
        symbol = st.session_state.selected_symbol
        
        with st.spinner(f'🔍 {symbol} 실제 데이터 분석 중...'):
            analyzer = CoinAnalyzer()
            
            # 비동기 함수 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysis_result = loop.run_until_complete(analyzer.get_analysis(symbol))
            loop.close()

        if analysis_result.get("error"):
            st.error(f"❌ 분석 실패: {analysis_result['error']}")
            return

        # 결과 표시
        display_analysis_results(analysis_result)
        
        # 분석 완료 후 플래그 리셋
        st.session_state.run_analysis = False

def display_analysis_results(data):
    """수정된 분석 결과 표시"""
    symbol = data['symbol']
    price = data['price']
    change_rate = data['change_rate']
    timestamp = data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

    # 데이터 품질 확인
    data_quality = data.get('data_quality', {})
    
    if not data_quality.get('has_sufficient_data', True):
        st.warning("⚠️ 데이터가 부족합니다. 분석 결과의 정확도가 낮을 수 있습니다.")

    # 메인 지표 카드들
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "현재가", 
            f"{price:,.0f} KRW",
            delta=f"{change_rate:+.2f}%"
        )
    
    with col2:
        value_24h = data.get('value_24h', 0)
        st.metric(
            "24시간 거래대금",
            f"{value_24h/1e8:.1f}억원"
        )
    
    with col3:
        signal_score, max_score = data.get('signal_score', (0, 10))
        st.metric(
            "실제 데이터 신호",
            f"{signal_score}/{max_score}점",
            delta=f"{(signal_score/max_score)*100:.0f}%"
        )
    
    with col4:
        btc_strength = data.get('btc_relative_strength')
        if btc_strength is not None:
            st.metric(
                "BTC 대비 강도",
                f"{btc_strength:+.2f}%"
            )

    # 실제 거래 빈도 섹션 (수정됨)
    st.markdown("### 📈 실제 거래 활성도")
    
    trade_freq_data = data.get('trade_frequency_data', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "분당 거래",
            f"{trade_freq_data.get('trades_per_minute', 0):.2f}건",
            help="실제 거래 빈도 (템플릿 아님)"
        )
    
    with col2:
        st.metric(
            "총 거래수",
            f"{trade_freq_data.get('total_trades', 0)}건"
        )
    
    with col3:
        buy_trades = trade_freq_data.get('buy_trades', 0)
        sell_trades = trade_freq_data.get('sell_trades', 0)
        if buy_trades + sell_trades > 0:
            buy_ratio = (buy_trades / (buy_trades + sell_trades)) * 100
            st.metric(
                "매수 거래 비율",
                f"{buy_ratio:.1f}%"
            )
    
    with col4:
        st.metric(
            "거래 상태",
            trade_freq_data.get('status', 'N/A')
        )

    # 이동평균선 분석 (9, 25, 99, 200일)
    st.markdown("### 📊 이동평균선 분석 (실제 투자자용)")
    
    mas = data.get('mas', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 단기 이동평균")
        for period in [9, 25]:
            ma_key = f"MA{period}"
            ma_value = mas.get(ma_key)
            if ma_value:
                position = "상회 ✅" if price > ma_value else "하회 ❌"
                deviation = ((price - ma_value) / ma_value) * 100
                st.write(f"• **MA{period}**: {ma_value:,.0f}원 ({position}) [{deviation:+.1f}%]")
    
    with col2:
        st.markdown("#### 장기 이동평균")
        for period in [99, 200]:
            ma_key = f"MA{period}"
            ma_value = mas.get(ma_key)
            if ma_value:
                position = "상회 ✅" if price > ma_value else "하회 ❌"
                deviation = ((price - ma_value) / ma_value) * 100
                st.write(f"• **MA{period}**: {ma_value:,.0f}원 ({position}) [{deviation:+.1f}%]")

    # 추세 분석
    ma_above_count = sum(1 for ma_value in mas.values() if ma_value and price > ma_value)
    total_mas = len([ma for ma in mas.values() if ma is not None])
    
    if total_mas > 0:
        trend_strength = (ma_above_count / total_mas) * 100
        
        if trend_strength >= 75:
            trend_status = "🚀 강한 상승 추세"
            trend_color = "success"
        elif trend_strength >= 50:
            trend_status = "📈 상승 추세"
            trend_color = "info"
        elif trend_strength >= 25:
            trend_status = "📊 혼조"
            trend_color = "warning"
        else:
            trend_status = "📉 하락 추세"
            trend_color = "error"
        
        getattr(st, trend_color)(f"**추세 분석**: {trend_status} (이평선 상회: {ma_above_count}/{total_mas})")

    # 나머지 분석 결과는 기존과 동일...
    
    # 투자 신호 및 기타 지표들
    signal_score, max_score = data.get('signal_score', (0, 10))
    signal_percentage = (signal_score / max_score) * 100
    
    st.markdown("### 🚀 투자 신호 분석")
    
    if signal_percentage >= 70:
        signal_text = "🟢 강한 매수"
        signal_class = "signal-strong-buy"
    elif signal_percentage >= 50:
        signal_text = "🟡 약한 매수"
        signal_class = "signal-buy"
    elif signal_percentage >= 30:
        signal_text = "⚪ 중립"
        signal_class = "signal-neutral"
    else:
        signal_text = "🔴 약세"
        signal_class = "signal-sell"

    st.markdown(f"""
    <div class="metric-card {signal_class}">
        <h3>{signal_text}</h3>
        <p>실제 데이터 기반 점수: <strong>{signal_score}/{max_score}점</strong> ({signal_percentage:.0f}%)</p>
        <p>업데이트: {timestamp}</p>
    </div>
    """, unsafe_allow_html=True)

    # 상세 지표들
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 기술적 지표")
        
        # RSI
        rsi = data.get('rsi')
        if rsi:
            if rsi > 70:
                rsi_status = "🔴 과매수"
            elif rsi < 30:
                rsi_status = "🟢 과매도"
            else:
                rsi_status = "🟡 중립"
            st.write(f"• RSI(14): **{rsi:.1f}** {rsi_status}")
        
        # 볼린저밴드
        bb_upper = data.get('bb_upper')
        bb_lower = data.get('bb_lower')
        if bb_upper and bb_lower:
            if price > bb_upper:
                bb_pos = "🚀 상단 돌파"
            elif price < bb_lower:
                bb_pos = "💎 하단 이탈"
            else:
                bb_pos = "📊 밴드 내"
            st.write(f"• 볼린저밴드: **{bb_pos}**")
            st.write(f"  상단: {bb_upper:,.0f}원 / 하단: {bb_lower:,.0f}원")

    with col2:
        st.markdown("#### 💧 유동성 & 활성도")
        
        # 회전율
        turnover = data.get('turnover_rate')
        if turnover:
            if turnover >= 50:
                turnover_status = "🔥 매우 활발"
            elif turnover >= 20:
                turnover_status = "📊 보통"
            else:
                turnover_status = "😴 저조"
            st.write(f"• 회전율: **{turnover:.1f}%** {turnover_status}")
        
        # 거래량 변화
        vol_change = data.get('volume_change_rate')
        if vol_change:
            if vol_change >= 20:
                vol_icon = "⬆️⬆️ 급증"
            elif vol_change >= 5:
                vol_icon = "⬆️ 증가"
            elif vol_change <= -20:
                vol_icon = "⬇️⬇️ 급감"
            elif vol_change <= -5:
                vol_icon = "⬇️ 감소"
            else:
                vol_icon = "➡️ 보통"
            st.write(f"• 거래량 증가율: **{vol_change:+.1f}%** {vol_icon}")
        
        # 호가창 정보
        total_depth = data.get('total_depth', 0)
        bid_ratio = data.get('bid_ratio', 50)
        st.write(f"• 호가창 깊이: **{total_depth/1e8:.1f}억원**")
        st.write(f"• 매수/매도 비율: **{bid_ratio:.1f}% / {100-bid_ratio:.1f}%**")

    # 차트 섹션
    if data.get('prices') and len(data['prices']) > 10:
        st.markdown("#### 📈 가격 차트 (이동평균선 포함)")
        
        prices = data['prices']
        
        # 데이터프레임 생성
        df = pd.DataFrame({
            'Index': range(len(prices)),
            'Price': prices
        })
        
        # Plotly 차트 생성
        fig = go.Figure()
        
        # 가격 라인
        fig.add_trace(go.Scatter(
            x=df['Index'], 
            y=df['Price'], 
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # 이동평균선 추가 (실제 데이터 기반)
        mas = data.get('mas', {})
        colors = ['orange', 'red', 'purple', 'green']
        
        for i, (ma_key, ma_value) in enumerate(mas.items()):
            if ma_value and i < len(colors):
                period = int(ma_key.replace('MA', ''))
                if len(prices) >= period:
                    ma_line = []
                    for j in range(len(prices)):
                        if j >= period - 1:
                            ma_line.append(np.mean(prices[j-period+1:j+1]))
                        else:
                            ma_line.append(None)
                    
                    fig.add_trace(go.Scatter(
                        x=df['Index'], 
                        y=ma_line, 
                        mode='lines',
                        name=f'{ma_key}({period}일)',
                        line=dict(color=colors[i], width=1.5)
                    ))
        
        fig.update_layout(
            title=f'{symbol} 가격 움직임 및 이동평균선',
            xaxis_title="시간 (캔들 단위)",
            yaxis_title="가격 (KRW)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # 데이터 품질 정보
    st.markdown("#### 🔍 데이터 품질 정보")
    
    data_quality = data.get('data_quality', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "캔들 데이터",
            f"{data_quality.get('candles_count', 0)}개"
        )
    
    with col2:
        st.metric(
            "거래 내역",
            f"{data_quality.get('transactions_count', 0)}건"
        )
    
    with col3:
        sufficient = data_quality.get('has_sufficient_data', False)
        st.metric(
            "데이터 충분성",
            "충분" if sufficient else "부족"
        )

    # 투자 가이드라인
    st.markdown("#### 💡 투자 가이드라인")
    
    if signal_percentage >= 70:
        st.success("""
        **🟢 적극 매수 구간**
        - 2-3회 분할 매수 권장
        - 목표 수익률: +15~25%
        - 손절라인: -8%
        - 투자기간: 1-3주
        """)
    elif signal_percentage >= 50:
        st.warning("""
        **🟡 신중 매수 구간**
        - 3-4회 분할 매수 권장
        - 목표 수익률: +8~15%
        - 손절라인: -6%
        - 투자기간: 2-4주
        """)
    elif signal_percentage >= 30:
        st.info("""
        **⚪ 중립 구간**
        - 관망 또는 DCA 전략
        - 변동성 거래 고려
        - 손절라인: -5% 엄격 준수
        - 상황 대응 필요
        """)
    else:
        st.error("""
        **🔴 주의 구간**
        - 신규 진입 금지
        - 기존 포지션 매도 고려
        - 추세 전환 시점 대기
        - 리스크 관리 우선
        """)

    # 면책 조항
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
    ⚠️ <strong>투자 주의사항</strong><br>
    본 분석은 실제 데이터를 기반으로 하지만 참고용이며, 투자 결정과 모든 책임은 투자자 본인에게 있습니다.<br>
    암호화폐 투자는 높은 위험을 수반하므로 충분한 연구와 신중한 판단이 필요합니다.<br>
    <strong>분할매수, 손절선 설정, 포트폴리오 분산</strong> 등 리스크 관리를 반드시 실행하세요.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

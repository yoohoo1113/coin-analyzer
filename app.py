# -*- coding: utf-8 -*-
"""
Streamlit 웹 앱 버전 코인 분석기
- 기존 analyze_coin.py를 웹 인터페이스로 변환
- 아이폰 브라우저에서 접근 가능
"""

import streamlit as st
import asyncio
import aiohttp
import numpy as np
from collections import deque
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="🪙 실시간 코인 분석기",
    page_icon="🪙",
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

# 기존 클래스들을 그대로 사용 (동일한 분석 로직)
# --- Configuration ---
CANDLE_HISTORY_SIZE = 100
MA_PERIODS = [5, 20]
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
TRADE_FREQ_HIGH_THRESHOLD = 20.0
TRADE_FREQ_LOW_THRESHOLD = 5.0
DEPTH_LARGE_THRESHOLD = 10_000_000_000
DEPTH_SMALL_THRESHOLD = 1_000_000_000

# 신호 분석 기준값
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
BTC_STRONG_OUTPERFORM = 3.0
BTC_STRONG_UNDERPERFORM = -3.0

# 기존 클래스들 (streamlit용으로 약간 수정)
class BithumbApi:
    BASE_URL = "https://api.bithumb.com/public"

    def __init__(self):
        self.session = None

    async def create_session(self):
        timeout = aiohttp.ClientTimeout(total=10)
        self.session = aiohttp.ClientSession(timeout=timeout)

    async def close_session(self):
        if self.session:
            await self.session.close()

    async def fetch(self, endpoint: str) -> dict | None:
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    st.error(f"API Error: Status {resp.status} for {url}")
                    return None
                data = await resp.json()
                if data.get("status") != "0000":
                    data.pop('date', None)
                    st.error(f"API Error: {data.get('message')}")
                    return None
                return data.get("data")
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

    async def get_candlestick(self, symbol: str, chart_intervals: str = "1m"):
        return await self.fetch(f"candlestick/{symbol.upper()}_KRW/{chart_intervals}")

# 다른 클래스들은 기존과 동일하게 유지...
# (DataProcessor, SignalAnalyzer, CoinAnalyzer 클래스들)

class DataProcessor:
    @staticmethod
    def calculate_ma(prices: list, period: int) -> float | None:
        if len(prices) < period: return None
        return np.mean(prices[-period:])

    @staticmethod
    def calculate_bollinger_bands(prices: list, period: int, std_dev: int):
        if len(prices) < period: return None, None, None
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        return sma + (std * std_dev), sma, sma - (std * std_dev)

    @staticmethod
    def calculate_rsi(prices: list, period: int) -> float | None:
        if len(prices) < period + 1: return None
        deltas = np.diff(prices)
        gains = deltas[deltas >= 0]
        losses = -deltas[deltas < 0]
        avg_gain = np.mean(gains[-period:]) if len(gains) > 0 else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) > 0 else 1
        if avg_loss == 0: return 100.0
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
    def calculate_orderbook_depth(orderbook: dict):
        try:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            total_bid_amount = sum(float(bid["price"]) * float(bid["quantity"]) for bid in bids)
            total_ask_amount = sum(float(ask["price"]) * float(ask["quantity"]) for ask in asks)
            total_depth = total_bid_amount + total_ask_amount
            return total_bid_amount, total_ask_amount, total_depth
        except (KeyError, ValueError, TypeError):
            return None, None, None

    @staticmethod
    def calculate_trade_frequency(transactions: list) -> float | None:
        if not transactions or len(transactions) == 0:
            return None
        try:
            time_range_minutes = 30
            return len(transactions) / time_range_minutes
        except Exception:
            return None

    @staticmethod
    def calculate_btc_relative_strength(coin_change_rate: float, btc_change_rate: float) -> float | None:
        if btc_change_rate is not None:
            return coin_change_rate - btc_change_rate
        return None

class StreamlitCoinAnalyzer:
    """Streamlit용 코인 분석기"""
    
    def __init__(self):
        self.api = None
        self.processor = DataProcessor()

    async def get_analysis(self, symbol: str):
        """분석 실행 및 결과 반환"""
        if not self.api:
            self.api = BithumbApi()
            await self.api.create_session()

        try:
            # 기본 데이터 수집
            results = await asyncio.gather(
                self.api.get_ticker(symbol),
                self.api.get_orderbook(symbol),
                self.api.get_all_tickers(),
                self.api.get_transaction_history(symbol),
                self.api.get_candlestick(symbol, "1m"),
                return_exceptions=True
            )
            
            ticker, orderbook, all_tickers, transactions, candles = results

            if not ticker or not orderbook:
                return {"error": "필수 데이터 수신 실패"}

            # 기본 정보 추출
            price = float(ticker["closing_price"])
            change_rate = float(ticker["fluctate_rate_24H"])
            value_24h = float(ticker["acc_trade_value_24H"])
            volume_24h = float(ticker.get("units_traded_24H", 0))

            # BTC 정보 추출
            btc_change_rate = None
            if all_tickers and not isinstance(all_tickers, Exception):
                all_tickers.pop('date', None)
                if 'BTC' in all_tickers:
                    btc_change_rate = float(all_tickers['BTC'].get('fluctate_rate_24H', 0))

            # 호가창 분석
            try:
                bids, asks = orderbook["bids"], orderbook["asks"]
                bid_depth = sum(float(b["price"]) * float(b["quantity"]) for b in bids)
                ask_depth = sum(float(a["price"]) * float(a["quantity"]) for a in asks)
                total_depth = bid_depth + ask_depth
                
                total_bid_qty = sum(float(b["quantity"]) for b in bids)
                total_ask_qty = sum(float(a["quantity"]) for a in asks)
                total_qty = total_bid_qty + total_ask_qty
                bid_ratio = (total_bid_qty / total_qty) * 100 if total_qty else 50
                
                spread = float(asks[0]["price"]) - float(bids[0]["price"])
                spread_rate = (spread / price) * 100
            except Exception:
                bid_depth = ask_depth = total_depth = 0
                bid_ratio = 50
                spread = spread_rate = 0

            # 캔들 데이터 처리
            prices = []
            volumes = []
            if candles and not isinstance(candles, Exception):
                for candle in candles[-50:]:  # 최근 50개
                    prices.append(float(candle[2]))  # 종가
                    volumes.append(float(candle[5]))  # 거래량

            # 기술적 지표 계산
            ma5 = self.processor.calculate_ma(prices, 5) if len(prices) >= 5 else None
            ma20 = self.processor.calculate_ma(prices, 20) if len(prices) >= 20 else None
            rsi = self.processor.calculate_rsi(prices, 14) if len(prices) >= 15 else None
            
            # 볼린저밴드
            bb_upper = bb_mid = bb_lower = None
            if len(prices) >= 20:
                bb_upper, bb_mid, bb_lower = self.processor.calculate_bollinger_bands(prices, 20, 2)

            # 추가 지표들
            market_cap = price * volume_24h if volume_24h > 0 else None
            turnover_rate = self.processor.calculate_turnover_rate(value_24h, market_cap)
            
            volume_ma = self.processor.calculate_ma(volumes, 20) if len(volumes) >= 20 else None
            volume_change_rate = self.processor.calculate_volume_change_rate(
                volumes[-1] if volumes else 0, volume_ma
            )
            
            btc_relative_strength = self.processor.calculate_btc_relative_strength(change_rate, btc_change_rate)
            trade_frequency = self.processor.calculate_trade_frequency(transactions if not isinstance(transactions, Exception) else None)

            # 신호 분석
            signal_score = self.calculate_signal_score({
                'turnover_rate': turnover_rate,
                'volume_change_rate': volume_change_rate,
                'btc_relative_strength': btc_relative_strength,
                'rsi': rsi,
                'price': price,
                'ma5': ma5,
                'ma20': ma20,
                'total_depth': total_depth,
                'trade_frequency': trade_frequency
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
                "ma5": ma5,
                "ma20": ma20,
                "rsi": rsi,
                "bb_upper": bb_upper,
                "bb_mid": bb_mid,
                "bb_lower": bb_lower,
                "turnover_rate": turnover_rate,
                "volume_change_rate": volume_change_rate,
                "trade_frequency": trade_frequency,
                "signal_score": signal_score,
                "prices": prices,
                "volumes": volumes
            }

        except Exception as e:
            return {"error": f"분석 중 오류: {str(e)}"}

    def calculate_signal_score(self, data):
        """간단한 신호 점수 계산"""
        score = 0
        max_score = 10

        # 회전율 (0-2점)
        turnover = data.get('turnover_rate')
        if turnover and turnover >= 50:
            score += 2
        elif turnover and turnover >= 20:
            score += 1

        # 거래량 증가 (0-2점)
        vol_change = data.get('volume_change_rate')
        if vol_change and vol_change >= 20:
            score += 2
        elif vol_change and vol_change >= 5:
            score += 1

        # BTC 대비 강도 (0-2점)
        btc_strength = data.get('btc_relative_strength')
        if btc_strength and btc_strength >= 3:
            score += 2
        elif btc_strength and btc_strength >= 1:
            score += 1

        # RSI 및 이동평균 (0-2점)
        rsi = data.get('rsi')
        price = data.get('price', 0)
        ma5 = data.get('ma5')
        ma20 = data.get('ma20')
        
        if rsi and 40 <= rsi <= 70:
            score += 1
        if ma5 and ma20 and price > ma5 > ma20:
            score += 1

        # 유동성 (0-2점)
        total_depth = data.get('total_depth')
        trade_freq = data.get('trade_frequency')
        
        if total_depth and total_depth >= 1_000_000_000:
            score += 1
        if trade_freq and trade_freq >= 10:
            score += 1

        return score, max_score

# Streamlit 메인 앱
def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🪙 실시간 코인 분석기</h1>
        <p>빗썸 거래소 실시간 데이터 기반 종합 분석</p>
    </div>
    """, unsafe_allow_html=True)

    # 사이드바
    st.sidebar.title("🎯 분석 설정")
    
    # 코인 선택
    popular_coins = ["BTC", "ETH", "XRP", "ADA", "DOT", "GMT", "DOGE", "SHIB", "MATIC", "SOL"]
    
    input_method = st.sidebar.radio("코인 선택 방법", ["인기 코인", "직접 입력"])
    
    if input_method == "인기 코인":
        symbol = st.sidebar.selectbox("인기 코인 선택", popular_coins)
    else:
        symbol = st.sidebar.text_input("코인 심볼 입력", "BTC").upper()

    # 분석 실행 버튼
    if st.sidebar.button("🔍 분석 실행", type="primary"):
        if symbol:
            st.session_state.selected_symbol = symbol
            st.session_state.run_analysis = True

    # 자동 새로고침 옵션
    auto_refresh = st.sidebar.checkbox("🔄 30초 자동 새로고침")
    if auto_refresh:
        st.rerun()

    # 분석 실행
    if hasattr(st.session_state, 'run_analysis') and st.session_state.run_analysis:
        symbol = st.session_state.selected_symbol
        
        with st.spinner(f'🔍 {symbol} 분석 중...'):
            analyzer = StreamlitCoinAnalyzer()
            
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
    """분석 결과 표시"""
    symbol = data['symbol']
    price = data['price']
    change_rate = data['change_rate']
    timestamp = data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

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
            "투자 신호 점수",
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

    # 투자 신호 섹션
    st.markdown("### 🚀 투자 신호 분석")
    
    signal_score, max_score = data.get('signal_score', (0, 10))
    signal_percentage = (signal_score / max_score) * 100
    
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
        <p>종합 점수: <strong>{signal_score}/{max_score}점</strong> ({signal_percentage:.0f}%)</p>
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
            rsi_color = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
            st.write(f"• RSI(14): **{rsi:.1f}** {rsi_color}")
        
        # 이동평균
        ma5 = data.get('ma5')
        ma20 = data.get('ma20')
        if ma5 and ma20:
            ma_trend = "📈 상승" if price > ma5 > ma20 else "📉 하락" if price < ma5 < ma20 else "📊 혼조"
            st.write(f"• MA(5): **{ma5:,.0f}원**")
            st.write(f"• MA(20): **{ma20:,.0f}원**")
            st.write(f"• 추세: **{ma_trend}**")

        # 볼린저밴드
        bb_upper = data.get('bb_upper')
        bb_lower = data.get('bb_lower')
        if bb_upper and bb_lower:
            if price > bb_upper:
                bb_pos = "🚀 상단 돌파"
            elif price < bb_lower:
                bb_pos = "📉 하단 이탈"
            else:
                bb_pos = "📊 밴드 내"
            st.write(f"• 볼린저밴드: **{bb_pos}**")
            st.write(f"  상단: {bb_upper:,.0f}원 / 하단: {bb_lower:,.0f}원")

    with col2:
        st.markdown("#### 💧 유동성 & 활성도")
        
        # 회전율
        turnover = data.get('turnover_rate')
        if turnover:
            turnover_status = "🔥 매우 활발" if turnover >= 50 else "📊 보통" if turnover >= 20 else "😴 저조"
            st.write(f"• 회전율: **{turnover:.1f}%** {turnover_status}")
        
        # 거래량 변화
        vol_change = data.get('volume_change_rate')
        if vol_change:
            vol_icon = "⬆️⬆️" if vol_change >= 20 else "⬆️" if vol_change >= 5 else "⬇️" if vol_change <= -5 else "➡️"
            st.write(f"• 거래량 증가율: **{vol_change:+.1f}%** {vol_icon}")
        
        # 호가창 정보
        total_depth = data.get('total_depth', 0)
        bid_ratio = data.get('bid_ratio', 50)
        st.write(f"• 호가창 깊이: **{total_depth/1e8:.1f}억원**")
        st.write(f"• 매수/매도 비율: **{bid_ratio:.1f}% / {100-bid_ratio:.1f}%**")
        
        # 거래 빈도
        trade_freq = data.get('trade_frequency')
        if trade_freq:
            freq_status = "⚡⚡" if trade_freq >= 20 else "⚡" if trade_freq >= 10 else "💤"
            st.write(f"• 거래 빈도: **{trade_freq:.1f}건/분** {freq_status}")

    # 차트 섹션
    if data.get('prices') and len(data['prices']) > 10:
        st.markdown("#### 📈 가격 차트")
        
        prices = data['prices']
        
        # 간단한 라인 차트
        df = pd.DataFrame({
            'Index': range(len(prices)),
            'Price': prices
        })
        
        fig = px.line(df, x='Index', y='Price', title=f'{symbol} 최근 가격 움직임')
        fig.update_layout(
            xaxis_title="시간 (분 단위)",
            yaxis_title="가격 (KRW)",
            height=400
        )
        
        # 이동평균선 추가
        if len(prices) >= 20:
            ma5_line = [np.mean(prices[max(0, i-4):i+1]) for i in range(len(prices))]
            ma20_line = [np.mean(prices[max(0, i-19):i+1]) for i in range(len(prices))]
            
            fig.add_scatter(x=df['Index'], y=ma5_line, name='MA(5)', line=dict(color='orange'))
            fig.add_scatter(x=df['Index'], y=ma20_line, name='MA(20)', line=dict(color='red'))
        
        st.plotly_chart(fig, use_container_width=True)

    # 추천 액션
    st.markdown("#### 💡 투자 가이드라인")
    
    if signal_percentage >= 70:
        st.success("""
        **🟢 적극 매수 구간**
        - 2-3회 분할 매수 권장
        - 목표 수익률: +15~25%
        - 손절라인: -8%
        """)
    elif signal_percentage >= 50:
        st.warning("""
        **🟡 신중 매수 구간**
        - 3-4회 분할 매수 권장
        - 목표 수익률: +8~15%
        - 손절라인: -6%
        """)
    elif signal_percentage >= 30:
        st.info("""
        **⚪ 중립 구간**
        - 관망 또는 DCA 전략
        - 변동성 거래 고려
        - 손절라인: -5% 엄격 준수
        """)
    else:
        st.error("""
        **🔴 주의 구간**
        - 신규 진입 금지
        - 기존 포지션 매도 고려
        - 추세 전환 시점 대기
        """)

    # 면책 조항
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
    ⚠️ <strong>면책조항</strong><br>
    본 분석은 참고용이며, 투자 결정에 대한 모든 책임은 투자자 본인에게 있습니다.<br>
    암호화폐 투자는 높은 위험을 수반하므로 신중한 판단이 필요합니다.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from datetime import date, timedelta

st.set_page_config(
    page_title="PSX AI Stock Assistant",
    page_icon="📈",
    layout="wide"
)

# Consistent, clean chart styling across all matplotlib plots
plt.style.use('seaborn-v0_8-whitegrid')

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.title("📊 PSX AI Assistant")
st.sidebar.markdown("---")
st.sidebar.markdown("### About This App")
st.sidebar.markdown("""
This app uses **Linear Regression** to predict 
Pakistan Stock Exchange (PSX) stock prices.

**Features used:**
- Close Price
- MA5 (5-day Moving Average)
- MA10 (10-day Moving Average)  
- MA20 (20-day Moving Average)
- Daily Change %

**Popular Tickers:**
- OGDC.KA
- HBL.KA
- LUCK.KA
- ENGRO.KA
- PSO.KA
- UBL.KA
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### Project Info")
st.sidebar.markdown("""
- **Course:** AI Lab
- **Topic:** Supervised Learning - Regression
- **Model:** Linear Regression
""")

tab1, tab2 = st.tabs(["📈 Stock Price Predictor", "🤖 AI Assistant"])

# ═════════════════════════════════════════
# HELPER — calculate future trading dates (skips weekends)
# ═════════════════════════════════════════
def get_future_trading_dates(from_date, n):
    """Returns a list of n trading dates (Mon-Fri) after from_date."""
    future = []
    check = pd.Timestamp(from_date)
    while len(future) < n:
        check += timedelta(days=1)
        if check.weekday() < 5:   # 0=Mon ... 4=Fri, skip Sat/Sun
            future.append(check)
    return future

# ═════════════════════════════════════════
# TAB 1 — STOCK PREDICTOR
# ═════════════════════════════════════════
with tab1:
    st.title("📈 Pakistan Stock Exchange (PSX) Price Predictor")
    st.markdown("Enter a PSX ticker with `.KA` suffix, e.g. **OGDC.KA**, **HBL.KA**, **LUCK.KA**, **ENGRO.KA**")

    col_a, col_b = st.columns(2)
    with col_a:
        ticker = st.text_input("Enter Stock Ticker:", "OGDC.KA")
    with col_b:
        days_ahead = st.slider("Predict how many days ahead?", min_value=1, max_value=10, value=3)

    if st.button("🔍 Predict", use_container_width=True):
        with st.spinner("Fetching latest data and training model..."):

            # ── FIX 1: end = tomorrow, since yfinance's end date is exclusive.
            # Using date.today() alone can silently miss today's bar if it exists.
            end_date = date.today() + timedelta(days=1)
            data = yf.download(
                ticker,
                start="2023-01-01",
                end=end_date.strftime("%Y-%m-%d"),
                progress=False
            )
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

        if data.empty:
            st.error("❌ No data found. Try OGDC.KA, HBL.KA, LUCK.KA, or ENGRO.KA")
        else:
            data = data[['Close']].copy()
            data['MA5']            = data['Close'].rolling(window=5).mean()
            data['MA10']           = data['Close'].rolling(window=10).mean()
            data['MA20']           = data['Close'].rolling(window=20).mean()
            data['Daily_Change_%'] = data['Close'].pct_change() * 100

            # ── FIX 2: Shift target FORWARD — no data leakage ──
            data['Target'] = data['Close'].shift(-days_ahead)

            # NEW: keep the true latest row (with known features but no Target yet)
            # separately, BEFORE dropping NaNs — this is what we'll actually predict from.
            latest_row_features = data[['Close', 'MA5', 'MA10', 'MA20', 'Daily_Change_%']].iloc[[-1]].copy()
            latest_row_date     = data.index[-1]

            data = data.dropna()

            # ── EDA ──
            st.subheader("📋 Exploratory Data Analysis (EDA)")
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.markdown(f"**Dataset Shape:** `{data.shape}`")
                st.markdown(f"**Missing Values:** `{int(data.isnull().sum().sum())}`")
                st.markdown(f"**Date Range:** {data.index[0].date()} to {data.index[-1].date()}")
                st.markdown(f"**Total Trading Days:** `{len(data)}`")
            with col_e2:
                st.markdown("**Statistical Summary:**")
                st.dataframe(
                    data[['Close','MA5','MA10','MA20','Daily_Change_%']].describe().round(2),
                    use_container_width=True
                )
            st.markdown("---")

            features = ['Close', 'MA5', 'MA10', 'MA20', 'Daily_Change_%']
            X = data[features]
            y = data['Target']

            # ── FIX 3: Time-based split — never shuffle time series ──
            split     = int(len(X) * 0.8)
            X_train   = X.iloc[:split]
            X_test    = X.iloc[split:]
            y_train   = y.iloc[:split]
            y_test    = y.iloc[split:]

            # ── FIX 4: Scale features — Close=320, Change%=0.8 need equal weight ──
            scaler      = StandardScaler()
            X_train_s   = scaler.fit_transform(X_train)
            X_test_s    = scaler.transform(X_test)

            model = LinearRegression()
            model.fit(X_train_s, y_train)
            predictions = model.predict(X_test_s)

            mae  = mean_absolute_error(y_test, predictions)
            mse  = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2   = r2_score(y_test, predictions)

            if r2 >= 0.90:
                perf = "🟢 Excellent"
            elif r2 >= 0.70:
                perf = "🟡 Good"
            else:
                perf = "🔴 Low Predictability"

            st.success(f"✅ Model trained successfully! Performance: {perf}")

            # ── Metrics ──
            st.subheader(f"📊 Results for {ticker.upper()} ({days_ahead}-day ahead prediction)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE",      f"{mae:.2f} PKR")
            m2.metric("MSE",      f"{mse:.2f}")
            m3.metric("RMSE",     f"{rmse:.2f} PKR")
            m4.metric("R² Score", f"{r2:.4f}")
            st.markdown("---")

            # ── Actual vs Predicted ──
            st.subheader("📊 Actual vs Predicted Price")
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(y_test.values, label='Actual Price',    color='blue',   linewidth=2)
            ax1.plot(predictions,   label='Predicted Price', color='orange', linewidth=2, linestyle='--')
            ax1.set_title(f"{ticker.upper()} — Actual vs Predicted ({days_ahead} days ahead)", fontsize=14)
            ax1.set_xlabel("Days")
            ax1.set_ylabel("Price (PKR)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            plt.close(fig1)

            # ── Residual Plot ──
            st.subheader("📉 Residual Plot")
            residuals = y_test.values.flatten() - predictions.flatten()
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.plot(residuals, color='red', label='Residuals', linewidth=1.5)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
            ax2.fill_between(range(len(residuals)), residuals, 0, alpha=0.2, color='red')
            ax2.set_title("Residual Plot — Actual minus Predicted", fontsize=14)
            ax2.set_xlabel("Days")
            ax2.set_ylabel("Residual (PKR)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close(fig2)

            # ── FIX 5: Feature importance after scaling — coefficients now comparable ──
            st.subheader("🔍 Feature Importance (Standardized Coefficients)")
            coefs  = pd.Series(model.coef_, index=features)
            colors = ['green' if c > 0 else 'red' for c in coefs.sort_values()]
            fig3, ax3 = plt.subplots(figsize=(9, 4))
            coefs.sort_values().plot(kind='barh', ax=ax3, color=colors)
            ax3.set_title("Feature Importance — Standardized Regression Coefficients\n(features scaled so coefficients are directly comparable)", fontsize=12)
            ax3.set_xlabel("Standardized Coefficient Value")
            ax3.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
            plt.close(fig3)

            # ── FIX 6: Learning curve with TimeSeriesSplit ──
            st.subheader("📈 Learning Curve")
            tscv = TimeSeriesSplit(n_splits=5)
            train_sizes, train_scores, test_scores = learning_curve(
                LinearRegression(), X, y,
                cv=tscv,
                scoring='r2',
                train_sizes=np.linspace(0.1, 1.0, 10)
            )
            fig4, ax4 = plt.subplots(figsize=(10, 4))
            ax4.plot(train_sizes, train_scores.mean(axis=1),
                     label='Training Score',   color='blue',  marker='o')
            ax4.plot(train_sizes, test_scores.mean(axis=1),
                     label='Validation Score', color='green', marker='s')
            ax4.fill_between(train_sizes,
                             train_scores.mean(axis=1) - train_scores.std(axis=1),
                             train_scores.mean(axis=1) + train_scores.std(axis=1),
                             alpha=0.1, color='blue')
            ax4.fill_between(train_sizes,
                             test_scores.mean(axis=1) - test_scores.std(axis=1),
                             test_scores.mean(axis=1) + test_scores.std(axis=1),
                             alpha=0.1, color='green')
            ax4.set_title("Learning Curve — Model Generalization (TimeSeriesSplit CV)", fontsize=14)
            ax4.set_xlabel("Training Size")
            ax4.set_ylabel("R² Score")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            st.pyplot(fig4)
            plt.close(fig4)

            # ── Recent Data ──
            st.subheader("🗂️ Recent Data (Training Set)")
            st.caption(
                f"Showing the last 10 rows used to train the model. The 'Target' column is the actual "
                f"Close price {days_ahead} trading day(s) later — these rows are older than the very "
                f"latest data point (shown in 'Next Prediction' below) because their future target is "
                f"already known."
            )
            st.dataframe(data.tail(10), use_container_width=True)

            # ── FIX 7 (UPDATED): Prediction with transparent dates + confidence range ──
            st.subheader("🔮 Next Prediction")

            # Use the TRUE latest available row (kept before dropna) for the prediction input,
            # not the last row of the training set — otherwise predicting further ahead
            # paradoxically uses older "current price" data.
            X_latest      = scaler.transform(latest_row_features)
            next_price    = model.predict(X_latest)[0]
            current_price = float(latest_row_features['Close'].iloc[0])
            change        = next_price - current_price
            pct           = (change / current_price) * 100
            low_range     = next_price - rmse
            high_range    = next_price + rmse

            # Dates reflect the TRUE latest data point, not the training set's last row.
            # target_date is computed using real trading days (skips weekends), not raw calendar days.
            last_data_date = latest_row_date.date()
            future_dates   = get_future_trading_dates(latest_row_date, days_ahead)
            target_date    = future_dates[-1]
            today_date     = date.today()
            data_lag_days  = (today_date - last_data_date).days

            if data_lag_days > 0:
                st.warning(
                    f"⚠️ **Data freshness notice:** The most recent PSX data available "
                    f"from this source is dated **{last_data_date}** "
                    f"({data_lag_days} day(s) behind today, {today_date}). "
                    f"This is a known limitation of free PSX data feeds (via Yahoo Finance), "
                    f"which can lag by several days — unlike US stock data, which updates same-day. "
                    f"Predicting for: **{target_date.strftime('%A, %B %d, %Y')}** "
                    f"({days_ahead} trading day(s) from the last available data)."
                )

            p1, p2, p3 = st.columns(3)
            p1.metric(f"Last Close ({last_data_date})", f"{current_price:.2f} PKR")
            p2.metric(f"Predicted ({target_date.strftime('%b %d, %Y')})", f"{next_price:.2f} PKR")
            p3.metric("Expected Change", f"{change:+.2f} PKR", f"{pct:+.2f}%")

            # FIX 8: Confidence interval + disclaimer
            st.info(
                f"📊 **Prediction Range (±1 RMSE):** "
                f"{low_range:.2f} PKR  →  {high_range:.2f} PKR  "
                f"| Target Date: **{target_date.strftime('%A, %B %d, %Y')}** "
                f"({days_ahead} trading day(s) from the last available data point)"
            )
            st.caption(
                "⚠️ For educational purposes only. "
                "This model uses Linear Regression on historical price data "
                "and does not constitute financial advice. "
                "Always do your own research before investing. "
                "Note: 'days ahead' refers to trading days from the last data point available, "
                "not necessarily today's date, due to data feed lag (see notice above)."
            )


# ═════════════════════════════════════════
# TAB 2 — AI CHATBOT
# ═════════════════════════════════════════
with tab2:
    st.title("🤖 AI Stock Assistant")
    st.markdown("Ask me anything about **stocks**, **PSX**, or **machine learning**!")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Hello! I'm your PSX AI Assistant. Ask me anything about stocks, PSX, or machine learning!"
        })

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            p = prompt.lower()

            if any(w in p for w in ["increase", "up", "rise", "buy", "much", "gain"]):
                answer = "📈 Go to the **Stock Predictor tab**, enter the ticker (e.g. HBL.KA), set days ahead, and click Predict. The **Next Prediction** box shows current price vs predicted price with expected % change!"

            elif any(w in p for w in ["decrease", "down", "fall", "sell", "drop", "loss"]):
                answer = "📉 Check the **Stock Predictor tab** — if Expected Change shows a negative %, the model predicts a price drop. Always check R² score for reliability."

            elif any(w in p for w in ["predict", "forecast", "future", "tomorrow", "next", "days"]):
                answer = "🔮 Go to **Stock Predictor tab**, enter ticker like OGDC.KA, set the days slider, click Predict. The **Next Prediction** section at the bottom shows the future price with a confidence range!"

            elif any(w in p for w in ["r2", "r²", "r square", "accurate", "accuracy", "reliable"]):
                answer = "📊 **R² Score** measures model accuracy:\n- 🟢 Above 0.90 = Excellent\n- 🟡 0.70–0.90 = Good\n- 🔴 Below 0.70 = Low predictability\n\nAfter proper feature scaling, expect R² around 0.55–0.75 — which is honest and realistic for stock prediction!"

            elif any(w in p for w in ["mse", "mean squared", "mae", "mean absolute", "rmse", "error", "loss"]):
                answer = "📏 **Error Metrics:**\n- **MAE** = average error in PKR (easy to understand)\n- **MSE** = squared error (penalizes large errors more)\n- **RMSE** = square root of MSE, same unit as price\n\nLower values = better model!"

            elif any(w in p for w in ["scale", "scaler", "standardscaler", "normalize"]):
                answer = "⚖️ **Feature Scaling** is applied in this app using StandardScaler. Close Price (~320 PKR) and Daily Change% (~0.8%) are on very different scales — without scaling, Close Price would dominate the model unfairly. Scaling makes all features equally weighted!"

            elif any(w in p for w in ["confidence", "range", "interval", "uncertainty"]):
                answer = "📊 The **Prediction Range** shown below the prediction is ±1 RMSE. This means the true price is likely to fall within that range. A tighter range = more reliable model!"

            elif any(w in p for w in ["lag", "delay", "outdated", "old data", "stale", "freshness"]):
                answer = "🕒 **Data Lag Notice:** PSX data via free sources like Yahoo Finance can lag several days behind today's date, unlike US stocks which update same-day. The app shows a warning with the exact last available date whenever this happens, so predictions are always dated transparently."

            elif any(w in p for w in ["cnn", "convolutional"]):
                answer = "🖼️ **CNN (Convolutional Neural Network)** is specialized for images. In this project CNN achieved **91.63% accuracy** on Fashion-MNIST, outperforming ANN by **3.76%** using convolutional layers to detect patterns!"

            elif any(w in p for w in ["ann", "neural network", "artificial"]):
                answer = "🧠 **ANN (Artificial Neural Network)** results:\n- Iris dataset: **100% accuracy**\n- Fashion-MNIST: **87.87% accuracy**\n\nANN treats images as flat arrays while CNN uses spatial structure."

            elif any(w in p for w in ["fashion", "mnist", "clothing", "image"]):
                answer = "👗 **Fashion-MNIST** has 70,000 images of 10 clothing categories:\n- **ANN:** 87.87% accuracy\n- **CNN:** 91.63% accuracy\n- CNN wins by **3.76%**!"

            elif any(w in p for w in ["iris", "flower", "setosa", "versicolor"]):
                answer = "🌸 **Iris dataset** has 150 samples of 3 flower species. Our ANN achieved **100% test accuracy** using 3 dense layers with ReLU activation!"

            elif any(w in p for w in ["knn", "k nearest", "k-nearest", "euclidean"]):
                answer = "📍 **KNN Results on Wine dataset:**\n- K=3 → Predicted Class 1\n- K=5 → Predicted Class 1\n- K=7 → Flipped to Class 0!\n\nShows why K selection matters — larger K considers wider neighborhood."

            elif any(w in p for w in ["logistic", "classification", "wine"]):
                answer = "📊 **Logistic Regression** on Wine dataset achieved **100% accuracy** after feature scaling with StandardScaler. Used for multiclass classification of 3 wine types!"

            elif any(w in p for w in ["linear regression", "regression model", "how does"]):
                answer = "📐 **Linear Regression** finds the best line through data points. This app uses 5 features:\n- Close Price\n- MA5, MA10, MA20 (Moving Averages)\n- Daily Change %\n\nAll features are scaled with StandardScaler before training to predict stock price N days ahead!"

            elif any(w in p for w in ["moving average", "ma5", "ma10", "ma20"]):
                answer = "📊 **Moving Averages** smooth price fluctuations:\n- **MA5** = 5-day average (short term)\n- **MA10** = 10-day average (medium term)\n- **MA20** = 20-day average (long term)\n\nWhen MA5 crosses above MA20 it often signals a price increase!"

            elif any(w in p for w in ["ogdc", "oil gas"]):
                answer = "🛢️ **OGDC** (Oil & Gas Development Company) is Pakistan's largest state-owned energy company. Enter **OGDC.KA** in the predictor tab to see its stock prediction!"

            elif any(w in p for w in ["hbl", "habib bank"]):
                answer = "🏦 **HBL** (Habib Bank Limited) is Pakistan's largest private bank. Enter **HBL.KA** in the predictor tab to see its stock prediction!"

            elif any(w in p for w in ["luck", "lucky cement"]):
                answer = "🏗️ **LUCK** (Lucky Cement) is one of Pakistan's largest cement manufacturers. Enter **LUCK.KA** in the predictor tab!"

            elif any(w in p for w in ["engro"]):
                answer = "🌱 **ENGRO Corporation** is a Pakistani conglomerate in fertilizers, food and energy. Enter **ENGRO.KA** in the predictor tab!"

            elif any(w in p for w in ["psx", "pakistan stock", "karachi stock", "stock market"]):
                answer = "🏦 **PSX (Pakistan Stock Exchange)** is Pakistan's main stock market in Karachi. Formed in 2016, it lists 500+ companies across energy, banking, cement and fertilizer sectors!"

            elif any(w in p for w in ["feature importance", "which feature", "important"]):
                answer = "🔍 In our model (after StandardScaler), **Moving Averages** carry the most weight. The Feature Importance chart shows standardized coefficients — now directly comparable because all features are on the same scale!"

            elif any(w in p for w in ["overfit", "overfitting", "generalize"]):
                answer = "⚠️ **Overfitting** = model memorizes training data but fails on new data. Our **Learning Curve** uses TimeSeriesSplit (respects time order) to show how the model generalizes. If training and validation scores converge, the model is healthy!"

            elif any(w in p for w in ["learning curve"]):
                answer = "📈 The **Learning Curve** shows model performance vs training data size. We use **TimeSeriesSplit** (not random KFold) so future data never leaks into training. If training and validation scores converge, the model generalizes well!"

            elif any(w in p for w in ["residual", "residual plot"]):
                answer = "📉 **Residuals** = Actual Price minus Predicted Price. Random residuals around zero = good model fit. A pattern in residuals means the model is missing something (like volatility or volume)!"

            elif any(w in p for w in ["nlp", "sentiment", "text", "natural language"]):
                answer = "📰 **NLP Sentiment Analysis** (Part 5) classifies financial headlines as Positive or Negative using:\n1. Text cleaning\n2. Tokenization\n3. Stopword removal\n4. Lemmatization\n5. TF-IDF vectorization\n6. Logistic Regression classifier"

            elif any(w in p for w in ["hello", "hi", "hey", "salam", "assalam", "good morning", "good evening"]):
                answer = "👋 Hello! I'm your PSX AI Assistant. I can help with:\n- 📈 Stock predictions (will OGDC increase?)\n- 📊 ML concepts (what is R²? what is StandardScaler?)\n- 🧠 Project results (ANN vs CNN)\n- 🏦 PSX companies (what is HBL?)"

            elif any(w in p for w in ["thank", "thanks", "shukriya", "jazak", "great", "good job"]):
                answer = "😊 You're welcome! Your project covers Regression, Logistic Regression, KNN, ANN, CNN, and NLP Sentiment Analysis — very impressive AI Lab work!"

            elif any(w in p for w in ["project", "presentation", "parts", "complete"]):
                answer = "🎓 Your project covers:\n- ✅ **Part 1:** Linear Regression — PSX Stock Predictor (with scaling + confidence range)\n- ✅ **Part 2:** Logistic Regression + KNN — Wine dataset (100%)\n- ✅ **Part 3:** ANN on Iris (100% accuracy)\n- ✅ **Part 4:** ANN vs CNN on Fashion-MNIST (CNN 91.63%)\n- ✅ **Part 5:** NLP Sentiment Analysis on Financial News"

            else:
                answer = "🤔 Try asking about:\n- **Stock predictions** (will HBL increase? how much will OGDC gain?)\n- **ML metrics** (what is R²? what is RMSE? what is StandardScaler?)\n- **Models** (explain CNN, what is KNN?, what is linear regression?)\n- **PSX companies** (what is OGDC? tell me about HBL)\n- **Project parts** (what did ANN score? how does NLP work?)"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Chat cleared! Ask me anything about PSX stocks or machine learning!"
        }]
        st.rerun()
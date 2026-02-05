from flask import Flask, render_template, request
import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# ----------------- CONFIG -----------------
os.environ.setdefault("NEWS_API_KEY", "1b2519a9542b4c108e5678753641c6f8")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ----------------- Initialize VADER -----------------
analyzer = SentimentIntensityAnalyzer()

# ----------------- Flask -----------------
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index2.html")
@app.route("/health")
def health():
    return "App is running"

def fetch_news_texts(query, from_date, to_date, api_key, page_size=100):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
        "apiKey": api_key
    }
    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    articles = data.get("articles", [])
    texts = []
    for art in articles:
        title = art.get("title") or ""
        desc = art.get("description") or ""
        text = (title + ". " + desc).strip()
        pub = art.get("publishedAt", "")
        pub_date = pub[:10] if pub else ""
        if text:
            texts.append((text, pub_date))
    return texts

def get_daily_sentiment_from_texts(text_date_pairs):
    if not text_date_pairs:
        return pd.DataFrame(columns=["date", "sentiment_score", "label"])

    records = []
    for text, date_str in text_date_pairs:
        if not date_str:
            continue
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "POSITIVE"
        elif compound <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        records.append({"date": pd.to_datetime(date_str).date(),
                        "score": compound,
                        "label": label})

    if not records:
        return pd.DataFrame(columns=["date", "sentiment_score", "label"])

    df = pd.DataFrame(records)
    daily_score = df.groupby("date", as_index=False)["score"].mean().rename(columns={"score":"sentiment_score"})
    label_df = df.groupby("date")["label"].agg(lambda x: x.mode().iat[0] if not x.mode().empty else "NEUTRAL").reset_index()
    merged = pd.merge(daily_score, label_df, on="date", how="left")
    return merged

def fetch_intraday_15m(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, interval="15m")
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "datetime"}, inplace=True)
    df["date"] = pd.to_datetime(df["datetime"]).dt.date
    if "Close" not in df.columns:
        close_cols = [c for c in df.columns if c.lower().endswith("close")]
        if close_cols:
            df.rename(columns={close_cols[0]: "Close"}, inplace=True)
    return df

@app.route("/predict", methods=["POST"])
def predict():
    user_ticker = request.form.get("stock", "").strip()
    user_dt_str = request.form.get("datetime", "").strip()

    if not user_ticker or not user_dt_str:
        return render_template("index2.html", error="Please provide both stock ticker and date-time (YYYY-MM-DD HH:MM).")

    try:
        requested_dt = pd.to_datetime(user_dt_str).tz_localize(None)
    except Exception as e:
        return render_template("index2.html", error=f"Invalid datetime: {e}. Use YYYY-MM-DD HH:MM")

    today = datetime.now().date()
    start_date = today - timedelta(days=30)
    end_date = today

    try:
        # 1️⃣ Fetch News + Sentiment
        query = f'("{user_ticker}" OR "{user_ticker.upper()}") AND (stock OR shares OR earnings OR market OR price)'
        texts = fetch_news_texts(query, str(start_date), str(end_date), NEWS_API_KEY)
        daily_sent = get_daily_sentiment_from_texts(texts)

        if daily_sent.empty:
            daily_sent = pd.DataFrame({
                "date": [d.date() for d in pd.date_range(start=start_date, end=end_date)],
                "sentiment_score": [0.0]*((end_date - start_date).days + 1),
                "label": ["NEUTRAL"]*((end_date - start_date).days + 1)
            })

        # 2️⃣ Fetch Stock Data
        stock_df = fetch_intraday_15m(user_ticker, start_date, end_date)
        if stock_df.empty:
            return render_template("index2.html", error=f"No 15-min data for {user_ticker}")

        stock_df["datetime"] = pd.to_datetime(stock_df["datetime"]).dt.tz_localize(None)
        stock_df["date"] = stock_df["datetime"].dt.date

        daily_sent["date"] = pd.to_datetime(daily_sent["date"]).dt.date
        merged = pd.merge(stock_df, daily_sent, on="date", how="left")
        merged["sentiment_score"].fillna(0.0, inplace=True)
        merged["label"].fillna("NEUTRAL", inplace=True)

        # 3️⃣ Prophet
        prophet_df = merged[["datetime", "Close", "sentiment_score"]].rename(columns={"datetime": "ds", "Close": "y"})
        m = Prophet()
        m.add_regressor("sentiment_score")
        m.fit(prophet_df)

        target_ts = requested_dt
        last_hist = prophet_df["ds"].max()

        if target_ts <= last_hist:
            sent_val = daily_sent.loc[daily_sent["date"] == target_ts.date(), "sentiment_score"]
            target_sent = float(sent_val.iloc[0]) if not sent_val.empty else 0.0
            df_pred = pd.DataFrame({"ds": [target_ts], "sentiment_score": [target_sent]})
            pred = m.predict(df_pred)
            prophet_price = float(pred["yhat"].iloc[0])
        else:
            freq = "15T"
            future_index = pd.date_range(start=last_hist + pd.Timedelta(minutes=15), end=target_ts, freq=freq)
            future_df = pd.DataFrame({"ds": future_index})
            future_df["sentiment_score"] = future_df["ds"].dt.date.map(
                lambda d: float(daily_sent.loc[daily_sent["date"] == d, "sentiment_score"].iloc[0])
                if d in list(daily_sent["date"]) else 0.0
            )
            pred_future = m.predict(future_df)
            prophet_price = float(pred_future[pred_future["ds"] == target_ts]["yhat"].iloc[0])                             if target_ts in list(pred_future["ds"]) else float(pred_future["yhat"].iloc[-1])

        # 4️⃣ XGBoost
        xdf = merged[["datetime", "Close", "sentiment_score"]].dropna().copy()
        xdf["day"] = xdf["datetime"].dt.day
        xdf["month"] = xdf["datetime"].dt.month
        xdf["year"] = xdf["datetime"].dt.year
        xdf["hour"] = xdf["datetime"].dt.hour
        xdf["minute"] = xdf["datetime"].dt.minute
        xdf["dayofweek"] = xdf["datetime"].dt.dayofweek
        X = xdf[["day", "month", "year", "hour", "minute", "dayofweek", "sentiment_score"]].values
        y = xdf["Close"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        xgb = XGBRegressor(n_estimators=100, verbosity=0)
        xgb.fit(X_scaled, y)

        req_dt = requested_dt
        req_features = np.array([[req_dt.day, req_dt.month, req_dt.year,
                                  req_dt.hour, req_dt.minute, req_dt.dayofweek,
                                  float(daily_sent.loc[daily_sent["date"] == req_dt.date(), "sentiment_score"].iloc[0])
                                  if req_dt.date() in list(daily_sent["date"]) else 0.0]])
        req_scaled = scaler.transform(req_features)
        xgb_price = float(xgb.predict(req_scaled)[0])

        req_label = daily_sent.loc[daily_sent["date"] == req_dt.date(), "label"]
        label_display = req_label.iloc[0].title() if not req_label.empty else "Neutral"

        return render_template("index2.html",
                               ticker=user_ticker.upper(),
                               input_datetime=str(requested_dt),
                               prophet_price=round(prophet_price, 4),
                               xgb_price=round(xgb_price, 4),
                               sentiment_label=label_display)

    except Exception as e:
        return render_template("index2.html", error=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
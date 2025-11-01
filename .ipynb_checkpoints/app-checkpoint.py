from flask import Flask, render_template, request
import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# ----------------- CONFIG -----------------
os.environ.setdefault("NEWS_API_KEY", "1b2519a9542b4c108e5678753641c6f8")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ----------------- Load FinBERT once -----------------
finbert_pipeline = None
finbert_load_error = None
try:
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    finbert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
except Exception as e:
    finbert_pipeline = None
    finbert_load_error = str(e)

# ----------------- Flask -----------------
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index2.html")

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

    if finbert_pipeline is None:
        rows = []
        for text, date_str in text_date_pairs:
            if date_str:
                rows.append({"date": pd.to_datetime(date_str).date(), "score": 0.0, "label": "NEUTRAL"})
        if not rows:
            return pd.DataFrame(columns=["date","sentiment_score","label"])
        df = pd.DataFrame(rows)
        daily = df.groupby("date", as_index=False)["score"].mean().rename(columns={"score":"sentiment_score"})
        daily["label"] = "NEUTRAL"
        return daily

    texts = [t for t, d in text_date_pairs]
    results = finbert_pipeline(texts, batch_size=16, truncation=True)

    records = []
    for (text, date_str), res in zip(text_date_pairs, results):
        if not date_str:
            continue
        try:
            d = pd.to_datetime(date_str).date()
        except:
            continue
        score = float(res.get("score", 0.0))
        label = res.get("label", "").upper()
        records.append({"date": d, "score": score, "label": label})

    if not records:
        return pd.DataFrame(columns=["date","sentiment_score","label"])

    df = pd.DataFrame(records)
    daily_score = df.groupby("date", as_index=False)["score"].mean().rename(columns={"score":"sentiment_score"})
    label_df = df.groupby("date")["label"].agg(lambda x: x.mode().iat[0] if not x.mode().empty else "NEUTRAL").reset_index()
    merged = pd.merge(daily_score, label_df, on="date", how="left")
    return merged

def fetch_intraday_15m(ticker, start_date, end_date):
    yf_start = str(start_date)
    yf_end = str(end_date + timedelta(days=1))
    df = yf.download(ticker, start=start_date, end=end_date, interval="15m")

    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime':'datetime'}, inplace=True)

    df['date'] = pd.to_datetime(df['Datetime'] if 'Datetime' in df.columns else df['datetime']).dt.date
    if 'Close' not in df.columns:
        close_cols = [c for c in df.columns if c.lower().endswith('close')]
        if close_cols:
            df.rename(columns={close_cols[0]:'Close'}, inplace=True)
    return df

@app.route('/predict', methods=['POST'])
def predict():
    if finbert_load_error:
        finbert_msg = f"FinBERT load error at startup: {finbert_load_error}. Running with neutral sentiment."
    else:
        finbert_msg = None

    user_ticker = request.form.get('stock', '').strip()
    user_dt_str = request.form.get('datetime', '').strip()

    if not user_ticker or not user_dt_str:
        return render_template("index.html", error="Please provide both stock ticker and date-time (YYYY-MM-DD HH:MM).")

    try:
        requested_dt = pd.to_datetime(user_dt_str).tz_localize(None)
    except Exception as e:
        return render_template("index.html", error=f"Could not parse datetime: {e}. Use format YYYY-MM-DD HH:MM")

    today = datetime.now().date()
    past = today - timedelta(days=30)
    start_date = past
    end_date = today

    try:
        query = f'("{user_ticker}" OR "{user_ticker.upper()}") AND (stock OR shares OR earnings OR market OR price)'
        texts = fetch_news_texts(query, from_date=str(start_date), to_date=str(end_date), api_key=NEWS_API_KEY, page_size=100)
        daily_sent = get_daily_sentiment_from_texts(texts)

        if daily_sent.empty:
            daily_sent = pd.DataFrame({
                "date": [d for d in pd.date_range(start=start_date, end=end_date).date],
                "sentiment_score": [0.0]*((end_date-start_date).days+1),
                "label": ["NEUTRAL"]*((end_date-start_date).days+1)
            })

        stock_df = fetch_intraday_15m(user_ticker, start_date, end_date)
        if stock_df.empty:
            return render_template("index2.html", error=f"No intraday 15-min data found for {user_ticker} in the last 30 days (yfinance returned empty).")

        if 'Datetime' in stock_df.columns:
            stock_df['datetime'] = pd.to_datetime(stock_df['Datetime']).dt.tz_localize(None)
        else:
            if 'datetime' not in stock_df.columns:
                stock_df['datetime'] = pd.to_datetime(stock_df.iloc[:,0]).dt.tz_localize(None)
            else:
                stock_df['datetime'] = pd.to_datetime(stock_df['datetime']).dt.tz_localize(None)

        if 'Close' not in stock_df.columns:
            close_candidates = [c for c in stock_df.columns if c.lower().endswith('close')]
            if close_candidates:
                stock_df.rename(columns={close_candidates[0]:'Close'}, inplace=True)
            else:
                return render_template("index2.html", error="Could not find Close price column in stock data.")

        stock_df['date'] = stock_df['datetime'].dt.date
        daily_sent['date'] = pd.to_datetime(daily_sent['date']).dt.date
        merged = pd.merge(stock_df, daily_sent[['date','sentiment_score','label']], on='date', how='left')
        merged['sentiment_score'] = merged['sentiment_score'].fillna(method='ffill').fillna(0.0)
        merged['label'] = merged['label'].fillna(method='ffill').fillna("NEUTRAL")

        # Prophet
        prophet_df = merged[['datetime','Close','sentiment_score']].rename(columns={'datetime':'ds','Close':'y'}).dropna()
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)

        m = Prophet()
        m.add_regressor('sentiment_score')
        m.fit(prophet_df)

        last_hist = prophet_df['ds'].max()
        target_ts = pd.to_datetime(requested_dt).tz_localize(None)

        if target_ts <= last_hist:
            target_date = target_ts.date()
            sent_row = daily_sent[daily_sent['date'] == target_date]
            if not sent_row.empty:
                target_sent = float(sent_row['sentiment_score'].iloc[0])
            else:
                target_sent = float(daily_sent['sentiment_score'].iloc[-1])
            df_for_pred = pd.DataFrame({'ds':[target_ts], 'sentiment_score':[target_sent]})
            df_for_pred['ds'] = df_for_pred['ds'].dt.tz_localize(None)
            pred = m.predict(df_for_pred)
            prophet_price = float(pred['yhat'].iloc[0])
        else:
            freq = '15T'
            future_index = pd.date_range(start=last_hist + pd.Timedelta(minutes=15), end=target_ts, freq=freq)
            future_df = pd.DataFrame({'ds': future_index})
            def get_sent_for_ts(ts):
                d = ts.date()
                row = daily_sent[daily_sent['date'] == d]
                if not row.empty:
                    return float(row['sentiment_score'].iloc[0])
                else:
                    return float(daily_sent['sentiment_score'].iloc[-1])
            future_df['sentiment_score'] = future_df['ds'].apply(get_sent_for_ts)
            future_df['ds'] = future_df['ds'].dt.tz_localize(None)
            pred_future = m.predict(future_df)
            prophet_price = float(pred_future[pred_future['ds'] == target_ts]['yhat'].iloc[0]) if target_ts in list(pred_future['ds']) else float(pred_future['yhat'].iloc[-1])

        # XGBoost
        xdf = merged[['datetime','Close','sentiment_score']].dropna().copy()
        xdf['day'] = xdf['datetime'].dt.day
        xdf['month'] = xdf['datetime'].dt.month
        xdf['year'] = xdf['datetime'].dt.year
        xdf['hour'] = xdf['datetime'].dt.hour
        xdf['minute'] = xdf['datetime'].dt.minute
        xdf['dayofweek'] = xdf['datetime'].dt.dayofweek
        X = xdf[['day','month','year','hour','minute','dayofweek','sentiment_score']].values
        y = xdf['Close'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        xgb = XGBRegressor(n_estimators=100, verbosity=0)
        xgb.fit(X_scaled, y)

        req_dt = pd.to_datetime(requested_dt).tz_localize(None)
        req_row = {
            'day': req_dt.day,
            'month': req_dt.month,
            'year': req_dt.year,
            'hour': req_dt.hour,
            'minute': req_dt.minute,
            'dayofweek': req_dt.dayofweek,
        }
        sent_row = daily_sent[daily_sent['date'] == req_dt.date()]
        if not sent_row.empty:
            req_sent = float(sent_row['sentiment_score'].iloc[0])
            req_label = sent_row['label'].iloc[0]
        else:
            req_sent = float(daily_sent['sentiment_score'].iloc[-1])
            req_label = daily_sent['label'].iloc[-1]

        req_features = np.array([[req_row['day'], req_row['month'], req_row['year'],
                                  req_row['hour'], req_row['minute'], req_row['dayofweek'],
                                  req_sent]])
        req_scaled = scaler.transform(req_features)
        xgb_price = float(xgb.predict(req_scaled)[0])

        label_display = str(req_label).title() if pd.notna(req_label) else "Neutral"

        prophet_price_rounded = round(prophet_price, 4)
        xgb_price_rounded = round(xgb_price, 4)

        return render_template("index2.html",
                               ticker=user_ticker.upper(),
                               input_datetime=str(requested_dt),
                               prophet_price=prophet_price_rounded,
                               xgb_price=xgb_price_rounded,
                               sentiment_label=label_display,
                               finbert_msg=finbert_msg)

    except Exception as e:
        return render_template("index2.html", error=f"Error during processing: {e}")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
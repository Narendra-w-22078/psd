from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("model_gempa.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    df = pd.read_csv(file)

    X = df[["magnitudo", "kedalaman", "jarak"]]
    preds = model.predict(X)

    df["Prediksi"] = ["GEMPA" if p != 0 else "TIDAK GEMPA" for p in preds]

    return jsonify({
        "total": len(df),
        "gempa": int((df["Prediksi"] == "GEMPA").sum()),
        "tidak": int((df["Prediksi"] == "TIDAK GEMPA").sum()),
        "preview": df.head(10).to_dict(orient="records")
    })

if __name__ == "__main__":
    app.run()

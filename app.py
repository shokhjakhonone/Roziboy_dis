from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import json
import subprocess
import logging

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Функции для сохранения и загрузки настроек
def save_settings(settings):
    with open("config.json", "w") as f:
        json.dump(settings, f)

def load_settings():
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"method": "kmeans", "n_clusters": 3, "eps": 0.5, "min_samples": 5}

# Вызов функции для создания графика
def create_cluster_chart():
    subprocess.run(["python", "create_cluster_chart.py"])

# Главная страница с формой загрузки
@app.route("/", methods=["GET", "POST"])
def upload_file():
    settings = load_settings()
    if request.method == "POST":
        logging.info("Начало загрузки файла и настроек кластеризации")
        
        # Загрузка файла и параметров
        file = request.files["file"]
        settings["method"] = request.form["method"]
        settings["n_clusters"] = int(request.form["n_clusters"]) if settings["method"] == "kmeans" else None
        settings["eps"] = float(request.form["eps"]) if settings["method"] == "dbscan" else None
        settings["min_samples"] = int(request.form["min_samples"]) if settings["method"] == "dbscan" else None
        
        # Сохранение настроек
        save_settings(settings)
        
        # Подготовка данных из student-mat.csv
        data = pd.read_csv(file, sep=';')
        data_numeric = data[['age', 'absences', 'G1', 'G2', 'G3']].dropna()
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_numeric)
 
        # Кластеризация
        if settings["method"] == "kmeans":
            kmeans = KMeans(n_clusters=settings["n_clusters"], random_state=0, n_init=10)
            data["cluster"] = kmeans.fit_predict(data_scaled)
        elif settings["method"] == "dbscan":
            dbscan = DBSCAN(eps=settings["eps"], min_samples=settings["min_samples"])
            data["cluster"] = dbscan.fit_predict(data_scaled)

        # Сохранение результатов и обновление графика
        data.to_csv("static/student_clusters.csv", index=False)
        create_cluster_chart()
        logging.info("Файл кластеров сохранен, график обновлен")
        
        return redirect(url_for("results"))
    return render_template("upload.html")

# Страница с результатами
@app.route("/results")
def results():
    data = pd.read_csv("static/student_clusters.csv")
    return render_template("results.html", tables=[data.to_html(classes='data', index=False)], titles=data.columns.values)

if __name__ == "__main__":
    app.run(debug=True, port=11)

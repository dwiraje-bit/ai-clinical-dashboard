from flask import Flask, request, render_template_string, send_file, redirect, url_for, session
import pandas as pd
import sqlite3
from datetime import datetime
import os
import base64
from io import BytesIO

# ===== FIX FOR DEPLOYMENT =====
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "secret123"

# ===== DATABASE =====
conn = sqlite3.connect("history.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_diabetes TEXT,
    result_bp TEXT,
    probability TEXT,
    explanation TEXT,
    time TEXT
)
""")
conn.commit()

# ===== DATA =====
data = pd.read_csv("merged_diabetes_hypertension_dataset.csv")

le_gender = LabelEncoder()
le_activity = LabelEncoder()

data["Gender"] = le_gender.fit_transform(data["Gender"])
data["PhysicalActivity"] = le_activity.fit_transform(data["PhysicalActivity"])

X = data.drop(["Diabetes", "Hypertension"], axis=1)
y_d = data["Diabetes"]
y_b = data["Hypertension"]

X_train, _, y_train_d, _ = train_test_split(X, y_d, test_size=0.2, random_state=42)
X_train2, _, y_train_b, _ = train_test_split(X, y_b, test_size=0.2, random_state=42)

model_d = RandomForestClassifier()
model_b = RandomForestClassifier()

model_d.fit(X_train, y_train_d)
model_b.fit(X_train2, y_train_b)

styles = getSampleStyleSheet()
last_report = {}

# ===== LOGIN =====
login_html = """
<html>
<body style='font-family:Segoe UI;background:#1e293b;color:white;text-align:center;padding-top:150px;'>
<h2>🔐 Doctor Login</h2>
<form method="POST">
<input name="username" placeholder="Username" required><br><br>
<input type="password" name="password" placeholder="Password" required><br><br>
<button>Login</button>
</form>
<p style="color:red;">{{error}}</p>
</body>
</html>
"""

# ===== MAIN UI =====
html = """<!DOCTYPE html>
<html>
<head>
<style>
body{margin:0;font-family:Segoe UI;background:#f4f6f9;}
.header{font-size:28px;padding:20px;font-weight:bold;}
.container{padding:20px;}
.card{background:white;padding:20px;border-radius:15px;box-shadow:0 10px 25px rgba(0,0,0,0.1);margin-bottom:20px;}
button{padding:12px;width:100%;border:none;border-radius:10px;background:#16a34a;color:white;font-size:16px;}
.result{padding:12px;border-radius:10px;margin-top:10px;text-align:center;font-weight:bold;}
.high{background:#ef4444;color:white;}
.low{background:#22c55e;color:white;}
.badge{display:inline-block;padding:6px 10px;margin:5px;border-radius:20px;background:#e0f2fe;color:#0369a1;}
table{width:100%;border-collapse:collapse;}
th,td{padding:10px;border-bottom:1px solid #ddd;text-align:center;}
th{background:#111827;color:white;}
</style>
</head>
<body>

<div class="header">🚀 AI Clinical Dashboard</div>

<div class="container">

<div class="card">
<form method="POST">
<button name="action" value="random">🎲 Generate Smart Patient</button>
</form>

{% if patient %}
<h3>Patient Data</h3>
{% for k,v in patient.items() %}
<span class="badge"
style="
{% if (k == 'Glucose' and v > 140) or
      (k == 'BMI' and v > 30) or
      (k == 'BloodPressure' and v > 90) or
      (k == 'Insulin' and v > 180) or
      (k == 'Age' and v > 60) %}
    background:#ef4444;color:white;
{% endif %}
">
{{k}}: {{v}}
</span>
{% endfor %}
{% endif %}

{% if diabetes %}
<div class="result {{'high' if 'High' in diabetes else 'low'}}">
{{diabetes}} ({{prob_d}}%)
</div>
{% endif %}

{% if bp %}
<div class="result {{'high' if 'High' in bp else 'low'}}">
{{bp}} ({{prob_b}}%)
</div>
{% endif %}

{% if explanation %}
<h3>🧠 AI Explanation</h3>
<p>{{explanation}}</p>

<form action="/download">
<button>📄 Download Report</button>
</form>
{% endif %}
</div>

<div class="card">
<h3>📊 Patient History</h3>
<table>
<tr>
<th>Diabetes</th>
<th>BP</th>
<th>Probability</th>
<th>Date</th>
</tr>
{% for row in history %}
<tr>
<td>{{row[0]}}</td>
<td>{{row[1]}}</td>
<td>{{row[2]}}</td>
<td>{{row[3]}}</td>
</tr>
{% endfor %}
</table>
</div>

</div>
</body>
</html>
"""

# ===== ROUTES =====
@app.route("/", methods=["GET","POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "1234":
            session["user"] = "admin"
            return redirect(url_for("home"))
        else:
            error = "Invalid Credentials"
    return render_template_string(login_html, error=error)

@app.route("/home", methods=["GET","POST"])
def home():
    if "user" not in session:
        return redirect(url_for("login"))

    global last_report

    diabetes = bp = explanation = patient = None
    prob_d = prob_b = None

    if request.method == "POST":
        row = X.sample(1)
        patient = row.to_dict(orient="records")[0]

        pred_d = model_d.predict(row)[0]
        pred_b = model_b.predict(row)[0]

        prob_d = round(model_d.predict_proba(row)[0][1]*100,2)
        prob_b = round(model_b.predict_proba(row)[0][1]*100,2)

        diabetes = "High Risk of Diabetes" if pred_d==1 else "Low Risk of Diabetes"
        bp = "High Risk of Hypertension" if pred_b==1 else "Normal Blood Pressure"

        imp = model_d.feature_importances_
        idx = imp.argsort()[-3:][::-1]
        top = [X.columns[i] for i in idx]

        explanation = f"Key factors: {top[0]}, {top[1]}, {top[2]}"

        last_report = {
            "patient": patient,
            "diabetes": diabetes,
            "bp": bp,
            "prob_d": prob_d,
            "prob_b": prob_b,
            "explanation": explanation
        }

        cursor.execute(
            "INSERT INTO history (result_diabetes,result_bp,probability,explanation,time) VALUES (?,?,?,?,?)",
            (diabetes, bp, f"{prob_d}%/{prob_b}%", explanation, str(datetime.now()))
        )
        conn.commit()

    cursor.execute("SELECT result_diabetes,result_bp,probability,time FROM history ORDER BY id DESC")
    history = cursor.fetchall()

    return render_template_string(html, **locals())

# ===== REPORT =====
@app.route("/download")
def download():
    doc = SimpleDocTemplate("report.pdf")
    elements = []

    elements.append(Paragraph("<b>TERNA COLLEGE OF ENGINEERING</b>", styles['Title']))
    elements.append(Spacer(1,10))

    probs = [last_report.get("prob_d",0), last_report.get("prob_b",0)]
    labels = ["Diabetes", "BP"]

    plt.figure()
    plt.bar(labels, probs)
    plt.savefig("graph.png")
    plt.close()

    data_table = [["Parameter","Value"]]
    for k,v in last_report.get("patient",{}).items():
        data_table.append([k,str(v)])

    table = Table(data_table)
    table.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),1,colors.black),
        ("BACKGROUND",(0,0),(-1,0),colors.darkblue),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white)
    ]))

    elements.append(table)
    elements.append(Spacer(1,20))
    elements.append(Image("graph.png", width=400, height=250))

    doc.build(elements)
    return send_file("report.pdf", as_attachment=True)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ===== FIXED RUN =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)







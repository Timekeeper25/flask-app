import pandas as pd
from sklearn.cluster import KMeans
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from flask import Flask, render_template, url_for, redirect, request, session, jsonify
import seaborn as sns
from flask import Flask, request
from flask import render_template, url_for, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import re
from statistics import mean
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import subprocess
import pickle
import os

# mapping
lp = {
    1: 'L',
    2: 'P'
}

Minat = {
    1: 'Kepemimpinan', 
    2: 'Aktivitas Fisik dan Mengikuti Kompetisi', 
    3: 'Pelayanan Kemanusiaan',
    4: 'Bermusik dan Tampil di Publik', 
    5: 'Belajar Budaya Lokal', 
    6: 'Petualangan dan Kerja Tim'
}

Bakat = {
    1: 'Pengaturan Formasi di Paskibra',
    2: 'Berbagai Macam Olahraga', 
    3: 'Merawat dan Memberikan Pertolongan Kepada Orang yang Membutuhkan', 
    4: 'Bangun Perkemahan dan Inisiatif dalam Kegiatan Outdoor', 
    5: 'Memainkan Instrumen Musik', 
    6: 'Memainkan Instrumen Tradisional'
}

Hobi = {
    1: 'Menonton film', 
    2: 'Menyanyi atau Bermain Musik',
    3: 'Bermain game', 
    4: 'Olahraga',
    5: 'Menulis',
    6: 'Fotografi', 
    7: 'Membaca', 
    8: 'Melukis atau menggambar', 
    9: 'Memasak'
}

Ekstrakurikuler = {
    1: 'Karawitan',
    2: 'Marching Band',
    3: 'Olahraga',
    4: 'PMR',
    5: 'Paskibra',
    6: 'Pramuka'
}

def get_number_from_word(dictionary, word):
    word_lower = word.lower()
    for number, value in dictionary.items():
        value_lower = value.lower()
        if value_lower == word_lower:
            return number
    return None

def get_word_from_number(dictionary, number):
    return dictionary.get(number, None)

import csv

csv_file="quesioner.csv"
def save_to_csv(data):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        fieldnames = ['NISN', 'Nama Siswa', 'L/P', 'Minat', 'Bakat', 'Hobi', 'Tahun']
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=',')

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

def map_ekskul(df, col_name, ekskul_dict):
    if col_name not in df.columns or df[col_name].dtype != 'object':
        df[col_name] = df[col_name].astype('object')
    
    df.loc[:, col_name] = df['Cluster'].map(lambda x: ekskul_dict[(x - 1) % len(ekskul_dict)])
    return df

def mapping(df, col_name, map_func):
    if col_name not in df.columns or df[col_name].dtype != 'object':
        df[col_name] = df[col_name].astype('object')
    
    df.loc[:, col_name] = df[col_name].map(map_func)
    return df


####################
##########################################
app = Flask(__name__)
app.secret_key = 'secret'

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password=request.form['password']
        session['username'] = username
        if username=='admin' and password=='admin':
            session['level'] = 'admin'
            #processing()
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/quesioner', methods=['GET', 'POST'])
def quesioner():
    if request.method == 'POST':
        data = {
            'NISN': request.form['nisn'],
            'Nama Siswa': request.form['nama'],
            'L/P': request.form['lp'],
            'Minat': request.form['minat'],
            'Bakat': request.form['bakat'],
            'Hobi': request.form['hobi'],
            'Tahun': datetime.now().strftime('%Y')
        }

        filename = 'quesioner.csv'
        if os.path.isfile(filename):
            data_csv = pd.read_csv(filename)
            data_nisn = int(data['NISN'])
            csv_nisns = data_csv['NISN'].astype(int)

            if data_nisn in csv_nisns.values:
                error_message = "Data Anda Sudah Ada"
                return render_template('quesioner.html', error_message=error_message)
            else:
                save_to_csv(data)
                processed()
                return render_template('login.html')
        else:
            save_to_csv(data)
            return render_template('login.html')
    
    return render_template('quesioner.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    df = pd.read_csv('dataset.csv',sep=";")
    if request.method == 'POST':
        file = request.files['filedata']
        file.save('dataset.csv')
        df = pd.read_csv('dataset.csv',index_col=0)
        #processing()

    return render_template('dashboard.html', column_names=df.columns.values, row_data=list(df.values.tolist()),
                           dataset=len(list(df.values.tolist())), zip=zip)

@app.route('/dataset')
def dataset():
    df = pd.read_csv("dataset.csv",sep=";")
    return render_template('dataset.html', column_names=df.columns.values, row_data=list(df.values.tolist()),dataset=len(list(df.values.tolist())), zip=zip)

@app.route('/processed')
def processed():
    df = pd.read_csv("dataset.csv", sep=";")

    df['L/P'] = df['L/P'].apply(lambda x: get_number_from_word(lp, x))
    df['Minat'] = df['Minat'].apply(lambda x: get_number_from_word(Minat, x))
    df['Bakat'] = df['Bakat'].apply(lambda x: get_number_from_word(Bakat, x))
    df['Hobi'] = df['Hobi'].apply(lambda x: get_number_from_word(Hobi, x))

    df.to_csv("processed.csv", index=False)
    return render_template('processed.html', column_names=df.columns.values, row_data=list(df.values.tolist()),
                           dataset=len(list(df.values.tolist())), zip=zip)

@app.route('/kmeans', methods=['GET', 'POST'])
def kmeans():
    # Membaca data dari file CSV
    df = pd.read_csv('processed.csv',sep=",",index_col = False)
    print(df.columns)
    # Melakukan klasterisasi dengan K-Means hanya pada kolom yang relevan
    X = ['L/P', 'Minat', 'Bakat', 'Hobi']
    for tahun in df['Tahun'].unique():
        data_tahun = df[df['Tahun'] == tahun][X]
        # Melakukan klasterisasi dengan K-Means
        n_clusters = 6
        if len(data_tahun) >= n_clusters:
            # Melakukan klasterisasi dengan K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_tahun)
            df.loc[df['Tahun'] == tahun, 'Cluster'] = kmeans.labels_ + 1
        else:
            print(f"Skip tahun {tahun} karena kurangnya sampel ({len(data_tahun)}) dari cluster ({n_clusters})")

    df['Ekstrakurikuler'] = df['Cluster'].map(Ekstrakurikuler)
    df = mapping(df, 'L/P', lp)
    df = mapping(df, 'Minat', Minat)
    df = mapping(df, 'Bakat', Bakat)
    df = mapping(df, 'Hobi', Hobi)
            
    df.to_csv("kmeans.csv")
    # Menampilkan hasil klasterisasi
    return render_template('kmeans.html', column_names=df.columns.values, row_data=list(df.values.tolist()),
                           dataset=len(list(df.values.tolist())), zip=zip)

@app.route('/predict', methods=['GET','POST'])
def predict():
    df = pd.read_csv('quesioner.csv')
    tahun_options = sorted(set(df['Tahun'].astype(str)))

    if request.method == 'POST':
        tahun_dipilih = request.form.get('tahun')
        print(f"Tahun dipilih: {tahun_dipilih}")

        df_filter = df[df['Tahun'] == int(tahun_dipilih)]
        print(f"Filtered DataFrame:\n{df_filter.head()}")  # Debug print

        if df_filter.empty:
            return jsonify({"error": "No data available for the selected year."})

        df_filter['L/P'] = df_filter['L/P'].apply(lambda x: get_number_from_word(lp, x))
        df_filter['Minat'] = df_filter['Minat'].apply(lambda x: get_number_from_word(Minat, x))
        df_filter['Bakat'] = df_filter['Bakat'].apply(lambda x: get_number_from_word(Bakat, x))
        df_filter['Hobi'] = df_filter['Hobi'].apply(lambda x: get_number_from_word(Hobi, x))

        X = ['L/P', 'Minat', 'Bakat', 'Hobi']

        n_clusters = 6
        if len(df_filter) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df_filter[X])
            df_filter['Cluster'] = kmeans.labels_ + 1
            print(f"Cluster labels added:\n{df_filter[['NISN', 'Cluster']].head()}")  # Debug print
        else:
            print(f"Skip tahun {tahun_dipilih} karena kurangnya sampel ({len(df_filter)}) dari cluster ({n_clusters})")
            return jsonify({"error": f"Not enough samples for clustering in year {tahun_dipilih}."})

        df_filter['Ekstrakurikuler'] = df_filter['Cluster'].map(Ekstrakurikuler)
        df_filter = mapping(df_filter, 'L/P', lp)
        df_filter = mapping(df_filter, 'Minat', Minat)
        df_filter = mapping(df_filter, 'Bakat', Bakat)
        df_filter = mapping(df_filter, 'Hobi', Hobi)

        print(f"Mapped DataFrame:\n{df_filter.head()}")  # Debug print
        df_filter.to_csv(f"Prediksi/{tahun_dipilih}.csv", index=False)

        return jsonify({
            "column_names": df_filter.columns.values.tolist(),
            "row_data": df_filter.fillna('').values.tolist()
        })

    return render_template('predict.html', tahun_options=tahun_options)

@app.route('/profil')
def profil():
    return render_template('profil.html')

@app.route('/logout')
def logout():
    return render_template('login.html')

if __name__ == '__main__':
  app.run(debug=True)
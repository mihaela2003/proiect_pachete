import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# setare stil grafice
sns.set_style("darkgrid")

# citire dataset
@st.cache_data
def load_data():
    return pd.read_csv("anime_cleaned.csv")

df = load_data()

# creare tab-uri
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["Descrierea Dataset-ului", "Explorarea Datelor", "Vizualizari & Filtrare",
                                              "Tratare valori lipsa si a valorilor extreme", "Codificarea datelor",
                                              "Agregare si prelucrari statistice", "Utilizarea functiilor de grup",
                                              "Geopandas", "Analiza datelor"])

# ---- DESCRIEREA DATASET-ULUI ----
with tab1:
    st.title("Descrierea Dataset-ului")

    st.subheader("Descriere Generala")
    st.write(
        "Acest dataset conține informatii despre diverse anime-uri, "
        "incluzand titlul, studioul producator, rating-ul si anul lansarii."
    )

    # descrierea coloanelor
    st.subheader("Coloane disponibile")
    st.markdown("""
    - **anime_id**: Id-ul anime-ului 
    - **title**: Numele anime-ului folosind romaji
    - **title_english**: Numele anime-ului in limba engleza
    - **type**: tipul de animw
    - **source**: Sursa de la care provine anime-ul (manga, light novel etc.)
    - **episodes**: Numarul de episoade  
    - **status**: Prezinta daca anime-ul a fost terminat de difuzat sau nu
    - **airing**: Variabila booleana care prezinta daca anime-ul mai este difuzat sau nu
    - **aired_string**: Perioada in care anime-ul a fost difuzat
    - **duration_min**: Durata minima a fiecarui episod (în minute)  
    - **rating**: Pentru ce grupa de varsta se incadreaza anime-ul
    - **score**: Score-ul IMDb/MAL  
    - **scored_by**: De cate persoane a fost votat anime-ul
    - **rank**: Rangul pe care la aprimit anime-ul in urma voturilor
    - **popularity**: Cat de popular este anime-ul
    - **members**: Cati membri are fandom-ul anime-ului respectiv pe platforma MyAnimeList
    - **favorites**: Cate persoane au adaugat la favorite anime-ul 
    - **studio**: Studio-ul care a produs anime-ul 
    - **genre**: Genurile anime-ului  
    - **aired_from_year**: Anul în care a fost lansat anime-ul   
    
    """)

    # link pentru descarcare dataset
    st.subheader("Descarca Dataset-ul")
    st.download_button(label="Descarca CSV", data=df.to_csv(index=False), file_name="anime_dataset.csv",
                       mime="text/csv")

# ----EXPLORAREA DATASET-ULUI ----
with tab2:
    st.title("Explorarea Dataset-ului")

    # selectare coloane de afișat
    st.subheader("Selecteaza coloanele pe care sa le afisam")
    selected_columns = st.multiselect("Alege coloanele", df.columns, default=df.columns)
    df_filtered = df[selected_columns]

    # verificam daca 'studio' exista inainte de filtrare
    if "studio" in df_filtered.columns:
        st.subheader("Filtrare Anime-uri dupa Studio")
        studio_list = df_filtered["studio"].dropna().unique()
        selected_studio = st.selectbox("Alege un studio", ["Toate"] + list(studio_list))
        if selected_studio != "Toate":
            df_filtered = df_filtered[df_filtered["studio"] == selected_studio]
    else:
        st.warning("Coloana 'studio' nu este selectata. Se afiseaza toate studiourile.")

    # verificam daca 'score' exista inainte de filtrare
    if "score" in df_filtered.columns:
        st.subheader("Filtrare dupa Score Minim")
        min_score = st.slider("Alege score-ul minim", float(df_filtered["score"].min()),
                              float(df_filtered["score"].max()), 7.0)
        df_filtered = df_filtered[df_filtered["score"] >= min_score]
    else:
        st.warning("Coloana 'score' nu este selectata. Filtrarea dupa scor a fost ignorata.")

    # afisare dataframe filtrat
    st.subheader("Datele Filtrate")
    st.dataframe(df_filtered)

    # descarcare dataset filtrat
    st.download_button(label="Descarcă Datele Filtrate", data=df_filtered.to_csv(index=False),
                       file_name="anime_filtered.csv", mime="text/csv")

# ----VIZUALIZARI & FILTRARE ----
with tab3:
    st.title("Vizualizari & Filtrare")

    # top 10 Studiouri cu cele mai multe anime-uri
    st.subheader("Top 10 Studiouri cu Cele Mai Multe Anime-uri")
    top_studios = df["studio"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(y=top_studios.index, x=top_studios.values,hue=top_studios.index, palette="coolwarm", ax=ax)
    ax.set_xlabel("Numar de Anime-uri")
    ax.set_ylabel("Studio")
    st.pyplot(fig)

    # top 10 anime-uri dupa rating
    st.subheader("Top 10 Anime-uri Dupa Score")
    top_anime = df.nlargest(10, "score")[["title", "score", "studio"]]
    st.dataframe(top_anime)

    # distribuția ratingurilor
    st.subheader("Distributia Ratingurilor Anime-urilor")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["rating"], bins=20, kde=True, color="blue", ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Numar de Anime-uri")
    st.pyplot(fig)

    # distribuția duratei episoadelor
    st.subheader("Distributia Duratei Episoadelor")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["duration_min"], bins=15, kde=True, color="green", ax=ax)
    ax.set_xlabel("Durata (minute)")
    ax.set_ylabel("Numar de Anime-uri")
    st.pyplot(fig)

    # distributia anime-urilor dupa anul lansarii
    st.subheader("Distributia Anime-urilor dupa Anul Lansarii")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["aired_from_year"], bins=30, kde=True, color="purple", ax=ax)
    ax.set_xlabel("Anul Lansarii")
    ax.set_ylabel("Numar de Anime-uri")
    st.pyplot(fig)

with tab4:
    st.title("Tratare valori lipsa")

    if "df" not in st.session_state:
        st.session_state.df = df

    df = st.session_state.df

    # analiza statistica descriptiva
    st.subheader("Statistici descriptive pentru coloanele numerice:")
    st.write(df.describe())

    # Calcul valori lipsă
    missing_vals = df.isnull().sum()
    missing_percent = (missing_vals / len(df)) * 100

    # Afișează numărul și procentul valorilor lipsă
    st.subheader("Valori Lipsa pe Coloane")
    missing_df = pd.DataFrame({
        "Valori Lipsa": missing_vals,
        "Procent (%)": missing_percent
    })
    missing_df = missing_df[missing_df["Valori Lipsa"] > 0].sort_values("Procent (%)", ascending=False)
    st.dataframe(missing_df)

    # Vizualizare grafică a valorilor lipsă
    st.subheader("Grafic - Procentul Valorilor Lipsa")
    if not missing_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        missing_df["Procent (%)"].plot(kind="barh", color="orange", ax=ax)
        ax.set_title("Procentul valorilor lipsa per coloana")
        ax.set_xlabel("Procent (%)")
        ax.set_ylabel("Coloana")
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        st.pyplot(fig)
    else:
        st.success("Nu există valori lipsă în dataset!")

    # o mica concluzie pt grafic si tabel
    st.write("In urma tabelului 'Valori Lipsa pe Coloane' si a graficului 'Grafic - Procentul Valorilor Lipsa' putem observa ca, coloana title_english are aproximativ 48% din valori lipsa.")
    st.write("Coloana title_english nu este una importanta in setul nostru de date, deci putem elimina aceasta coloana")

    if st.button("Elimina coloana title_english"):
        if 'title_english' in df.columns:
            df.drop(columns=['title_english'], inplace=True)
            st.success("Coloana 'title_english' a fost stearsa!")
        else:
            st.warning("Coloana 'title_english' nu exista in dataset.")

    # pt genre si rating
    st.write("Tinand cont ca, coloanele rating si genre au foarte putine randuri care nu au valoare, o sa eliminam aceste randuri din dataframe.")
    if st.button("Sterge randurile cu valori lipsa in 'genre' sau 'rating'"):
        initial_rows = len(df)
        df.dropna(subset=['genre', 'rating'], inplace=True)
        removed_rows = initial_rows - len(df)
        st.success(f"Sterse {removed_rows} randuri cu valori lipsa in 'genre' sau 'rating'!")
        st.write(f"Numar total de randuri ramase: {len(df)}")

    # pt coloana rank
    st.write("Pentru coloana rank o sa folosim mai multe metode prin care sa imputam valori pentru valorile lipsa")
    # imputare simpla
    simple_method = st.selectbox("Selectati metoda de imputare simpla:", ["mediana", "media", "modul"])
    if st.button("Aplica imputare simpla"):
        if simple_method == "mediana":
            impute_value = df['rank'].median()
        elif simple_method == "media":
            impute_value = df['rank'].mean()
        else:
            impute_value = df['rank'].mode()[0]
        df['rank'].fillna(impute_value, inplace=True)
        st.success(f"Valori lipsă în 'rank' completate cu {simple_method.lower()} ({impute_value:.2f})")
        st.write(f"Număr de valori lipsă rămase: {df['rank'].isnull().sum()}")

    st.subheader("Detectarea și tratarea valorilor extreme")

    numerical_cols = df.select_dtypes(include=['number']).columns

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        num_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        if num_outliers > 0:
            st.write(f"Coloana **{col}** are {num_outliers} valori extreme.")

            # Creare figuri pentru vizualizare
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Histogramă
            sns.histplot(df[col], bins=30, kde=True, ax=axes[0])
            axes[0].set_title(f"Histogramă pentru {col}")

            # Boxplot
            sns.boxplot(x=df[col], ax=axes[1])
            axes[1].set_title(f"Boxplot pentru {col}")

            # Density Plot
            sns.kdeplot(df[col], fill=True, ax=axes[2])
            axes[2].set_title(f"Density Plot pentru {col}")

            st.pyplot(fig)

            if st.button(f"Elimină valorile extreme pentru {col}"):
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                st.success(f"Valori extreme eliminate din {col}!")

    # Scatter Plot și Pair Plot pentru mai multe variabile
    if len(numerical_cols) > 1:
        st.subheader("Scatter Plot și Pair Plot")

        x_var = st.selectbox("Alege variabila X", numerical_cols)
        y_var = st.selectbox("Alege variabila Y", numerical_cols)

        if x_var and y_var:
            fig_scatter = plt.figure(figsize=(7, 5))
            sns.scatterplot(x=df[x_var], y=df[y_var])
            plt.title(f"Scatter Plot: {x_var} vs {y_var}")
            st.pyplot(fig_scatter)

        if st.button("Generează Pair Plot"):
            fig_pair = sns.pairplot(df[numerical_cols])
            st.pyplot(fig_pair)

with tab5:
    st.subheader("Codificarea datelor")

    df = st.session_state.df

    if "label_encoders" not in st.session_state:
        st.session_state.label_encoders = {}

    if st.button("Aplică Label Encoding pentru 'rating' și 'status'"):
        label_encoders = {}

        for col in ['rating', 'status']:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

        st.session_state.df = df
        st.session_state.label_encoders = label_encoders

        st.success("Label Encoding aplicat cu succes!")

        # Creare și afișare tabele de mapping
        for col, mapping in label_encoders.items():
            st.write(f"**Mapping pentru '{col}':**")
            mapping_df = pd.DataFrame(list(mapping.items()), columns=["Valoare Originală", "Valoare Codificată"])
            st.dataframe(mapping_df)

        st.write("**Primele 10 rânduri după Label Encoding:**")
        st.dataframe(df.head(10))

    if st.button("Aplică One-Hot Encoding pentru 'source'"):
        df = pd.get_dummies(df, columns=['source'], drop_first=True)
        st.session_state.df = df
        st.success("One-Hot Encoding aplicat pentru 'source'!")
        st.write("**Primele 10 rânduri după One-Hot Encoding:**")
        st.dataframe(df.head(10))


with tab6:
    st.subheader("Agregare si prelucrari statistice")
    df = st.session_state.df

    # Selectare coloane pentru agregare
    col1, col2, col3 = st.columns(3)

    with col1:
        group_column = st.selectbox(
            "Selectați coloana pentru grupare:",
            options=df.columns  # Limităm la coloane cu puține valori unice
        )

    with col2:
        numeric_column = st.selectbox(
            "Selectați coloana numerică pentru calcul:",
            options=df.select_dtypes(include=['int64', 'float64']).columns
        )

    with col3:
        agg_function = st.selectbox(
            "Selectați funcția de agregare:",
            options=['Medie', 'Mediană', 'Mod', 'Sumă', 'Minim', 'Maxim', 'Deviație standard', 'Număr de înregistrări']
        )

    # Definim funcțiile de agregare
    agg_functions = {
        'Medie': 'mean',
        'Mediană': 'median',
        'Mod': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'Sumă': 'sum',
        'Minim': 'min',
        'Maxim': 'max',
        'Deviație standard': 'std',
        'Număr de înregistrări': 'count'
    }

    # Buton pentru executare
    if st.button("Calculează statistici"):
        try:
            # Grupăm și calculăm statisticile
            if agg_function == 'Mod':
                result = df.groupby(group_column)[numeric_column].agg(agg_functions[agg_function]).reset_index()
            else:
                result = df.groupby(group_column)[numeric_column].agg(agg_functions[agg_function]).reset_index()

            result.columns = [group_column, f"{agg_function} a {numeric_column}"]

            # Sortare rezultate
            if agg_function in ['Medie', 'Mediană', 'Sumă', 'Maxim']:
                result = result.sort_values(f"{agg_function} a {numeric_column}", ascending=False)
            else:
                result = result.sort_values(f"{agg_function} a {numeric_column}", ascending=True)

            # Afișare rezultate
            st.write(f"Statistici {agg_function.lower()} pentru {numeric_column} grupate după {group_column}:")
            st.dataframe(result)

            # Vizualizare grafică
            fig, ax = plt.subplots(figsize=(10, 6))

            # Limităm la primele 20 de rezultate pentru vizualizare mai bună
            display_data = result.head(20)

            if agg_function == 'Număr de înregistrări':
                display_data.plot(kind='bar', x=group_column, y=f"{agg_function} a {numeric_column}", ax=ax,
                                  color='skyblue')
            else:
                display_data.plot(kind='bar', x=group_column, y=f"{agg_function} a {numeric_column}", ax=ax,
                                  color='salmon')

            ax.set_title(f"{agg_function} a {numeric_column} pe {group_column}")
            ax.set_xlabel(group_column)
            ax.set_ylabel(numeric_column)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"A apărut o eroare: {str(e)}")
            st.write("Sugestie: Asigurați-vă că ați selectat coloane valide pentru grupare și calcul.")

with tab7:
    df = st.session_state.df

    st.title("Functii Avansate de Grup")
    st.write("""
       Acest tab permite utilizarea diverselor functii de grup pentru analiza datelor.
       Selectati coloanele si operatiile dorite pentru a obține rezultate personalizate.
       """)

    # Divizare pe coloane pentru layout
    col1, col2 = st.columns(2)

    with col1:
        # Selectare coloane de grupare
        group_cols = st.multiselect(
            "Selectati coloanele de grupare:",
            options=df.columns,
            help="Puteti selecta una sau mai multe coloane pentru grupare"
        )

        # Selectare coloane numerice pentru agregare
        numeric_cols = st.multiselect(
            "Selectati coloanele numerice pentru calcul:",
            options=df.select_dtypes(include=['int64', 'float64']).columns,
            help="Selectati coloanele asupra carora sa se aplice operatiile"
        )

    with col2:
        # Selectare funcții de agregare
        agg_functions = st.multiselect(
            "Selectati functiile de agregare:",
            options=['sum', 'mean', 'median', 'min', 'max', 'count', 'std', 'var', 'first', 'last'],
            default=['mean'],
            help="Alegeti una sau mai multe functii de agregare"
        )

        # Opțiuni suplimentare
        sort_results = st.checkbox("Sortare rezultate", value=True)
        show_sample = st.checkbox("Afișează eșantion de date", value=True)

    # Buton de executare
    if st.button("Aplica functiile de grup"):
        if not group_cols or not numeric_cols or not agg_functions:
            st.warning("Selectati cel putin o coloana de grupare, o coloana numerică și o functie!")
        else:
            try:
                # Aplicare funcții de grup
                grouped = df.groupby(group_cols)[numeric_cols].agg(agg_functions)

                # Resetare index pentru afișare mai bună
                grouped = grouped.reset_index()

                # Flatten multi-level columns
                grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

                # Sortare rezultate dacă este selectat
                if sort_results and len(group_cols) == 1:
                    sort_col = grouped.columns[1]  # Prima coloană numerică
                    grouped = grouped.sort_values(sort_col, ascending=False)

                # Afișare rezultate
                st.subheader("Rezultate grupate")
                st.dataframe(grouped)

                # Descărcare rezultate
                csv = grouped.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descarca rezultatele",
                    data=csv,
                    file_name='grupare_rezultate.csv',
                    mime='text/csv'
                )

                # Afișare eșantion dacă este selectat
                if show_sample:
                    st.subheader("Esantion de date grupate")
                    st.write(grouped.head(10))

                # Vizualizare grafică pentru grupări simple
                if len(group_cols) == 1 and len(numeric_cols) == 1:
                    st.subheader("Vizualizare grafica")

                    # Selectare tip grafic
                    chart_type = st.selectbox(
                        "Tip grafic:",
                        options=['Bar', 'Line', 'Area', 'Pie'],
                        index=0
                    )

                    # Limitare la primele 20 de rezultate pentru vizualizare
                    plot_data = grouped.head(20)

                    fig, ax = plt.subplots(figsize=(10, 6))

                    if chart_type == 'Bar':
                        plot_data.plot.bar(
                            x=group_cols[0],
                            y=f"{numeric_cols[0]}_{agg_functions[0]}",
                            ax=ax,
                            color='skyblue'
                        )
                    elif chart_type == 'Line':
                        plot_data.plot.line(
                            x=group_cols[0],
                            y=f"{numeric_cols[0]}_{agg_functions[0]}",
                            ax=ax,
                            marker='o',
                            color='green'
                        )
                    elif chart_type == 'Area':
                        plot_data.plot.area(
                            x=group_cols[0],
                            y=f"{numeric_cols[0]}_{agg_functions[0]}",
                            ax=ax,
                            alpha=0.4,
                            color='purple'
                        )
                    elif chart_type == 'Pie':
                        plot_data.plot.pie(
                            y=f"{numeric_cols[0]}_{agg_functions[0]}",
                            labels=plot_data[group_cols[0]],
                            ax=ax,
                            autopct='%1.1f%%'
                        )

                    plt.title(f"{agg_functions[0].upper()} de {numeric_cols[0]} pe {group_cols[0]}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Eroare la grupare: {str(e)}")
                st.write("Sugestii:")
                st.write("- Asigurati-va ca coloanele de grupare nu contin prea multe valori unice")
                st.write("- Pentru coloanele non-numerice, folositi doar functii precum 'count', 'first', 'last'")

with tab8:
    st.title("Selecteaza tari pentru harta")

    gdf = gpd.read_file("ne_10m_admin_0_countries.zip")

    countries = gdf['NAME'].unique()
    selected_countries = st.multiselect("Alege tarile: ", sorted(countries))

    if selected_countries:
        filtered_gdf = gdf[gdf['NAME'].isin(selected_countries)]

        fig, ax = plt.subplots(figsize=(10,6))

        filtered_gdf.plot(ax=ax, color='lightgreen', edgecolor='black')
        st.pyplot(fig)
    else:
        st.write("Selecteaza cel putin o tara pentru a vedea harta")

with tab9:
    st.title("Analiza datelor")

    option = st.selectbox("Selecteaza tipul de analiza a datelor", ["Clusterizare", "Regresie logistica", "Regresie multipla"])

    if option == "Clusterizare":
        df = st.session_state.df

        missing_values = df.isnull().sum()
        st.write("missing values")
        st.write(missing_values)

        st.write("O sa eleminam pentru clusterizare anumite coloane care contin date de tip sir de caractere pentru a face mai usoara aplicarea algoritmului pentru cluster.")
        st.write("Coloanele pe care le vom elimina din setul de date sunt: title, airing, aired_string si studio.")
        st.write("Pentru restul coloanelor de tip sir de caractere vom aplica Label Encoding.")

        clusterizareDf = df.copy()
        clusterizareDf.drop(['title'], axis=1, inplace=True)
        clusterizareDf.drop(['studio'], axis=1, inplace=True)
        clusterizareDf.drop(['airing'], axis=1, inplace=True)
        clusterizareDf.drop(['aired_string'], axis=1, inplace=True)
        st.write(
            "Coloana genre o vom trata usor mai diferit deoarece un anime aparatine mai multor genuri, asa ca vom lua genul principal si pentru acela aplica label encoding, iar coloana genre o vom sterge")
        clusterizareDf['first_genre'] = df['genre'].apply(lambda x: x.split(',')[0].strip() if pd.notnull(x) else '')
        clusterizareDf.drop(['genre'], axis=1, inplace=True)

        label_encoders = {}

        coloane_categoricale = ['type', 'rating', 'status', 'source', 'first_genre']
        for col in coloane_categoricale:
            if col in clusterizareDf.columns:
                le = LabelEncoder()
                clusterizareDf[col] = le.fit_transform(clusterizareDf[col].astype(str))
                label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

        st.write("De asemenea vom da drop la coloana anime_id si rank")
        clusterizareDf.drop(['anime_id'], axis=1, inplace=True)
        clusterizareDf.drop(['rank'], axis=1, inplace=True)


        st.write("Cum arata datele in dupa stergerea coloanelor si aplicarea label encoding")
        st.dataframe(clusterizareDf.head())

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        date_scaled = pd.DataFrame(scaler.fit_transform(clusterizareDf))

        fig, ax = plt.subplots()
        sns.boxplot(data=date_scaled, ax=ax)
        ax.set_title("Boxplot pentru detectarea outlierelor")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        correlation_matrix = date_scaled.corr()
        sns.heatmap(correlation_matrix, annot=True, ax=ax2)
        ax2.set_title('Heatmap pentru analiza corelatiei')
        st.pyplot(fig2)

        st.subheader("Selecteaza coloanele pentru care vrei sa calculezi numarul de clustere si pentru care o sa evaluam modelul")
        selected_columns = st.multiselect("Alege coloanele", clusterizareDf.columns, default=clusterizareDf.columns)

        X = clusterizareDf[selected_columns].values

        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
        ax.set_title('The Elbow Method')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)

        # Fitting K-Means to the dataset
        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(X)

        # Visualising the clusters
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.scatterplot(x=X[y_kmeans == 0, 0], y=X[y_kmeans == 0, 1], color='yellow', label='Cluster 1', s=50, ax=ax)
        sns.scatterplot(x=X[y_kmeans == 1, 0], y=X[y_kmeans == 1, 1], color='blue', label='Cluster 2', s=50, ax=ax)
        sns.scatterplot(x=X[y_kmeans == 2, 0], y=X[y_kmeans == 2, 1], color='green', label='Cluster 3', s=50, ax=ax)
        sns.scatterplot(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], color='red',
                        label='Centroids', s=300, marker=',', ax=ax)

        ax.set_title('Clusters of customers')
        ax.set_xlabel(selected_columns[0])
        ax.set_ylabel(selected_columns[1])
        ax.legend()
        ax.grid(False)
        st.pyplot(fig)

        # Evaluarea modelului K-Means
        # -----------------------------------------------------------

        from sklearn.metrics import silhouette_score, roc_curve, auc

        # Silhouette Score
        """
        Măsoară cât de apropiat este un punct de clusterul său comparativ cu celelalte clustere.
        Scorul este între -1 și 1:

        ~1 → punctul este bine încadrat

        ~0 → este la graniță între clustere

        < 0 → probabil este pus greșit în cluster"""

        best_k = 3
        best_score = -1

        silhouette_scores = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            preds = kmeans.fit_predict(X)
            score = silhouette_score(X, preds)
            silhouette_scores.append(score)

            if score > best_score:
                best_score = score
                best_k = k

        st.write(f"**Numarul optim de clustere (Silhouette):** `{best_k}` (scor: `{best_score:.4f}`)")

        sil_score = silhouette_score(X, y_kmeans)
        st.subheader("Evaluare Model")
        st.write(f"**Silhouette Score pentru {best_k} clustere:** `{sil_score:.4f}`")

        # Interpretare:
        # +1 = clustere bine definite, 0 = la limită, <0 = atribuire greșită

        st.write("**Silhouette Scores pentru diferite valori ale lui k:**")
        for k in range(2, 11):
            km = KMeans(n_clusters=k, init='k-means++', random_state=42)
            preds = km.fit_predict(X)
            score = silhouette_score(X, preds)
            st.write(f'k = {k} --> silhouette score = {score:.4f}')
    elif option == "Regresie logistica":
        df_log = st.session_state.df.copy()

        st.write("## Regresie Logistica - Clasificarea anime-urilor ca 'bune' sau 'slabe'")

        df_log.drop(['anime_id', 'title', 'rank', 'aired_string', 'airing'], axis=1, inplace=True)

        st.write("Variabila tinta binara pe care o sa o creeam pentru a determina daca un anime este bun sau nu este score.")
        st.write("Daca score >= 7.5 anime-ul este bun, iar daca score < 7.5 anime-ul este slab")
        df_log['target'] = df_log['score'].apply(lambda x: 1 if x >= 7.5 else 0)
        df_log.drop(['score'], axis=1, inplace=True)

        df_log['first_genre'] = df_log['genre'].apply(lambda x: x.split(',')[0].strip() if pd.notnull(x) else '')
        df_log.drop(['genre'], axis=1, inplace=True)

        categorical_cols = ['type', 'source', 'rating', 'status', 'studio', 'first_genre']
        df_log = pd.get_dummies(df_log, columns=categorical_cols, drop_first=True)

        df_log.fillna(df_log.median(numeric_only=True), inplace=True)

        st.write("Dimensiunea finala a setului de date pentru regresie logistica:", df_log.shape)

        # Separare X și y
        X = df_log.drop(['score', 'target'], axis=1, errors='ignore')
        y = df_log['target']

        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write(f"Dimensiuni seturi de date: Antrenare - {X_train.shape}, Test - {X_test.shape}")

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        st.subheader("Performanta modelului")
        st.write(f"**Acuratete:** {accuracy_score(y_test, y_pred):.4f}")
        st.text("Matricea de Confuzie")
        st.write(confusion_matrix(y_test, y_pred))
        st.text("Raport de Clasificare")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Coeficientii modelului")
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_[0]
        }).sort_values('Coefficient', ascending=False)

        st.dataframe(coef_df)

        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        st.subheader("Curba ROC și AUC")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)

        st.write(f"**AUC (Area Under Curve):** `{roc_auc:.4f}`")

    elif option == "Regresie multipla":
        df_regresie_multipla = st.session_state.df.copy()
        df_regresie_multipla.drop(['title', 'rank', 'anime_id', 'airing', 'aired_score'], axis=1, inplace=True)

        st.write("**Structura datelor pentru regresie**")
        st.write(f"Datele contin {df_regresie_multipla.shape[0]} observatii si {df_regresie_multipla.shape[1]} coloane dupa eliminarea celor nedorite")

        missing_values = df.isnull().sum()
        st.write("Valori lipsa")
        st.write(missing_values)

        df_regresie_multipla['first_genre'] = df['genre'].apply(lambda x: x.split(',')[0].strip() if pd.notnull(x) else '')
        df_regresie_multipla.drop(['genre'], axis=1, inplace=True)

        st.write("**Codificare variabile categorice**")
        st.write("Selectam anumite coloane categorice relevante pentru care o sa aplicam One-Hot Encoding")

        categorical_cols = ['first_genre', 'rating', 'studio', 'source', 'type']
        df_regresie_multipla = pd.get_dummies(df_regresie_multipla, columns=categorical_cols, drop_first=True)

        st.write("Dimensiunea datelor după codificare:", df_regresie_multipla.shape)
        st.write("Exemplu de coloane create:", list(df_regresie_multipla.columns)[-10:])

        st.write("**Selectare variabile pentru model**")
        st.write("""
               **Variabila dependenta (target):** 'score' - scorul IMDb/MAL al anime-ului  
               **Variabile independente (predictori):** caracteristici numerice și cele codificate
               """)

        exclude_cols = ['anime_id', 'title_english', 'aired_string', 'status', 'airing']
        X = df_regresie_multipla.drop(['score'] + exclude_cols, axis=1, errors='ignore')
        y = df_regresie_multipla['score']

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write(f"Dimensiuni seturi de date: Antrenare - {X_train.shape}, Test - {X_test.shape}")

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("**Metricile modelului:**")
        st.write(f"- Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"- R-squared (R²): {r2:.4f}")

        st.subheader("Coeficientii modelului")
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', ascending=False)

        st.dataframe(coef_df)

        st.subheader("Predictii vs Valori Reale")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_xlabel('Scor Real')
        ax.set_ylabel('Scor Prezis')
        ax.set_title('Comparare intre scorurile reale si cele prezise')
        st.pyplot(fig)

        st.subheader("Analiza reziduurilor")
        residuals = y_test - y_pred
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        sns.histplot(residuals, kde=True, ax=ax1)
        ax1.set_title('Distribuția reziduurilor')

        sns.scatterplot(x=y_pred, y=residuals, ax=ax2)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_title('Reziduuri vs Predicții')

        st.pyplot(fig)
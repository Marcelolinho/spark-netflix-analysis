from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    udf, col, explode, split, lower, regexp_replace, trim, 
    when, sum as spark_sum, array_contains, size, filter as spark_filter
)
from pyspark.sql.functions import min as spark_min, max as spark_max, round as spark_round
from pyspark.sql.types import StringType, IntegerType, ArrayType
from nltk.corpus import stopwords
import nltk
from collections import Counter

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

spark = SparkSession.builder \
    .appName("NetflixAnalysis") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

netflix_df = spark.read.csv(
    "./dataset/netflix_titles.csv",
    header=True,
    inferSchema=True
)

amazon_prime_df = spark.read.csv(
    "./dataset/amazon_prime_titles.csv",
    header=True,
    inferSchema=True
)

disney_plus_df = spark.read.csv(
    "./dataset/disney_plus_titles.csv",
    header=True,
    inferSchema=True
)

netflix_series_df = netflix_df.filter(col("type") == "TV Show")
netflix_movies_df = netflix_df.filter(col("type") == "Movie")

amazon_series_df = amazon_prime_df.filter(col("type") == "TV Show")
amazon_movies_df = amazon_prime_df.filter(col("type") == "Movie")

disney_series_df = disney_plus_df.filter(col("type") == "TV Show")
disney_movies_df = disney_plus_df.filter(col("type") == "Movie")

def remove_stopwords(text):
    """Remove stopwords e retorna string com palavras significativas"""
    if text is None or text == "":
        return ""
    # Converter para minúsculas, remover pontuação
    text = text.lower()
    words = text.split()
    # Filtrar stopwords e palavras muito curtas
    filtered_words = [word for word in words 
                     if word not in stop_words and len(word) > 2 
                     and word.isalpha()]
    return " ".join(filtered_words)

remove_stopwords_udf = udf(remove_stopwords, StringType())

netflix_series_df = netflix_series_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))
netflix_movies_df = netflix_movies_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))

amazon_movies_df = amazon_movies_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))
amazon_series_df = amazon_series_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))

disney_movies_df = disney_movies_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))
disney_series_df = disney_series_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))



# Cache dos DataFrames para reutilização
netflix_series_df.cache()
netflix_movies_df.cache()
amazon_series_df.cache()
amazon_movies_df.cache()
disney_series_df.cache()
disney_movies_df.cache()



def get_top_words(df, column_name, top_n=10):
    """Extrai as top N palavras mais frequentes de uma coluna"""
    # Explodir palavras e contar frequência
    words_df = df.select(explode(split(col(column_name), " ")).alias("word")) \
        .filter(col("word") != "") \
        .groupBy("word") \
        .count() \
        .orderBy(col("count").desc()) \
        .limit(top_n)
    
    return words_df.collect()

netflix_series_top_desc_words = get_top_words(netflix_series_df, "description_scraped", 10)
netflix_movies_top_desc_words = get_top_words(netflix_movies_df, "description_scraped", 10)

amazon_movies_top_desc_words = get_top_words(amazon_movies_df, "description_scraped", 10)
amazon_series_top_desc_words = get_top_words(amazon_series_df, "description_scraped", 10)

disney_movies_top_desc_words = get_top_words(disney_movies_df, "description_scraped", 10)
disney_series_top_desc_words = get_top_words(disney_series_df, "description_scraped", 10)

def get_top_genres(df, top_n=5):
    """Extrai os top N gêneros mais frequentes"""
    genres_df = df.select(explode(split(col("listed_in"), ",")).alias("genre")) \
        .withColumn("genre", trim(col("genre"))) \
        .filter(col("genre") != "") \
        .groupBy("genre") \
        .count() \
        .orderBy(col("count").desc()) \
        .limit(top_n)
    
    return genres_df.collect()

netflix_series_top_genres = get_top_genres(netflix_series_df, 5)
netflix_movies_top_genres = get_top_genres(netflix_movies_df, 5)

amazon_series_top_genres = get_top_genres(amazon_series_df, 5)
amazon_movies_top_genres = get_top_genres(amazon_movies_df, 5)

disney_series_top_genres = get_top_genres(disney_series_df, 5)
disney_movies_top_genres = get_top_genres(disney_movies_df, 5)

def get_top_title_words(df, top_n=10):
    """Extrai as top N palavras mais frequentes dos títulos"""
    title_words_df = df.select(
        explode(split(lower(regexp_replace(col("title"), "[^a-zA-Z\\s]", "")), " ")).alias("word")
    ).filter(col("word") != "") \
     .filter(~col("word").isin(list(stop_words))) \
     .filter(col("word").cast("string").isNotNull()) \
     .filter(col("word") != "") \
     .groupBy("word") \
     .count() \
     .orderBy(col("count").desc()) \
     .limit(top_n)
    
    return title_words_df.collect()

netflix_series_top_title_words = get_top_title_words(netflix_series_df, 10)
netflix_movies_top_title_words = get_top_title_words(netflix_movies_df, 10)

amazon_series_top_title_words = get_top_title_words(amazon_series_df, 10)
amazon_movies_top_title_words = get_top_title_words(amazon_movies_df, 10)

disney_series_top_title_words = get_top_title_words(disney_series_df, 10)
disney_movies_top_title_words = get_top_title_words(disney_movies_df, 10)

def calculate_scores(df, top_desc_words, top_genres, top_title_words):
    """Calcula pontuação baseada nos 4 critérios"""
    desc_word_scores = {row['word']: (10 - idx) for idx, row in enumerate(top_desc_words)}
    genre_scores = {row['genre']: (5 - idx) * 5 for idx, row in enumerate(top_genres)}
    title_word_scores = {row['word']: (10 - idx) for idx, row in enumerate(top_title_words)}

    def score_description(desc_text):
        if desc_text is None or desc_text == "":
            return 0
        words = desc_text.split()
        score = sum(desc_word_scores.get(word, 0) for word in words)
        return int(score)
    
    score_description_udf = udf(score_description, IntegerType())
    
    def score_genres(genres_text):
        if genres_text is None or genres_text == "":
            return 0
        genres = [g.strip() for g in genres_text.split(",")]
        score = sum(genre_scores.get(genre, 0) for genre in genres)
        return int(score)
    
    score_genres_udf = udf(score_genres, IntegerType())
    
    def score_year(year):
        if year is None:
            return 0
        try:
            year = int(year)
            if 2000 <= year <= 2005:
                return 5
            elif 2006 <= year <= 2010:
                return 10
            elif 2011 <= year <= 2015:
                return 15
            elif 2016 <= year <= 2021:
                return 20
            else:
                return 0
        except:
            return 0
    
    score_year_udf = udf(score_year, IntegerType())
    
    def score_title(title_text):
        if title_text is None or title_text == "":
            return 0
        title_clean = title_text.lower()
        words = [w for w in title_clean.split() if w.isalpha() and w not in stop_words]
        score = sum(title_word_scores.get(word, 0) for word in words)
        return int(score)
    
    score_title_udf = udf(score_title, IntegerType())
    
    df = df.withColumn("points_criterion_1", score_description_udf(col("description_scraped")))
    df = df.withColumn("points_criterion_2", score_genres_udf(col("listed_in")))
    df = df.withColumn("points_criterion_3", score_year_udf(col("release_year")))
    df = df.withColumn("points_criterion_4", score_title_udf(col("title")))
    
    df = df.withColumn(
        "total_points",
        col("points_criterion_1") + 
        col("points_criterion_2") + 
        col("points_criterion_3") + 
        col("points_criterion_4")
    )
    
    # Calcular min e max pontuação para normalização
    
    stats = df.agg(
        spark_min("total_points").alias("min_points"),
        spark_max("total_points").alias("max_points")
    ).collect()[0]
    
    min_points = stats["min_points"]
    max_points = stats["max_points"]
    
    # Adicionar coluna de sucesso (60-95% usando Min-Max Scaling)
    if max_points == min_points:
        # Se todos têm a mesma pontuação, atribuir 77.5% (média de 60 e 95)
        df = df.withColumn("sucesso", spark_round(col("total_points") * 0 + 77.5, 2))
    else:
        # sucesso = 60 + ((pontuacao - min_pontuacao) / (max_pontuacao - min_pontuacao)) * 35
        df = df.withColumn(
            "sucesso",
            spark_round(
                60 + ((col("total_points") - min_points) / (max_points - min_points)) * 35,
                2
            )
        )
    
    return df

# Calculando tudo dos datasets

netflix_series_scored = calculate_scores(netflix_series_df, netflix_series_top_desc_words, netflix_series_top_genres, netflix_series_top_title_words)
netflix_movies_scored = calculate_scores(netflix_movies_df, netflix_movies_top_desc_words, netflix_movies_top_genres, netflix_movies_top_title_words)

amazon_series_scored = calculate_scores(amazon_series_df, amazon_series_top_desc_words, amazon_series_top_genres, amazon_series_top_title_words)
amazon_movies_scored = calculate_scores(amazon_movies_df, amazon_movies_top_desc_words, amazon_movies_top_genres, amazon_movies_top_title_words)

disney_series_scored = calculate_scores(disney_series_df, disney_series_top_desc_words, disney_series_top_genres, disney_series_top_title_words)
disney_movies_scored = calculate_scores(disney_movies_df, disney_movies_top_desc_words, disney_movies_top_genres, disney_movies_top_title_words)

# Selecionando os top 15 para a IA criar depois

netflix_top_15_series = netflix_series_scored.orderBy(col("total_points").desc()).limit(15)
netflix_top_15_movies = netflix_movies_scored.orderBy(col("total_points").desc()).limit(15)

amazon_top_15_series = amazon_series_scored.orderBy(col("total_points").desc()).limit(15)
amazon_top_15_movies = amazon_movies_scored.orderBy(col("total_points").desc()).limit(15)

disney_top_15_series = disney_series_scored.orderBy(col("total_points").desc()).limit(15)
disney_top_15_movies = disney_movies_scored.orderBy(col("total_points").desc()).limit(15)

netflix_series_to_save = netflix_top_15_series.select(
    "title", "listed_in", "release_year", "rating", "description",
    "points_criterion_1", "points_criterion_2", 
    "points_criterion_3", "points_criterion_4", "total_points", "sucesso"
)

netflix_movies_to_save = netflix_top_15_movies.select(
    "title", "listed_in", "release_year", "rating", "description",
    "points_criterion_1", "points_criterion_2", 
    "points_criterion_3", "points_criterion_4", "total_points", "sucesso"
)

amazon_movies_to_save = amazon_top_15_movies.select(
    "title", "listed_in", "release_year", "rating", "description",
    "points_criterion_1", "points_criterion_2", 
    "points_criterion_3", "points_criterion_4", "total_points", "sucesso"
)

amazon_series_tos_save = amazon_top_15_series.select(
    "title", "listed_in", "release_year", "rating", "description",
    "points_criterion_1", "points_criterion_2", 
    "points_criterion_3", "points_criterion_4", "total_points", "sucesso"
)

disney_series_to_save = disney_top_15_series.select(
    "title", "listed_in", "release_year", "rating", "description",
    "points_criterion_1", "points_criterion_2", 
    "points_criterion_3", "points_criterion_4", "total_points", "sucesso"
)

disney_movies_to_save = disney_top_15_movies.select(
    "title", "listed_in", "release_year", "rating", "description",
    "points_criterion_1", "points_criterion_2", 
    "points_criterion_3", "points_criterion_4", "total_points", "sucesso"
)

netflix_series_to_save.coalesce(1).write.mode("overwrite").option("header", "true").csv("./output/netflix_top_15_series_temp")
netflix_movies_to_save.coalesce(1).write.mode("overwrite").option("header", "true").csv("./output/netflix_top_15_movies_temp")

amazon_series_tos_save.coalesce(1).write.mode("overwrite").option("header", "true").csv("./output/amazon_top_15_series_temp")
amazon_movies_to_save.coalesce(1).write.mode("overwrite").option("header", "true").csv("./output/amazon_top_15_movies_temp")

disney_series_to_save.coalesce(1).write.mode("overwrite").option("header", "true").csv("./output/disney_top_15_series_temp")
disney_movies_to_save.coalesce(1).write.mode("overwrite").option("header", "true").csv("./output/disney_top_15_movies_temp")


# netflix_series_results = netflix_series_to_save.collect()
# netflix_movies_results = netflix_movies_to_save.collect()

# amazon_series_results = amazon_series_tos_save.collect()
# amazon_movies_results = amazon_movies_to_save.collect()

# disney_series_results = disney_series_to_save.collect()
# disney_movies_results = disney_movies_to_save.collect()

import os
import shutil

def move_csv_file(temp_dir, final_name):
    """Move o arquivo CSV da pasta temporária para o nome final"""
    if os.path.exists(temp_dir):
        csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
        if csv_files:
            temp_file = os.path.join(temp_dir, csv_files[0])
            shutil.move(temp_file, final_name)
            shutil.rmtree(temp_dir)
            print(f"✓ {final_name} criado com sucesso!")

move_csv_file("./output/netflix_top_15_series_temp", "./analysis/netflix_top_15_series.csv")
move_csv_file("./output/netflix_top_15_movies_temp", "./analysis/netflix_top_15_movies.csv")

move_csv_file("./output/amazon_top_15_series_temp", "./analysis/amazon_top_15_series.csv")
move_csv_file("./output/amazon_top_15_movies_temp", "./analysis/amazon_top_15_movies.csv")

move_csv_file("./output/disney_top_15_series_temp", "./analysis/disney_top_15_series.csv")
move_csv_file("./output/disney_top_15_movies_temp", "./analysis/disney_top_15_movies.csv")


# Este código foi gerado depois de rodar a primeira vez o script acima. Essa é a criação das séries e filmes que foi feita manualmente pois não tenho Tokens para utilizar uma IA generativa.

amazon_created_series = {
    "title": "Modern Genius Girl",
    "listed_in": "Comedy, Drama",
    "release_year": "2020",
    "rating": "ALL",
    "description": "Modern Genius Girl follows Maya Torres, a 14-year-old prodigy who creates an emotional-connection app. Suddenly famous, she faces jealousy and family secrets. When she discovers a mysterious scientific recipe book, Maya and her quirky friends embark on tech-magical adventures that test their hearts and minds. A lighthearted story about friendship, diversity, and the power of genius with kindness."
}

amazon_created_movie = {
    "title": "Summer Love",
    "listed_in": "Comedy, Drama",
    "release_year": "2019",
    "rating": "13+",
    "description": "Summer Love is a coming-of-age dramedy about Emma, a spirited 17-year-old whose summer plans fall apart after a family secret shatters her trust. Escaping to her grandmother’s beach house, she meets Alex, a charming musician hiding heartbreak of his own. Between laughter, late-night talks, and unexpected twists, Emma learns that love isn’t about perfection but the courage to forgive, to start over, and to believe in happiness again. A heartfelt story of second chances under the summer sun."
}

disney_created_movie = {
    "title": "Story of Frozen Terror",
    "listed_in": "Animation, Comedy",
    "release_year": "2017",
    "rating": "TV-G",
    "description": "Story of The Frozen Terror follows a group of unlikely heroes Frosty, a clumsy snow creature, and Lumi, a talking lantern who must save their icy village after a mysterious freeze begins turning everything to stone. Along the way, they meet quirky allies and face hilarious challenges as they uncover the secret behind the frozen terror. Blending laughter, heart, and a touch of Disney magic, this animated comedy reminds us that even in the coldest times, friendship can melt any fear."
}

disney_created_series = {
    "title": "Legend of the Fast Owl",
    "listed_in": "Action-Adventure, Animation",
    "release_year": "2016",
    "rating": "TV-Y7",
    "description": "Legend of the Fast Owl follows a young owl named Oliver who dreams of becoming the fastest flyer in the forest. With the help of his friends, he embarks on an epic adventure to compete in the Great Forest Race. Along the way, they encounter challenges that test their courage and friendship. This animated series is a heartwarming tale of perseverance, teamwork, and believing in oneself."
}

netflix_created_movie = {
    "title": "The Perfect Statue",
    "listed_in": "Dramas, International Movies",
    "release_year": "2021",
    "rating": "TV-14",
    "description": "After finding an ancient statue connected to his family, a young statue restorer goes on a journey that mixes love, greed and redemption. Alongside a young archeologist he finds out the statue hides secrets capable of changing the destiny of them both. Nevertheless, the closer they get to the truth, the further they drift apart."
}

netflix_created_series = {
    "title": "What Happens After an Accident",
    "listed_in": "International TV Shows, TV dramas",
    "release_year": "2021",
    "rating": "TV-14",
    "description": "A date arranged through a dating app turns into tragedy, forcing a single mother to fight for the son who survived by a miracle."
}

# Converter dicionários para DataFrames e calcular sucesso usando calculate_scores
# Esta parte ficou bem feia, porém como o que foi feito para todas a 6 variáveis é o mesmo, deixei assim para não me orientar melhor

amazon_series_custom_df = spark.createDataFrame([amazon_created_series])
amazon_series_custom_df = amazon_series_custom_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))
amazon_series_combined = amazon_series_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped").union(
    amazon_series_custom_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped")
)
amazon_series_combined_scored = calculate_scores(amazon_series_combined, amazon_series_top_desc_words, amazon_series_top_genres, amazon_series_top_title_words)
amazon_series_custom_scored = amazon_series_combined_scored.filter(col("title") == amazon_created_series["title"])

amazon_movie_custom_df = spark.createDataFrame([amazon_created_movie])
amazon_movie_custom_df = amazon_movie_custom_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))
amazon_movie_combined = amazon_movies_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped").union(
    amazon_movie_custom_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped")
)
amazon_movie_combined_scored = calculate_scores(amazon_movie_combined, amazon_movies_top_desc_words, amazon_movies_top_genres, amazon_movies_top_title_words)
amazon_movie_custom_scored = amazon_movie_combined_scored.filter(col("title") == amazon_created_movie["title"])

disney_movie_custom_df = spark.createDataFrame([disney_created_movie])
disney_movie_custom_df = disney_movie_custom_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))
disney_movie_combined = disney_movies_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped").union(
    disney_movie_custom_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped")
)
disney_movie_combined_scored = calculate_scores(disney_movie_combined, disney_movies_top_desc_words, disney_movies_top_genres, disney_movies_top_title_words)
disney_movie_custom_scored = disney_movie_combined_scored.filter(col("title") == disney_created_movie["title"])

disney_series_custom_df = spark.createDataFrame([disney_created_series])
disney_series_custom_df = disney_series_custom_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))
disney_series_combined = disney_series_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped").union(
    disney_series_custom_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped")
)
disney_series_combined_scored = calculate_scores(disney_series_combined, disney_series_top_desc_words, disney_series_top_genres, disney_series_top_title_words)
disney_series_custom_scored = disney_series_combined_scored.filter(col("title") == disney_created_series["title"])

netflix_movie_custom_df = spark.createDataFrame([netflix_created_movie])
netflix_movie_custom_df = netflix_movie_custom_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))
netflix_movie_combined = netflix_movies_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped").union(
    netflix_movie_custom_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped")
)
netflix_movie_combined_scored = calculate_scores(netflix_movie_combined, netflix_movies_top_desc_words, netflix_movies_top_genres, netflix_movies_top_title_words)
netflix_movie_custom_scored = netflix_movie_combined_scored.filter(col("title") == netflix_created_movie["title"])

netflix_series_custom_df = spark.createDataFrame([netflix_created_series])
netflix_series_custom_df = netflix_series_custom_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))
netflix_series_combined = netflix_series_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped").union(
    netflix_series_custom_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped")
)
netflix_series_combined_scored = calculate_scores(netflix_series_combined, netflix_series_top_desc_words, netflix_series_top_genres, netflix_series_top_title_words)
netflix_series_custom_scored = netflix_series_combined_scored.filter(col("title") == netflix_created_series["title"])

amazon_movie_custom_df = spark.createDataFrame([amazon_created_movie])
amazon_movie_custom_df = amazon_movie_custom_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))
amazon_movie_combined = amazon_movies_scored.select("title", "listed_in", "release_year", "rating", "description", "description_scraped").union(
    amazon_movie_custom_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped")
)
amazon_movie_combined_scored = calculate_scores(amazon_movie_combined, amazon_movies_top_desc_words, amazon_movies_top_genres, amazon_movies_top_title_words)
amazon_movie_custom_scored = amazon_movie_combined_scored.filter(col("title") == amazon_created_movie["title"])

disney_movie_custom_df = spark.createDataFrame([disney_created_movie])
disney_movie_custom_df = disney_movie_custom_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))
disney_movie_combined = disney_movies_scored.select("title", "listed_in", "release_year", "rating", "description", "description_scraped").union(
    disney_movie_custom_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped")
)
disney_movie_combined_scored = calculate_scores(disney_movie_combined, disney_movies_top_desc_words, disney_movies_top_genres, disney_movies_top_title_words)
disney_movie_custom_scored = disney_movie_combined_scored.filter(col("title") == disney_created_movie["title"])

disney_series_custom_df = spark.createDataFrame([disney_created_series])
disney_series_custom_df = disney_series_custom_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))
disney_series_combined = disney_series_scored.select("title", "listed_in", "release_year", "rating", "description", "description_scraped").union(
    disney_series_custom_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped")
)
disney_series_combined_scored = calculate_scores(disney_series_combined, disney_series_top_desc_words, disney_series_top_genres, disney_series_top_title_words)
disney_series_custom_scored = disney_series_combined_scored.filter(col("title") == disney_created_series["title"])

netflix_movie_custom_df = spark.createDataFrame([netflix_created_movie])
netflix_movie_custom_df = netflix_movie_custom_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))
netflix_movie_combined = netflix_movies_scored.select("title", "listed_in", "release_year", "rating", "description", "description_scraped").union(
    netflix_movie_custom_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped")
)
netflix_movie_combined_scored = calculate_scores(netflix_movie_combined, netflix_movies_top_desc_words, netflix_movies_top_genres, netflix_movies_top_title_words)
netflix_movie_custom_scored = netflix_movie_combined_scored.filter(col("title") == netflix_created_movie["title"])

netflix_series_custom_df = spark.createDataFrame([netflix_created_series])
netflix_series_custom_df = netflix_series_custom_df.withColumn("description_scraped", remove_stopwords_udf(col("description")))
netflix_series_combined = netflix_series_scored.select("title", "listed_in", "release_year", "rating", "description", "description_scraped").union(
    netflix_series_custom_df.select("title", "listed_in", "release_year", "rating", "description", "description_scraped")
)
netflix_series_combined_scored = calculate_scores(netflix_series_combined, netflix_series_top_desc_words, netflix_series_top_genres, netflix_series_top_title_words)
netflix_series_custom_scored = netflix_series_combined_scored.filter(col("title") == netflix_created_series["title"])

result_amazon_series = amazon_series_custom_scored.select("title", "sucesso").collect()[0]
result_amazon_movie = amazon_movie_custom_scored.select("title", "sucesso").collect()[0]
result_disney_movie = disney_movie_custom_scored.select("title", "sucesso").collect()[0]
result_disney_series = disney_series_custom_scored.select("title", "sucesso").collect()[0]
result_netflix_movie = netflix_movie_custom_scored.select("title", "sucesso").collect()[0]
result_netflix_series = netflix_series_custom_scored.select("title", "sucesso").collect()[0]

# Print final para ver a porcentagem de sucesso calculada para cada Série/Filme gerado
print(f"{result_amazon_series['title']}: {result_amazon_series['sucesso']}%")
print(f"{result_amazon_movie['title']}: {result_amazon_movie['sucesso']}%")
print(f"{result_disney_movie['title']}: {result_disney_movie['sucesso']}%")
print(f"{result_disney_series['title']}: {result_disney_series['sucesso']}%")
print(f"{result_netflix_movie['title']}: {result_netflix_movie['sucesso']}%")
print(f"{result_netflix_series['title']}: {result_netflix_series['sucesso']}%")

# Fechar SparkSession
spark.stop()
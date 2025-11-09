from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    udf, col, explode, split, lower, regexp_replace, trim, 
    when, sum as spark_sum, array_contains, size, filter as spark_filter
)
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
    "points_criterion_3", "points_criterion_4", "total_points"
)

netflix_movies_to_save = netflix_top_15_movies.select(
    "title", "listed_in", "release_year", "rating", "description",
    "points_criterion_1", "points_criterion_2", 
    "points_criterion_3", "points_criterion_4", "total_points"
)

amazon_movies_to_save = amazon_top_15_movies.select(
    "title", "listed_in", "release_year", "rating", "description",
    "points_criterion_1", "points_criterion_2", 
    "points_criterion_3", "points_criterion_4", "total_points"
)

amazon_series_tos_save = amazon_top_15_series.select(
    "title", "listed_in", "release_year", "rating", "description",
    "points_criterion_1", "points_criterion_2", 
    "points_criterion_3", "points_criterion_4", "total_points"
)

disney_series_to_save = disney_top_15_series.select(
    "title", "listed_in", "release_year", "rating", "description",
    "points_criterion_1", "points_criterion_2", 
    "points_criterion_3", "points_criterion_4", "total_points"
)

disney_movies_to_save = disney_top_15_movies.select(
    "title", "listed_in", "release_year", "rating", "description",
    "points_criterion_1", "points_criterion_2", 
    "points_criterion_3", "points_criterion_4", "total_points"
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

# Fechar SparkSession
spark.stop()
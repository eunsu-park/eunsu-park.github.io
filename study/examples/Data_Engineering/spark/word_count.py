"""
PySpark Word Count 예제

Spark의 기본 동작을 보여주는 클래식한 Word Count 예제입니다.

실행:
  spark-submit word_count.py
  또는
  python word_count.py (로컬 모드)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, lower, regexp_replace, col, desc


def word_count_rdd(spark, text_data):
    """RDD API를 사용한 Word Count"""
    print("=" * 50)
    print("RDD API Word Count")
    print("=" * 50)

    sc = spark.sparkContext

    # RDD 생성
    lines_rdd = sc.parallelize(text_data)

    # Word Count 처리
    word_counts = (
        lines_rdd
        .flatMap(lambda line: line.lower().split())  # 단어로 분리
        .map(lambda word: (word, 1))                  # (word, 1) 쌍 생성
        .reduceByKey(lambda a, b: a + b)              # 단어별 합계
        .sortBy(lambda x: x[1], ascending=False)      # 빈도순 정렬
    )

    # 결과 출력
    print("\nTop 10 words (RDD):")
    for word, count in word_counts.take(10):
        print(f"  {word}: {count}")

    return word_counts


def word_count_df(spark, text_data):
    """DataFrame API를 사용한 Word Count"""
    print("\n" + "=" * 50)
    print("DataFrame API Word Count")
    print("=" * 50)

    # DataFrame 생성
    df = spark.createDataFrame([(line,) for line in text_data], ["line"])

    # Word Count 처리
    word_counts = (
        df
        .select(explode(split(lower(col("line")), r"\s+")).alias("word"))
        # 특수문자 제거
        .withColumn("word", regexp_replace(col("word"), r"[^a-z0-9]", ""))
        # 빈 문자열 제외
        .filter(col("word") != "")
        # 그룹화 및 카운트
        .groupBy("word")
        .count()
        # 정렬
        .orderBy(desc("count"))
    )

    print("\nTop 10 words (DataFrame):")
    word_counts.show(10, truncate=False)

    return word_counts


def word_count_sql(spark, text_data):
    """Spark SQL을 사용한 Word Count"""
    print("\n" + "=" * 50)
    print("Spark SQL Word Count")
    print("=" * 50)

    # DataFrame 생성 및 뷰 등록
    df = spark.createDataFrame([(line,) for line in text_data], ["line"])
    df.createOrReplaceTempView("lines")

    # SQL로 Word Count
    word_counts = spark.sql("""
        WITH words AS (
            SELECT explode(split(lower(line), '\\\\s+')) AS word
            FROM lines
        ),
        cleaned_words AS (
            SELECT regexp_replace(word, '[^a-z0-9]', '') AS word
            FROM words
            WHERE word != ''
        )
        SELECT
            word,
            COUNT(*) AS count
        FROM cleaned_words
        WHERE word != ''
        GROUP BY word
        ORDER BY count DESC
    """)

    print("\nTop 10 words (SQL):")
    word_counts.show(10, truncate=False)

    return word_counts


def main():
    # SparkSession 생성
    spark = SparkSession.builder \
        .appName("Word Count Example") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    # 로그 레벨 설정
    spark.sparkContext.setLogLevel("WARN")

    # 샘플 텍스트 데이터
    text_data = [
        "Apache Spark is a unified analytics engine for large-scale data processing",
        "Spark provides high-level APIs in Java, Scala, Python and R",
        "Spark also supports a rich set of higher-level tools",
        "Spark SQL for structured data processing",
        "MLlib for machine learning",
        "GraphX for graph processing",
        "Spark Streaming for real-time data processing",
        "Spark is designed to be fast and general-purpose",
        "Spark extends the popular MapReduce model",
        "Spark supports in-memory computing which can improve performance",
        "Data processing with Spark is efficient and scalable",
        "Spark can process data from various sources like HDFS, S3, Kafka",
    ]

    print("Sample Data:")
    for line in text_data[:3]:
        print(f"  {line}")
    print(f"  ... (total {len(text_data)} lines)")

    # 세 가지 방식으로 Word Count 실행
    word_count_rdd(spark, text_data)
    word_count_df(spark, text_data)
    word_count_sql(spark, text_data)

    # 결과 저장 예시
    print("\n" + "=" * 50)
    print("Saving Results")
    print("=" * 50)

    output_path = "/tmp/word_count_output"
    word_counts = word_count_df(spark, text_data)

    # Parquet 형식으로 저장
    word_counts.write.mode("overwrite").parquet(f"{output_path}/parquet")
    print(f"Saved to {output_path}/parquet")

    # CSV 형식으로 저장 (단일 파일)
    word_counts.coalesce(1).write.mode("overwrite").csv(
        f"{output_path}/csv",
        header=True
    )
    print(f"Saved to {output_path}/csv")

    # 통계 출력
    print("\n" + "=" * 50)
    print("Statistics")
    print("=" * 50)
    print(f"Total unique words: {word_counts.count()}")
    print(f"Total word occurrences: {word_counts.agg({'count': 'sum'}).collect()[0][0]}")

    # SparkSession 종료
    spark.stop()


if __name__ == "__main__":
    main()

"""
PySpark DataFrame 처리 예제

실제 데이터 처리 시나리오를 보여주는 예제입니다:
- 데이터 생성 및 로딩
- 변환 (필터, 집계, 조인)
- 윈도우 함수
- UDF (User Defined Functions)

실행:
  spark-submit data_processing.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, when, sum as _sum, avg, count, countDistinct,
    to_date, year, month, dayofweek,
    row_number, rank, lag, lead,
    udf, pandas_udf
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, DateType, TimestampType
)
from datetime import datetime, timedelta
import random


def create_sample_data(spark):
    """샘플 데이터 생성"""
    print("Creating sample data...")

    # 고객 데이터
    customers = [
        (1, "Alice", "Gold", "New York", "2020-01-15"),
        (2, "Bob", "Silver", "Los Angeles", "2020-03-20"),
        (3, "Charlie", "Bronze", "Chicago", "2021-06-10"),
        (4, "Diana", "Gold", "Houston", "2019-11-05"),
        (5, "Eve", "Silver", "Phoenix", "2022-02-28"),
    ]

    customers_schema = StructType([
        StructField("customer_id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("segment", StringType(), True),
        StructField("city", StringType(), True),
        StructField("join_date", StringType(), True),
    ])

    customers_df = spark.createDataFrame(customers, customers_schema)
    customers_df = customers_df.withColumn("join_date", to_date(col("join_date")))

    # 상품 데이터
    products = [
        (101, "Laptop", "Electronics", 999.99),
        (102, "Phone", "Electronics", 599.99),
        (103, "Tablet", "Electronics", 399.99),
        (104, "Headphones", "Electronics", 149.99),
        (105, "T-Shirt", "Clothing", 29.99),
        (106, "Jeans", "Clothing", 59.99),
        (107, "Sneakers", "Footwear", 89.99),
    ]

    products_df = spark.createDataFrame(
        products,
        ["product_id", "name", "category", "price"]
    )

    # 주문 데이터 생성
    random.seed(42)
    orders = []
    order_id = 1
    base_date = datetime(2024, 1, 1)

    for day in range(90):  # 90일치 데이터
        order_date = (base_date + timedelta(days=day)).strftime("%Y-%m-%d")
        num_orders = random.randint(5, 15)

        for _ in range(num_orders):
            orders.append((
                order_id,
                random.choice([1, 2, 3, 4, 5]),
                random.choice([101, 102, 103, 104, 105, 106, 107]),
                random.randint(1, 5),
                round(random.uniform(50, 500), 2),
                random.choice(["completed", "completed", "completed", "pending", "cancelled"]),
                order_date
            ))
            order_id += 1

    orders_df = spark.createDataFrame(
        orders,
        ["order_id", "customer_id", "product_id", "quantity", "amount", "status", "order_date"]
    )
    orders_df = orders_df.withColumn("order_date", to_date(col("order_date")))

    return customers_df, products_df, orders_df


def basic_transformations(orders_df):
    """기본 변환 작업"""
    print("\n" + "=" * 60)
    print("Basic Transformations")
    print("=" * 60)

    # 필터링
    completed_orders = orders_df.filter(col("status") == "completed")
    print(f"\nCompleted orders: {completed_orders.count()}")

    # 새 컬럼 추가
    enhanced_df = orders_df.withColumn(
        "order_tier",
        when(col("amount") > 300, "high")
        .when(col("amount") > 150, "medium")
        .otherwise("low")
    ).withColumn(
        "order_month", month(col("order_date"))
    ).withColumn(
        "order_year", year(col("order_date"))
    )

    print("\nEnhanced DataFrame:")
    enhanced_df.select("order_id", "amount", "order_tier", "order_date", "order_month").show(5)

    return enhanced_df


def aggregations(orders_df, customers_df):
    """집계 연산"""
    print("\n" + "=" * 60)
    print("Aggregations")
    print("=" * 60)

    # 전체 통계
    print("\nOverall Statistics:")
    orders_df.filter(col("status") == "completed").agg(
        count("*").alias("total_orders"),
        _sum("amount").alias("total_revenue"),
        avg("amount").alias("avg_order_value"),
        countDistinct("customer_id").alias("unique_customers")
    ).show()

    # 월별 통계
    print("\nMonthly Statistics:")
    monthly_stats = orders_df \
        .filter(col("status") == "completed") \
        .withColumn("month", month(col("order_date"))) \
        .groupBy("month") \
        .agg(
            count("*").alias("orders"),
            _sum("amount").alias("revenue"),
            avg("amount").alias("avg_value")
        ) \
        .orderBy("month")

    monthly_stats.show()

    # 고객 세그먼트별 통계
    print("\nCustomer Segment Statistics:")
    segment_stats = orders_df \
        .filter(col("status") == "completed") \
        .join(customers_df, "customer_id") \
        .groupBy("segment") \
        .agg(
            count("*").alias("orders"),
            _sum("amount").alias("revenue"),
            countDistinct("customer_id").alias("customers")
        ) \
        .withColumn("revenue_per_customer", col("revenue") / col("customers")) \
        .orderBy(col("revenue").desc())

    segment_stats.show()

    return monthly_stats


def window_functions(orders_df, customers_df):
    """윈도우 함수"""
    print("\n" + "=" * 60)
    print("Window Functions")
    print("=" * 60)

    completed = orders_df.filter(col("status") == "completed")

    # 고객별 윈도우
    customer_window = Window.partitionBy("customer_id").orderBy("order_date")

    # 고객별 누적 구매액, 주문 순서
    customer_analysis = completed \
        .withColumn("order_rank", row_number().over(customer_window)) \
        .withColumn("cumulative_amount", _sum("amount").over(customer_window)) \
        .withColumn("prev_order_amount", lag("amount", 1).over(customer_window)) \
        .withColumn("next_order_amount", lead("amount", 1).over(customer_window))

    print("\nCustomer Order Analysis:")
    customer_analysis \
        .filter(col("customer_id") == 1) \
        .select(
            "customer_id", "order_date", "amount",
            "order_rank", "cumulative_amount",
            "prev_order_amount", "next_order_amount"
        ) \
        .orderBy("order_date") \
        .show(10)

    # 일별 매출 순위
    daily_window = Window.orderBy(col("daily_revenue").desc())

    daily_ranking = completed \
        .groupBy("order_date") \
        .agg(_sum("amount").alias("daily_revenue")) \
        .withColumn("rank", rank().over(daily_window))

    print("\nTop 5 Revenue Days:")
    daily_ranking.filter(col("rank") <= 5).show()

    return customer_analysis


def join_operations(customers_df, products_df, orders_df):
    """조인 연산"""
    print("\n" + "=" * 60)
    print("Join Operations")
    print("=" * 60)

    # 세 테이블 조인
    full_data = orders_df \
        .join(customers_df, "customer_id", "left") \
        .join(products_df, "product_id", "left")

    print("\nJoined Data Sample:")
    full_data.select(
        "order_id", "name", "segment", "category", "amount", "status"
    ).show(5)

    # 카테고리별, 세그먼트별 매출
    category_segment = full_data \
        .filter(col("status") == "completed") \
        .groupBy("category", "segment") \
        .agg(
            count("*").alias("orders"),
            _sum("amount").alias("revenue")
        ) \
        .orderBy("category", "segment")

    print("\nRevenue by Category and Segment:")
    category_segment.show()

    return full_data


def user_defined_functions(orders_df):
    """UDF 사용"""
    print("\n" + "=" * 60)
    print("User Defined Functions")
    print("=" * 60)

    # Python UDF
    @udf(returnType=StringType())
    def categorize_amount(amount):
        if amount is None:
            return "unknown"
        elif amount > 400:
            return "premium"
        elif amount > 200:
            return "standard"
        else:
            return "economy"

    # UDF 적용
    with_category = orders_df.withColumn(
        "amount_category",
        categorize_amount(col("amount"))
    )

    print("\nWith Amount Category:")
    with_category.select("order_id", "amount", "amount_category").show(5)

    # 카테고리별 통계
    print("\nStatistics by Amount Category:")
    with_category \
        .filter(col("status") == "completed") \
        .groupBy("amount_category") \
        .agg(
            count("*").alias("count"),
            avg("amount").alias("avg_amount")
        ) \
        .orderBy(col("avg_amount").desc()) \
        .show()

    return with_category


def save_results(spark, df, output_path):
    """결과 저장"""
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    # Parquet 형식으로 저장 (파티션 포함)
    df.write \
        .mode("overwrite") \
        .partitionBy("status") \
        .parquet(f"{output_path}/orders_parquet")

    print(f"Saved to {output_path}/orders_parquet")

    # Delta Lake 형식 (설정된 경우)
    # df.write.format("delta").mode("overwrite").save(f"{output_path}/orders_delta")


def main():
    # SparkSession 생성
    spark = SparkSession.builder \
        .appName("Data Processing Example") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # 샘플 데이터 생성
    customers_df, products_df, orders_df = create_sample_data(spark)

    print("=" * 60)
    print("Data Overview")
    print("=" * 60)
    print(f"Customers: {customers_df.count()} rows")
    print(f"Products: {products_df.count()} rows")
    print(f"Orders: {orders_df.count()} rows")

    print("\nCustomers Schema:")
    customers_df.printSchema()

    print("\nOrders Sample:")
    orders_df.show(5)

    # 변환 작업 실행
    enhanced_orders = basic_transformations(orders_df)
    monthly_stats = aggregations(orders_df, customers_df)
    customer_analysis = window_functions(orders_df, customers_df)
    full_data = join_operations(customers_df, products_df, orders_df)
    categorized = user_defined_functions(orders_df)

    # 결과 저장
    save_results(spark, enhanced_orders, "/tmp/spark_output")

    # SparkSession 종료
    spark.stop()


if __name__ == "__main__":
    main()

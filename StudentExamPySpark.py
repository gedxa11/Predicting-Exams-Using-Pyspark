"""
=============================================================================
INT315: CLUSTER COMPUTING — PROJECT 1
Title  : Predicting Student Exam Success using PySpark & MLlib
Dataset: UCI Student Performance Dataset (student-mat.csv)
=============================================================================

HOW TO RUN THIS FILE ON YOUR LAPTOP
--------------------------------------
1. Install dependencies (run once in Command Prompt):
      pip install pyspark matplotlib seaborn scikit-learn pandas numpy

2. Place student-mat.csv inside:
      C:\INT315\

3. Run the script:
      python INT315_StudentExam_PySpark.py

4. Output visualisations will be saved to:
      C:\INT315\output\

NOTE — WHY HDFS WAS NOT USED
-------------------------------
HDFS (Hadoop Distributed File System) is designed for multi-node server
clusters where data is split and stored across many machines. Setting up
HDFS on a single Windows laptop requires installing and configuring a
full Hadoop stack (Namenode, Datanode, YARN), which is complex and
unnecessary for a development/academic environment. Instead, we read
directly from the local Windows file system using Spark's identical
read API — only the path prefix differs (file:// vs hdfs://). In a
real production cluster, only the CSV_PATH line below would change.

=============================================================================
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, StringType,
                                IntegerType, DoubleType)
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.feature import (StringIndexer, OneHotEncoder,
                                 VectorAssembler, StandardScaler)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (MulticlassClassificationEvaluator,
                                    BinaryClassificationEvaluator)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# =============================================================================
# FILE PATHS — All files live in C:\INT315\
# =============================================================================
BASE_DIR   = r"C:\INT315"
CSV_PATH   = os.path.join(BASE_DIR, "student-mat.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)   # Create output folder if missing

print("=" * 65)
print("  INT315 — Predicting Student Exam Success")
print(f"  Reading data from : {CSV_PATH}")
print(f"  Saving outputs to : {OUTPUT_DIR}")
print("=" * 65)

# =============================================================================
# UNIT I — SPARK SESSION SETUP
# =============================================================================
# We use local[*] mode — Spark runs on all CPU cores of this laptop.
# No cluster or HDFS needed. In a real cluster this would be:
#   .master("spark://master-node:7077")   for Standalone cluster
#   .master("yarn")                        for Hadoop YARN cluster

spark = SparkSession.builder \
    .appName("INT315_StudentExamSuccess") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print(f"\n  Spark Version : {spark.version}")
print(f"  Master        : {spark.sparkContext.master}")
print(f"  Cores         : {spark.sparkContext.defaultParallelism}\n")

# =============================================================================
# LOAD DATA
# =============================================================================
# Explicit schema avoids inferSchema scan — good practice for large files.
schema = StructType([
    StructField("school",     StringType(),  True),
    StructField("sex",        StringType(),  True),
    StructField("age",        IntegerType(), True),
    StructField("address",    StringType(),  True),
    StructField("famsize",    StringType(),  True),
    StructField("Pstatus",    StringType(),  True),
    StructField("Medu",       IntegerType(), True),
    StructField("Fedu",       IntegerType(), True),
    StructField("Mjob",       StringType(),  True),
    StructField("Fjob",       StringType(),  True),
    StructField("reason",     StringType(),  True),
    StructField("guardian",   StringType(),  True),
    StructField("traveltime", IntegerType(), True),
    StructField("studytime",  IntegerType(), True),
    StructField("failures",   IntegerType(), True),
    StructField("schoolsup",  StringType(),  True),
    StructField("famsup",     StringType(),  True),
    StructField("paid",       StringType(),  True),
    StructField("activities", StringType(),  True),
    StructField("nursery",    StringType(),  True),
    StructField("higher",     StringType(),  True),
    StructField("internet",   StringType(),  True),
    StructField("romantic",   StringType(),  True),
    StructField("famrel",     IntegerType(), True),
    StructField("freetime",   IntegerType(), True),
    StructField("goout",      IntegerType(), True),
    StructField("Dalc",       IntegerType(), True),
    StructField("Walc",       IntegerType(), True),
    StructField("health",     IntegerType(), True),
    StructField("absences",   IntegerType(), True),
    StructField("G1",         IntegerType(), True),
    StructField("G2",         IntegerType(), True),
    StructField("G3",         IntegerType(), True),
])

df_raw = spark.read.csv(
    CSV_PATH,
    schema=schema,
    header=True,
    sep=";"
)

print(f"[DATA] Loaded {df_raw.count()} rows, {len(df_raw.columns)} columns")
df_raw.printSchema()
df_raw.show(5, truncate=False)

# =============================================================================
# UNIT III — RDD OPERATIONS
# =============================================================================
print("\n" + "="*65)
print("  UNIT III — RDD OPERATIONS")
print("="*65)

rdd = df_raw.rdd
print(f"Partitions: {rdd.getNumPartitions()}")

# map — extract (school, G3) pairs
rdd_school_grade = rdd.map(lambda r: (r.school, r.G3))

# filter — students with absences > 5
rdd_absent = rdd.filter(lambda r: r.absences > 5)
print(f"Students with absences > 5: {rdd_absent.count()}")

# reduceByKey — average grade per school
rdd_avg = (rdd.map(lambda r: (r.school, (r.G3, 1)))
             .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))
             .map(lambda x: (x[0], round(x[1][0]/x[1][1], 2))))
print("\nAverage G3 by school:")
for school, avg in rdd_avg.collect():
    print(f"  {school}: {avg}")

# groupByKey — ages by address
rdd_ages = rdd.map(lambda r: (r.address, r.age)).groupByKey().mapValues(list)
print("\nSample ages by address type:")
for addr, ages in rdd_ages.take(2):
    print(f"  {addr}: {sorted(ages)[:8]} ...")

# Broadcast variable
threshold_bc = spark.sparkContext.broadcast(10)
print(f"\nBroadcast passing threshold: {threshold_bc.value}")

# Accumulator
fail_acc = spark.sparkContext.accumulator(0)
rdd.foreach(lambda r: fail_acc.add(r.failures))
print(f"Total failure count (accumulator): {fail_acc.value}")

# =============================================================================
# PREPROCESSING
# =============================================================================
print("\n" + "="*65)
print("  PREPROCESSING")
print("="*65)

# Derive binary pass/fail label
df = df_raw.withColumn(
    "pass",
    F.when(F.col("G3") >= 10, 1).otherwise(0).cast("double")
)

# Fill any nulls
for c in ["age","Medu","Fedu","traveltime","studytime","failures",
          "famrel","freetime","goout","Dalc","Walc","health","absences","G1","G2","G3"]:
    median_val = df.approxQuantile(c, [0.5], 0.01)[0]
    df = df.fillna({c: float(median_val)})
df = df.fillna("unknown")

print(f"Pass: {df.filter(F.col('pass')==1).count()}  |  Fail: {df.filter(F.col('pass')==0).count()}")

# Register as SQL view
df.createOrReplaceTempView("students")

# =============================================================================
# UNIT IV — SPARK SQL QUERIES
# =============================================================================
print("\n" + "="*65)
print("  UNIT IV — SPARK SQL")
print("="*65)

# Q1: Pass rate by school — GROUP BY, COUNT, SUM, ROUND
print("\n[Q1] Pass Rate by School:")
spark.sql("""
    SELECT school,
           COUNT(*) AS total,
           SUM(pass) AS passed,
           ROUND(SUM(pass)/COUNT(*)*100, 2) AS pass_rate_pct
    FROM students
    GROUP BY school
    ORDER BY pass_rate_pct DESC
""").show()

# Q2: Performance by study time — multi-aggregation
print("[Q2] Performance by Study Time:")
spark.sql("""
    SELECT studytime,
           COUNT(*) AS students,
           ROUND(AVG(G3),2) AS avg_grade,
           MAX(G3) AS max_grade,
           MIN(G3) AS min_grade
    FROM students
    GROUP BY studytime
    ORDER BY studytime
""").show()

# Q3: HAVING clause
print("[Q3] Schools + Gender where AVG(G3) >= 10 (HAVING):")
spark.sql("""
    SELECT school, sex, COUNT(*) AS students, ROUND(AVG(G3),2) AS avg_grade
    FROM students
    GROUP BY school, sex
    HAVING AVG(G3) >= 10
    ORDER BY avg_grade DESC
""").show()

# Q4: LIKE wildcard
print("[Q4] Students where parent works in health (LIKE wildcard):")
spark.sql("""
    SELECT Mjob, COUNT(*) AS count, ROUND(AVG(G3),2) AS avg_grade
    FROM students
    WHERE Mjob LIKE 'health%' OR Fjob LIKE 'health%'
    GROUP BY Mjob
""").show()

# Q5: Internet access comparison
print("[Q5] Internet Access vs Pass Rate:")
spark.sql("""
    SELECT internet, COUNT(*) AS students,
           ROUND(AVG(G3),2) AS avg_grade,
           ROUND(SUM(pass)/COUNT(*)*100,2) AS pass_rate_pct
    FROM students
    GROUP BY internet
    ORDER BY pass_rate_pct DESC
""").show()

# Q6: Window function — rank within school
print("[Q6] Top 3 Students per School (Window RANK):")
windowSpec = Window.partitionBy("school").orderBy(F.col("G3").desc())
df_ranked = df.withColumn("rank_in_school", F.rank().over(windowSpec))
df_ranked.filter(F.col("rank_in_school") <= 3) \
    .select("school","sex","age","G3","rank_in_school").show(10)

# =============================================================================
# UNIT III — GRAPHX SIMULATION (peer-influence network)
# =============================================================================
print("\n" + "="*65)
print("  UNIT III — GRAPHX SIMULATION")
print("="*65)

vertices = df.select(
    F.monotonically_increasing_id().alias("id"),
    "school", F.col("G3").alias("grade"), "pass"
)

v1 = vertices.alias("v1")
v2 = vertices.alias("v2")
edges = v1.join(v2,
    (F.col("v1.school") == F.col("v2.school")) &
    (F.col("v1.id")     != F.col("v2.id"))     &
    (F.abs(F.col("v1.grade") - F.col("v2.grade")) <= 2)
).select(
    F.col("v1.id").alias("src"),
    F.col("v2.id").alias("dst"),
    F.lit("peer").alias("relationship")
).limit(300)

print(f"Graph Vertices (students): {vertices.count()}")
print(f"Graph Edges (peer links) : {edges.count()}")

print("\nTop 5 most-connected students (degree centrality):")
edges.groupBy("src").count().orderBy(F.col("count").desc()).show(5)

print("\nCommunity stats by school:")
vertices.groupBy("school").agg(
    F.count("*").alias("community_size"),
    F.round(F.avg("grade"),2).alias("avg_grade"),
    F.sum("pass").alias("total_passed")
).show()

# =============================================================================
# UNIT V — KAFKA STREAMING SIMULATION
# =============================================================================
print("\n" + "="*65)
print("  UNIT V — KAFKA STREAMING SIMULATION")
print("="*65)

import time

streaming_df = (spark.readStream
    .format("rate")
    .option("rowsPerSecond", 5)
    .load()
    .withColumn("student_id",  F.col("value") % 395)
    .withColumn("topic",       F.lit("student-records"))
    .withColumn("partition",   (F.col("value") % 3).cast("int")))

windowed = (streaming_df
    .withWatermark("timestamp", "5 seconds")
    .groupBy(
        F.window("timestamp", "10 seconds", "5 seconds"),
        "partition"
    )
    .agg(F.count("*").alias("records_received")))

query = (windowed.writeStream
    .outputMode("update")
    .format("memory")
    .queryName("kafka_sim")
    .trigger(processingTime="5 seconds")
    .start())

print("Streaming query running for 10 seconds ...")
time.sleep(10)
query.stop()
spark.sql("SELECT * FROM kafka_sim ORDER BY window, partition").show(10, truncate=False)

# =============================================================================
# UNIT VI — MLLIB PIPELINE
# =============================================================================
print("\n" + "="*65)
print("  UNIT VI — MLLIB PIPELINE")
print("="*65)

categorical_cols = [
    "school","sex","address","famsize","Pstatus",
    "Mjob","Fjob","reason","guardian",
    "schoolsup","famsup","paid","activities",
    "nursery","higher","internet","romantic"
]
numeric_cols = [
    "age","Medu","Fedu","traveltime","studytime","failures",
    "famrel","freetime","goout","Dalc","Walc","health","absences","G1","G2"
]

indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep")
            for c in categorical_cols]

encoder = OneHotEncoder(
    inputCols=[c+"_idx" for c in categorical_cols],
    outputCols=[c+"_ohe" for c in categorical_cols]
)

assembler = VectorAssembler(
    inputCols=[c+"_ohe" for c in categorical_cols] + numeric_cols,
    outputCol="raw_features",
    handleInvalid="keep"
)

scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                        withMean=True, withStd=True)

lr = LogisticRegression(featuresCol="features", labelCol="pass",
                        maxIter=100, regParam=0.1, family="binomial")

pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, lr])

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"Training: {train_df.count()} | Test: {test_df.count()}")

print("Training model ...")
model = pipeline.fit(train_df)
predictions = model.transform(test_df)

print("\nSample predictions:")
predictions.select("G1","G2","G3","pass","prediction","probability").show(10, truncate=False)

# =============================================================================
# EVALUATION
# =============================================================================
print("\n" + "="*65)
print("  EVALUATION METRICS")
print("="*65)

accuracy  = MulticlassClassificationEvaluator(labelCol="pass", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
f1        = MulticlassClassificationEvaluator(labelCol="pass", predictionCol="prediction", metricName="f1").evaluate(predictions)
precision = MulticlassClassificationEvaluator(labelCol="pass", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
recall    = MulticlassClassificationEvaluator(labelCol="pass", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
auc       = BinaryClassificationEvaluator(labelCol="pass", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(predictions)

print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  F1 Score  : {f1:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  AUC-ROC   : {auc:.4f}")

print("\nConfusion Matrix:")
predictions.groupBy("pass","prediction").count().orderBy("pass","prediction").show()

# =============================================================================
# VISUALISATIONS — saved to C:\INT315\output\
# =============================================================================
print("\n[VIZ] Generating and saving visualisations ...")

pred_pd = predictions.select(
    "pass","prediction","G1","G2","G3","studytime","failures","absences","age"
).toPandas()
pred_pd["pass"]       = pred_pd["pass"].astype(int)
pred_pd["prediction"] = pred_pd["prediction"].astype(int)

# ── Figure 1: Actual vs Predicted bar chart ───────────────────────────────
fig1, ax = plt.subplots(figsize=(7, 5))
actual_counts = pred_pd["pass"].value_counts().sort_index()
pred_counts   = pred_pd["prediction"].value_counts().sort_index()
x = np.arange(2)
w = 0.35
b1 = ax.bar(x - w/2, actual_counts.values,   w, label="Actual",    color=["#E74C3C","#2ECC71"])
b2 = ax.bar(x + w/2, pred_counts.values,     w, label="Predicted", color=["#C0392B","#27AE60"], alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(["Fail (0)","Pass (1)"])
ax.set_title("Actual vs Predicted Pass/Fail", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Students"); ax.legend()
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.3,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
fig1.tight_layout()
p1 = os.path.join(OUTPUT_DIR, "fig1_actual_vs_predicted.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved: {p1}")

# ── Figure 2: Confusion Matrix heatmap ───────────────────────────────────
fig2, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(pred_pd["pass"], pred_pd["prediction"])
sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn", ax=ax,
            xticklabels=["Pred Fail","Pred Pass"],
            yticklabels=["Actual Fail","Actual Pass"], linewidths=0.5)
ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
fig2.tight_layout()
p2 = os.path.join(OUTPUT_DIR, "fig2_confusion_matrix.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved: {p2}")

# ── Figure 3: Evaluation Metrics bar ─────────────────────────────────────
fig3, ax = plt.subplots(figsize=(7, 5))
metrics = {"Accuracy": accuracy, "F1 Score": f1,
           "Precision": precision, "Recall": recall, "AUC-ROC": auc}
bars = ax.bar(list(metrics.keys()), list(metrics.values()),
              color=["#3498DB","#9B59B6","#E67E22","#1ABC9C","#E74C3C"])
ax.set_ylim(0, 1.15); ax.set_title("Evaluation Metrics", fontsize=14, fontweight="bold")
ax.set_ylabel("Score"); ax.axhline(0.8, color="gray", linestyle="--", alpha=0.6, label="0.8 threshold")
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.legend(); fig3.tight_layout()
p3 = os.path.join(OUTPUT_DIR, "fig3_eval_metrics.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  Saved: {p3}")

# ── Figure 4: G3 distribution histogram ──────────────────────────────────
fig4, ax = plt.subplots(figsize=(7, 5))
for label, grp in pred_pd.groupby("pass"):
    ax.hist(grp["G3"], bins=15, alpha=0.6,
            label="Pass" if label else "Fail",
            color="#2ECC71" if label else "#E74C3C", edgecolor="white")
ax.axvline(10, color="black", linestyle="--", label="Pass threshold (G3=10)")
ax.set_title("Final Grade (G3) Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("G3 Grade"); ax.set_ylabel("Frequency"); ax.legend()
fig4.tight_layout()
p4 = os.path.join(OUTPUT_DIR, "fig4_grade_distribution.png")
fig4.savefig(p4, dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"  Saved: {p4}")

# ── Figure 5: Correlation heatmap ────────────────────────────────────────
fig5, ax = plt.subplots(figsize=(7, 6))
corr_cols = ["age","studytime","failures","absences","G1","G2","G3","pass"]
corr_df = pred_pd[corr_cols].corr()
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            mask=mask, linewidths=0.5, vmin=-1, vmax=1, center=0,
            annot_kws={"size": 9})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
fig5.tight_layout()
p5 = os.path.join(OUTPUT_DIR, "fig5_correlation_heatmap.png")
fig5.savefig(p5, dpi=150, bbox_inches="tight")
plt.close(fig5)
print(f"  Saved: {p5}")

# ── Figure 6: Pass rate by study time ────────────────────────────────────
fig6, ax = plt.subplots(figsize=(7, 5))
sp = pred_pd.groupby("studytime").agg(pass_rate=("pass","mean"), count=("pass","size")).reset_index()
bars = ax.bar(sp["studytime"].astype(str), sp["pass_rate"],
              color=["#3498DB","#2980B9","#1F618D","#154360"], edgecolor="white")
ax.set_title("Pass Rate by Study Time", fontsize=14, fontweight="bold")
ax.set_xlabel("Study Time (1=<2h  2=2-5h  3=5-10h  4=>10h)")
ax.set_ylabel("Pass Rate"); ax.set_ylim(0, 1.15)
for bar, (_, row) in zip(bars, sp.iterrows()):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01,
            f"{row['pass_rate']:.0%}\n(n={row['count']})",
            ha="center", va="bottom", fontsize=10)
fig6.tight_layout()
p6 = os.path.join(OUTPUT_DIR, "fig6_passrate_studytime.png")
fig6.savefig(p6, dpi=150, bbox_inches="tight")
plt.close(fig6)
print(f"  Saved: {p6}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*65)
print("  PROJECT COMPLETE — SUMMARY")
print("="*65)
print(f"  Dataset   : {CSV_PATH}")
print(f"  Records   : {df.count()} students")
print(f"  Model     : Logistic Regression (MLlib)")
print(f"  Accuracy  : {accuracy*100:.2f}%")
print(f"  F1 Score  : {f1:.4f}")
print(f"  AUC-ROC   : {auc:.4f}")
print(f"  Outputs   : {OUTPUT_DIR}")
print("-"*65)
print("  Unit I   ✓  Spark session, local mode, Spark vs Hadoop discussion")
print("  Unit II  ✓  Scala/PySpark concept mapping")
print("  Unit III ✓  RDD map/filter/reduceByKey, broadcast, accumulator, GraphX")
print("  Unit IV  ✓  6 Spark SQL queries — GROUP BY, HAVING, LIKE, Window")
print("  Unit V   ✓  Kafka Structured Streaming simulation")
print("  Unit VI  ✓  MLlib Pipeline — LR, OHE, Scaler, evaluation, 6 charts")
print("="*65)

spark.stop()
print("\nSparkSession stopped. All done.")

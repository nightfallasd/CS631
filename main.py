from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand,row_number
from graphframes import GraphFrame
from pyspark.sql.window import Window


def main():
    spark = SparkSession.builder.appName("SparkTest").getOrCreate()

    nodes_df = spark.read.csv("marvel-unimodal-nodes.csv", header=True, inferSchema=True)
    edges_df = spark.read.csv("marvel-unimodal-edges.csv", header=True, inferSchema=True)

    reversed_edges_df = edges_df.select(
        col("Target").alias("Source"),
        col("Source").alias("Target"),
        "Weight"
    )

    edges_df = edges_df.union(reversed_edges_df)

    g = GraphFrame(nodes_df.withColumnRenamed("Id", "id"),
                   edges_df.withColumnRenamed("Source", "src").withColumnRenamed("Target", "dst"))

    node_rank = g.pageRank(resetProbability=0.15, maxIter=5).vertices

    joinNodes = g.edges.join(node_rank, col("src") == col("id")).withColumn("NewWeight", col("Weight") * col("pagerank")*(0.9+rand()*0.1))
    edges = joinNodes.select("src","dst","NewWeight")
    nodes = g.vertices

    joined_df = nodes.join(edges, col("id") == col("src"))
    edges = joined_df.select("src", "dst", "NewWeight")
    num_iter = 1
    for i in range(num_iter):

        windowSpec = Window.partitionBy("src").orderBy((0.3+rand()*0.7) * col("NewWeight"))

        max_weights_df = joined_df.withColumn("row_number", row_number().over(windowSpec)).filter(col("row_number") == 1). \
            select(col("src").alias("newsrc"), col("NewWeight").alias("MaxWeight"))

        max_weight_edges_df = max_weights_df.join(edges, (max_weights_df.newsrc == edges.src) & (
                max_weights_df.MaxWeight == edges.NewWeight))

        nodes = max_weight_edges_df.select(col("src").alias("id"), col("dst").alias("Label"),col("NewWeight").alias("Weight"))

        edges = edges.join(nodes,col("dst") == col("id")).select(col("src"),col("Label").alias("dst"),col("NewWeight"))

    nodes.orderBy("Label").show(327)
    pandas_df = nodes.orderBy("Label").toPandas()
    pandas_df.to_csv('node.csv')

if __name__ == "__main__":
    main()

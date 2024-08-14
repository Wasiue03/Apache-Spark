from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark Session
spark = SparkSession.builder.appName("LinearRegressionExample").getOrCreate()

# Load data
data = spark.read.csv("emp.csv", header=True, inferSchema=True)

# Combine features into a single vector column
assembler = VectorAssembler(inputCols=["Age", "Experience"], outputCol="features")
data = assembler.transform(data)

# Select the required columns (features and label)
final_data = data.select("features", "Salary")

# Split the data into training and testing sets
train_data, test_data = final_data.randomSplit([0.8, 0.2])

# Initialize Linear Regression model
lr = LinearRegression(labelCol="Salary")

# Fit the model to the training data
lr_model = lr.fit(train_data)

# Make predictions on test data
predictions = lr_model.transform(test_data)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="Salary", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Inspect the model's coefficients and intercept
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

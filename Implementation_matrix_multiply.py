import random
from timeit import default_timer as timer
from scipy.sparse import lil_matrix
from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix


# Initialize Spark session
spark_session = SparkSession.builder.appName('MatrixMultiplicationComparison').master('local').getOrCreate()
spark_session.sparkContext.setLogLevel("ERROR")

# Global matrix size
matrix_size = 2  # Set the matrix size (N x N)

# Function to create a matrix of size 'size' filled with random values within a range
def generate_matrix(size, value_range):
    return [[random.randint(-value_range, value_range) for _ in range(size)] for _ in range(size)]

# Function to create an empty matrix of given size, initialized with zeros
def create_zero_matrix(size):
    return [[0 for _ in range(size)] for _ in range(size)]

# Function for sequential matrix multiplication
def sequential_matrix_multiply(A, B, C, size):
    for i in range(size):
        for j in range(size):
            total = 0
            for k in range(size):
                total += A[i][k] * B[k][j]
            C[i][j] = total
    return C

# Function to convert an RDD to a BlockMatrix for Spark computation
def convert_to_block_matrix(rdd, rows, columns):
    return IndexedRowMatrix(rdd.zipWithIndex().map(lambda x: IndexedRow(x[1], x[0]))).toBlockMatrix(rows, columns)

# Function to convert a BlockMatrix back to a local array (NumPy / SciPy format)
def block_matrix_to_array(block_matrix):
    result = lil_matrix((block_matrix.numRows(), block_matrix.numCols()))
    for row in block_matrix.rows.collect():
        result[row.index] = row.vector
    return result

# Matrix generation for test
matrix_A = generate_matrix(matrix_size, 500)
matrix_B = generate_matrix(matrix_size, 500)
result_matrix = create_zero_matrix(matrix_size)

# Sequential multiplication
print('Starting sequential matrix multiplication...')
start_time = timer()
result_matrix = sequential_matrix_multiply(matrix_A, matrix_B, result_matrix, matrix_size)
end_time = timer()
print('Sequential execution time (seconds):', end_time - start_time)

# Convert matrices to RDDs for Spark
rdd_A = spark_session.sparkContext.parallelize(matrix_A)
rdd_B = spark_session.sparkContext.parallelize(matrix_B)

# Spark matrix multiplication using BlockMatrix
print('Starting Spark-based matrix multiplication...')
start_time = timer()
block_matrix_A = convert_to_block_matrix(rdd_A, matrix_size, matrix_size)
block_matrix_B = convert_to_block_matrix(rdd_B, matrix_size, matrix_size)
spark_result = block_matrix_A.multiply(block_matrix_B)
end_time = timer()
print('Spark execution time (seconds):', end_time - start_time)

# Convert Spark result back to a local array
local_result = block_matrix_to_array(spark_result.toIndexedRowMatrix())

# Display matrices and results (only for small matrices)
if matrix_size <= 4:
    print("Sequential result matrix:")
    for row in result_matrix:
        print(row)
    
    print("\nSpark result matrix:")
    print(local_result)



import random
import json
from timeit import default_timer as timer
from numpy import mat
from scipy.sparse import lil_matrix
from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import *


app_name = 'Matrix Multiplication with Spark'
master = 'local'
spark = SparkSession.builder.appName(app_name).master(master).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
num_tests = 10
matrix_sizes = [1, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
def generate_matrix(dim, max_value):
    return [[random.randint((max_value * -1), max_value) for _ in range(dim)] for _ in range(dim)]
def create_zero_matrix(dim):
    return [[0 for _ in range(dim)] for _ in range(dim)]
def multiply_matrices(A, B, C, dim):
    for i in range(dim):
        for j in range(dim):
            total = 0
            for k in range(dim):
                total += A[i][k] * B[k][j]
            C[i][j] = total
    return C
def to_block_matrix(rdd, rows, columns):
    return IndexedRowMatrix(rdd.zipWithIndex().map(lambda i: IndexedRow(i[1], i[0]))).toBlockMatrix(rows, columns)
def convert_indexed_row_matrix_to_array(matrix):
    result = lil_matrix((matrix.numRows(), matrix.numCols()))
    for indexed_row in matrix.rows.collect():
        result[indexed_row.index] = indexed_row.vector
    return result
def retrieve_matrices(matrix_list, iteration, dim):
    for x in range(len(matrix_list)):
        if matrix_list[x]['Dimension'] == dim:
            temp = matrix_list[x]['Matrices']
            for y in range(len(temp)):
                if temp[y]['Iteration'] == iteration:
                    return [temp[y]['A'], temp[y]['B']]
def sequential_test(num_tests, matrix_sizes, max_value):
    result_data = {'TestType': 'Sequential Tests', 'Results': []}
    matrix_data = []
    for dim in matrix_sizes:
        print(f"Running sequential matrix multiplication of {dim} x {dim} matrices.")
        test_results = {'Dimension': dim, 'Execution Times': [0] * num_tests}
        matrices = {'Dimension': dim, 'Matrices': []}
        for i in range(num_tests):
            A = generate_matrix(dim, max_value)
            B = generate_matrix(dim, max_value)
            C = create_zero_matrix(dim)
            start = timer()
            C = multiply_matrices(A, B, C, dim)
            end = timer()
            test_results['Execution Times'][i] = end - start
            matrices['Matrices'].append({'Iteration': i, 'A': A, 'B': B})
        result_data['Results'].append(test_results)
        matrix_data.append(matrices)
        print(f"Completed testing sequential matrix multiplication of {dim} x {dim} matrices.")
    return [result_data, matrix_data]
def spark_test(num_tests, matrix_sizes, matrix_data):
    result_data = {'TestType': 'Spark Tests', 'Results': []}
    for dim in matrix_sizes:
        print(f"Running Spark matrix multiplication of {dim} x {dim} matrices.")
        test_results = {'Dimension': dim, 'Execution Times': [0] * num_tests}
        for i in range(num_tests):
            A, B = retrieve_matrices(matrix_data, i, dim)
            A_rdd = spark.sparkContext.parallelize(A)
            B_rdd = spark.sparkContext.parallelize(B)
            start = timer()
            C_matrix = to_block_matrix(A_rdd, dim, dim).multiply(to_block_matrix(B_rdd, dim, dim))
            end = timer()
            test_results['Execution Times'][i] = end - start
        result_data['Results'].append(test_results)
        print(f"Completed testing Spark matrix multiplication of {dim} x {dim} matrices.")
    return result_data
def execute_tests(num_tests, matrix_sizes):
    print("Initializing testing process...")
    sequential_results, matrices = sequential_test(num_tests, matrix_sizes, matrix_sizes[-1])
    spark_results = spark_test(num_tests, matrix_sizes, matrices)
    print("Saving results...")
    with open("matrix_multiplication_results.json", "w") as output_file:
        output_file.write("[")
        output_file.write(json.dumps(sequential_results))
        output_file.write(",")
        output_file.write(json.dumps(spark_results))
        output_file.write("]")
    print("Testing complete. Results saved to 'matrix_multiplication_results.json'.")
execute_tests(num_tests, matrix_sizes)


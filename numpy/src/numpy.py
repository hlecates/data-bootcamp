import numpy as np

def basic_array_creation():
    # Create array from list
    arr1 = np.array([1, 2, 3, 4, 5])
    print(f"From list: {arr1}")
    
    # Create 2D array
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"2D array:\n{arr2}")
    
    # Create arrays with specific values
    zeros = np.zeros((3, 4))
    print(f"Zeros array:\n{zeros}")
    
    ones = np.ones((2, 3))
    print(f"Ones array:\n{ones}")
    
    # Create range array
    range_arr = np.arange(0, 10, 2)
    print(f"Range array: {range_arr}")
    
    # Create evenly spaced array
    linspace_arr = np.linspace(0, 1, 5)
    print(f"Linspace array: {linspace_arr}")
    
    # Create random arrays
    random_arr = np.random.rand(3, 3)
    print(f"Random array:\n{random_arr}")
    
    normal_arr = np.random.randn(2, 2)
    print(f"Normal distribution array:\n{normal_arr}")

def array_properties():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"Array:\n{arr}")
    print(f"Shape: {arr.shape}")
    print(f"Size: {arr.size}")
    print(f"Number of dimensions: {arr.ndim}")
    print(f"Data type: {arr.dtype}")
    print(f"Item size: {arr.itemsize} bytes")

def indexing_and_slicing():
    # Create a 2D array
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(f"Original array:\n{arr}")
    
    # Basic indexing
    print(f"Element at (0,1): {arr[0, 1]}")
    print(f"First row: {arr[0]}")
    print(f"Second column: {arr[:, 1]}")
    
    # Slicing
    print(f"Rows 0-1, columns 1-3:\n{arr[0:2, 1:3]}")
    print(f"Every second element: {arr[::2]}")
    print(f"Reversed array:\n{arr[::-1]}")
    
    # Boolean indexing
    mask = arr > 5
    print(f"Boolean mask:\n{mask}")
    print(f"Elements > 5: {arr[mask]}")

def array_operations():
    arr1 = np.array([1, 2, 3, 4])
    arr2 = np.array([5, 6, 7, 8])
    
    print(f"Array 1: {arr1}")
    print(f"Array 2: {arr2}")
    
    # Arithmetic operations
    print(f"Addition: {arr1 + arr2}")
    print(f"Multiplication: {arr1 * arr2}")
    print(f"Power: {arr1 ** 2}")
    
    # Comparison operations
    print(f"Greater than 2: {arr1 > 2}")
    print(f"Equal to 3: {arr1 == 3}")
    
    # Aggregation
    print(f"Sum: {arr1.sum()}")
    print(f"Mean: {arr1.mean()}")
    print(f"Standard deviation: {arr1.std()}")
    print(f"Min: {arr1.min()}, Max: {arr1.max()}")

def mathematical_functions():
    arr = np.array([1, 2, 3, 4, 5])
    print(f"Original array: {arr}")
    
    # Basic math
    print(f"Square root: {np.sqrt(arr)}")
    print(f"Square: {np.square(arr)}")
    print(f"Exponential: {np.exp(arr)}")
    print(f"Natural log: {np.log(arr)}")
    
    # Trigonometric functions
    angles = np.array([0, np.pi/2, np.pi])
    print(f"Angles: {angles}")
    print(f"Sine: {np.sin(angles)}")
    print(f"Cosine: {np.cos(angles)}")
    
    # Rounding
    float_arr = np.array([1.1, 2.7, 3.3, 4.9])
    print(f"Float array: {float_arr}")
    print(f"Round: {np.round(float_arr)}")
    print(f"Floor: {np.floor(float_arr)}")
    print(f"Ceiling: {np.ceil(float_arr)}")

def statistical_functions():
    # Create sample data
    data = np.random.normal(0, 1, 1000)
    print(f"Sample data (first 10): {data[:10]}")
    
    # Descriptive statistics
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Median: {np.median(data):.4f}")
    print(f"Standard deviation: {np.std(data):.4f}")
    print(f"Variance: {np.var(data):.4f}")
    
    # Percentiles
    print(f"25th percentile: {np.percentile(data, 25):.4f}")
    print(f"75th percentile: {np.percentile(data, 75):.4f}")
    
    # Correlation
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])
    correlation = np.corrcoef(x, y)[0, 1]
    print(f"Correlation between x and y: {correlation:.4f}")

def array_manipulation():
    # Reshaping
    arr = np.arange(12)
    print(f"Original array: {arr}")
    
    reshaped = arr.reshape(3, 4)
    print(f"Reshaped to 3x4:\n{reshaped}")
    
    # Transpose
    transposed = reshaped.T
    print(f"Transposed:\n{transposed}")
    
    # Concatenation
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    concatenated = np.concatenate([arr1, arr2])
    print(f"Concatenated: {concatenated}")
    
    # Stacking
    stacked = np.vstack([arr1, arr2])
    print(f"Vertically stacked:\n{stacked}")
    
    # Splitting
    split_arr = np.array([1, 2, 3, 4, 5, 6])
    splits = np.split(split_arr, 3)
    print(f"Split into 3 parts: {splits}")

def broadcasting_demo():
    # Scalar broadcasting
    arr = np.array([1, 2, 3, 4])
    result = arr + 5
    print(f"Array: {arr}")
    print(f"Array + 5: {result}")
    
    # Array broadcasting
    arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr2 = np.array([10, 20, 30])
    result = arr1 + arr2
    print(f"Array 1:\n{arr1}")
    print(f"Array 2: {arr2}")
    print(f"Broadcasted result:\n{result}")
    
    # Adding dimensions
    arr_1d = np.array([1, 2, 3])
    arr_col = arr_1d[:, np.newaxis]
    arr_row = arr_1d[np.newaxis, :]
    print(f"1D array: {arr_1d}")
    print(f"Column vector:\n{arr_col}")
    print(f"Row vector:\n{arr_row}")

def linear_algebra_demo():
    # Matrix operations
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    
    # Matrix multiplication
    C = np.dot(A, B)
    print(f"Matrix multiplication A @ B:\n{C}")
    
    # Alternative syntax
    C_alt = A @ B
    print(f"Using @ operator:\n{C_alt}")
    
    # Matrix properties
    print(f"Determinant of A: {np.linalg.det(A)}")
    print(f"Trace of A: {np.trace(A)}")
    
    # Matrix inverse
    A_inv = np.linalg.inv(A)
    print(f"Inverse of A:\n{A_inv}")
    
    # Eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(A)
    print(f"Eigenvalues: {eigenvals}")
    print(f"Eigenvectors:\n{eigenvecs}")
    
    # Solving linear equations Ax = b
    b = np.array([5, 6])
    x = np.linalg.solve(A, b)
    print(f"Solution to Ax = b: {x}")

def practical_examples():
    # Data normalization
    data = np.random.normal(100, 15, 1000)
    normalized = (data - np.mean(data)) / np.std(data)
    print(f"Original data mean: {np.mean(data):.2f}, std: {np.std(data):.2f}")
    print(f"Normalized data mean: {np.mean(normalized):.2f}, std: {np.std(normalized):.2f}")
    
    # Moving average
    time_series = np.random.randn(100)
    window_size = 5
    moving_avg = np.convolve(time_series, np.ones(window_size)/window_size, mode='valid')
    print(f"Moving average length: {len(moving_avg)}")
    
    # Boolean operations for data filtering
    temperatures = np.random.normal(20, 5, 100)
    hot_days = temperatures > 25
    cold_days = temperatures < 15
    print(f"Hot days (>25°C): {np.sum(hot_days)}")
    print(f"Cold days (<15°C): {np.sum(cold_days)}")
    
    # Vectorized operations (much faster than loops)
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x) * np.cos(x)
    print(f"Vectorized calculation length: {len(y)}")

if __name__ == "__main__":
    print("NumPy Examples and Demonstrations")
    print("=" * 40)
    
    basic_array_creation()
    array_properties()
    indexing_and_slicing()
    array_operations()
    mathematical_functions()
    statistical_functions()
    array_manipulation()
    broadcasting_demo()
    linear_algebra_demo()
    practical_examples()
    
    print("\n" + "=" * 40)
    print("All demonstrations completed!") 
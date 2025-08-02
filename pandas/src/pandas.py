import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('default')
sns.set_theme(style="whitegrid")

def basic_dataframe_creation():
    # Create DataFrame from dictionary
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'Age': [25, 30, 35, 28, 32],
        'City': ['NYC', 'LA', 'Chicago', 'Boston', 'Seattle'],
        'Salary': [50000, 60000, 70000, 55000, 65000]
    }
    df = pd.DataFrame(data)
    print("DataFrame from dictionary:")
    print(df)
    
    # Create DataFrame from list of lists
    data_list = [
        ['Alice', 25, 'NYC', 50000],
        ['Bob', 30, 'LA', 60000],
        ['Charlie', 35, 'Chicago', 70000],
        ['Diana', 28, 'Boston', 55000],
        ['Eve', 32, 'Seattle', 65000]
    ]
    columns = ['Name', 'Age', 'City', 'Salary']
    df2 = pd.DataFrame(data_list, columns=columns)
    print("\nDataFrame from list of lists:")
    print(df2)
    
    # Create DataFrame with random data
    np.random.seed(42)
    df_random = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.choice(['X', 'Y', 'Z'], 100),
        'D': np.random.randint(1, 100, 100)
    })
    print("\nDataFrame with random data:")
    print(df_random.head())
    
    return df, df_random

def data_exploration(df):
    print("DataFrame Info:")
    print(df.info())
    
    print("\nDataFrame Shape:")
    print(df.shape)
    
    print("\nDataFrame Head:")
    print(df.head())
    
    print("\nDataFrame Tail:")
    print(df.tail(3))
    
    print("\nDataFrame Description:")
    print(df.describe())
    
    print("\nDataFrame Columns:")
    print(df.columns.tolist())
    
    print("\nDataFrame Index:")
    print(df.index)
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    print(df.isnull().sum())

def indexing_and_selection(df):
    # Column selection
    print("Selecting single column:")
    print(df['Name'])
    
    print("\nSelecting multiple columns:")
    print(df[['Name', 'Age', 'Salary']])
    
    # Row selection by index
    print("\nSelecting first row:")
    print(df.iloc[0])
    
    print("\nSelecting first 3 rows:")
    print(df.iloc[0:3])
    
    # Row selection by label
    print("\nSelecting row by label:")
    print(df.loc[0])
    
    # Boolean indexing
    print("\nPeople older than 30:")
    print(df[df['Age'] > 30])
    
    print("\nPeople with salary > 55000:")
    print(df[df['Salary'] > 55000])
    
    # Multiple conditions
    print("\nPeople older than 30 AND salary > 55000:")
    print(df[(df['Age'] > 30) & (df['Salary'] > 55000)])
    
    # String filtering
    print("\nPeople from NYC:")
    print(df[df['City'] == 'NYC'])

def data_manipulation(df):
    # Adding new column
    df['Department'] = ['HR', 'IT', 'Sales', 'Marketing', 'IT']
    print("DataFrame with new column:")
    print(df)
    
    # Modifying existing column
    df['Salary'] = df['Salary'] * 1.1  # 10% raise
    print("\nDataFrame after salary increase:")
    print(df)
    
    # Creating calculated column
    df['Salary_Category'] = df['Salary'].apply(lambda x: 'High' if x > 60000 else 'Low')
    print("\nDataFrame with calculated column:")
    print(df)
    
    # Dropping columns
    df_clean = df.drop('Department', axis=1)
    print("\nDataFrame after dropping column:")
    print(df_clean)
    
    # Renaming columns
    df_renamed = df.rename(columns={'Name': 'Full_Name', 'City': 'Location'})
    print("\nDataFrame with renamed columns:")
    print(df_renamed)

def sorting_and_grouping(df):
    # Sorting
    print("Sorting by Age (ascending):")
    print(df.sort_values('Age'))
    
    print("\nSorting by Salary (descending):")
    print(df.sort_values('Salary', ascending=False))
    
    print("\nSorting by multiple columns:")
    print(df.sort_values(['Age', 'Salary'], ascending=[True, False]))
    
    # Grouping
    df['Department'] = ['HR', 'IT', 'Sales', 'Marketing', 'IT']
    
    print("\nGrouping by Department:")
    grouped = df.groupby('Department')
    print(grouped.groups)
    
    print("\nMean salary by department:")
    print(df.groupby('Department')['Salary'].mean())
    
    print("\nMultiple aggregations by department:")
    print(df.groupby('Department').agg({
        'Age': ['mean', 'min', 'max'],
        'Salary': ['mean', 'sum', 'count']
    }))

def data_cleaning(df):
    # Create DataFrame with missing values
    df_missing = df.copy()
    df_missing.loc[1, 'Age'] = np.nan
    df_missing.loc[2, 'Salary'] = np.nan
    df_missing.loc[3, 'City'] = np.nan
    
    print("DataFrame with missing values:")
    print(df_missing)
    
    print("\nMissing values count:")
    print(df_missing.isnull().sum())
    
    # Filling missing values
    df_filled = df_missing.copy()
    df_filled['Age'].fillna(df_filled['Age'].mean(), inplace=True)
    df_filled['Salary'].fillna(df_filled['Salary'].median(), inplace=True)
    df_filled['City'].fillna('Unknown', inplace=True)
    
    print("\nDataFrame after filling missing values:")
    print(df_filled)
    
    # Removing duplicates
    df_duplicates = pd.concat([df, df.iloc[0:2]])  # Add duplicate rows
    print("\nDataFrame with duplicates:")
    print(df_duplicates)
    
    df_no_duplicates = df_duplicates.drop_duplicates()
    print("\nDataFrame after removing duplicates:")
    print(df_no_duplicates)

def data_analysis(df):
    # Basic statistics
    print("Numeric columns statistics:")
    print(df.describe())
    
    print("\nAge statistics:")
    print(f"Mean age: {df['Age'].mean():.2f}")
    print(f"Median age: {df['Age'].median():.2f}")
    print(f"Standard deviation: {df['Age'].std():.2f}")
    print(f"Min age: {df['Age'].min()}")
    print(f"Max age: {df['Age'].max()}")
    
    # Value counts
    print("\nCity distribution:")
    print(df['City'].value_counts())
    
    # Correlation
    print("\nCorrelation between Age and Salary:")
    print(df['Age'].corr(df['Salary']))
    
    # Pivot tables
    df['Department'] = ['HR', 'IT', 'Sales', 'Marketing', 'IT']
    pivot_table = df.pivot_table(
        values='Salary', 
        index='Department', 
        columns='City', 
        aggfunc='mean'
    )
    print("\nPivot table (Salary by Department and City):")
    print(pivot_table)

def data_visualization(df):
    # Create a larger dataset for better visualization
    np.random.seed(42)
    large_df = pd.DataFrame({
        'Age': np.random.normal(35, 10, 100),
        'Salary': np.random.normal(60000, 15000, 100),
        'Department': np.random.choice(['HR', 'IT', 'Sales', 'Marketing'], 100),
        'Experience': np.random.randint(1, 20, 100)
    })
    
    # Histogram
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(large_df['Age'], bins=20, alpha=0.7, color='skyblue')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    
    # Scatter plot
    plt.subplot(2, 2, 2)
    plt.scatter(large_df['Age'], large_df['Salary'], alpha=0.6)
    plt.title('Age vs Salary')
    plt.xlabel('Age')
    plt.ylabel('Salary')
    
    # Box plot
    plt.subplot(2, 2, 3)
    large_df.boxplot(column='Salary', by='Department', ax=plt.gca())
    plt.title('Salary by Department')
    plt.suptitle('')  # Remove default title
    
    # Bar plot
    plt.subplot(2, 2, 4)
    dept_counts = large_df['Department'].value_counts()
    dept_counts.plot(kind='bar', color='lightgreen')
    plt.title('Employee Count by Department')
    plt.xlabel('Department')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    plt.close()

def time_series_example():
    # Create time series data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    ts_df = pd.DataFrame({
        'Date': dates,
        'Sales': np.random.normal(1000, 200, 100) + np.sin(np.arange(100) * 2 * np.pi / 7) * 100,
        'Temperature': np.random.normal(20, 5, 100),
        'DayOfWeek': dates.dayofweek
    })
    
    print("Time series data:")
    print(ts_df.head())
    
    # Set date as index
    ts_df.set_index('Date', inplace=True)
    
    print("\nTime series with date index:")
    print(ts_df.head())
    
    # Resampling
    monthly_sales = ts_df['Sales'].resample('M').mean()
    print("\nMonthly average sales:")
    print(monthly_sales.head())
    
    # Rolling average
    rolling_avg = ts_df['Sales'].rolling(window=7).mean()
    print("\n7-day rolling average of sales:")
    print(rolling_avg.head(10))
    
    # Plot time series
    plt.figure(figsize=(12, 6))
    plt.plot(ts_df.index, ts_df['Sales'], label='Daily Sales', alpha=0.7)
    plt.plot(ts_df.index, rolling_avg, label='7-day Rolling Average', linewidth=2)
    plt.title('Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

def merging_and_joining():
    # Create two DataFrames
    df1 = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'Age': [25, 30, 35, 28, 32]
    })
    
    df2 = pd.DataFrame({
        'ID': [1, 2, 3, 6, 7],
        'Salary': [50000, 60000, 70000, 80000, 90000],
        'Department': ['HR', 'IT', 'Sales', 'Marketing', 'IT']
    })
    
    print("DataFrame 1:")
    print(df1)
    print("\nDataFrame 2:")
    print(df2)
    
    # Inner join
    inner_merged = pd.merge(df1, df2, on='ID', how='inner')
    print("\nInner join:")
    print(inner_merged)
    
    # Left join
    left_merged = pd.merge(df1, df2, on='ID', how='left')
    print("\nLeft join:")
    print(left_merged)
    
    # Right join
    right_merged = pd.merge(df1, df2, on='ID', how='right')
    print("\nRight join:")
    print(right_merged)
    
    # Outer join
    outer_merged = pd.merge(df1, df2, on='ID', how='outer')
    print("\nOuter join:")
    print(outer_merged)

def advanced_operations(df):
    # Apply function
    df['Age_Category'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Middle' if x < 40 else 'Senior')
    print("DataFrame with age categories:")
    print(df)
    
    # Map function
    city_mapping = {'NYC': 'New York', 'LA': 'Los Angeles', 'Chicago': 'Chicago', 'Boston': 'Boston', 'Seattle': 'Seattle'}
    df['City_Full'] = df['City'].map(city_mapping)
    print("\nDataFrame with full city names:")
    print(df)
    
    # Cut function for binning
    df['Salary_Bin'] = pd.cut(df['Salary'], bins=3, labels=['Low', 'Medium', 'High'])
    print("\nDataFrame with salary bins:")
    print(df)
    
    # String operations
    df['Name_Length'] = df['Name'].str.len()
    df['Name_Upper'] = df['Name'].str.upper()
    print("\nDataFrame with string operations:")
    print(df[['Name', 'Name_Length', 'Name_Upper']])

def performance_analysis(df):
    # Create larger dataset for performance testing
    np.random.seed(42)
    large_df = pd.DataFrame({
        'A': np.random.randn(10000),
        'B': np.random.randn(10000),
        'C': np.random.choice(['X', 'Y', 'Z'], 10000),
        'D': np.random.randint(1, 100, 10000)
    })
    
    print("Large DataFrame shape:", large_df.shape)
    
    # Memory usage
    print(f"Memory usage: {large_df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Performance comparison
    import time
    
    # Method 1: Using apply
    start_time = time.time()
    result1 = large_df['A'].apply(lambda x: x * 2)
    time1 = time.time() - start_time
    
    # Method 2: Using vectorized operation
    start_time = time.time()
    result2 = large_df['A'] * 2
    time2 = time.time() - start_time
    
    print(f"Apply method time: {time1:.4f} seconds")
    print(f"Vectorized method time: {time2:.4f} seconds")
    print(f"Speed improvement: {time1/time2:.1f}x faster")

if __name__ == "__main__":
    print("Pandas Examples and Demonstrations")
    print("=" * 40)
    
    # Create sample data
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'Age': [25, 30, 35, 28, 32],
        'City': ['NYC', 'LA', 'Chicago', 'Boston', 'Seattle'],
        'Salary': [50000, 60000, 70000, 55000, 65000]
    }
    df = pd.DataFrame(data)
    
    basic_dataframe_creation()
    data_exploration(df)
    indexing_and_selection(df)
    data_manipulation(df)
    sorting_and_grouping(df)
    data_cleaning(df)
    data_analysis(df)
    data_visualization(df)
    time_series_example()
    merging_and_joining()
    advanced_operations(df)
    performance_analysis(df)
    
    print("\n" + "=" * 40)
    print("All pandas demonstrations completed!") 
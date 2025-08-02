import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('default')
sns.set_theme(style="whitegrid")

def basic_line_plots():
    # Create sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot multiple lines
    ax.plot(x, y1, 'b-', linewidth=2, label='sin(x)')
    ax.plot(x, y2, 'r--', linewidth=2, label='cos(x)')
    
    # Customize the plot
    ax.set_title('Sine and Cosine Functions', fontsize=14, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()

def scatter_plots():
    # Create sample data
    np.random.seed(42)
    x = np.random.randn(100)
    y = np.random.randn(100)
    colors = np.random.rand(100)
    sizes = np.random.randint(20, 200, 100)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Basic scatter plot
    scatter1 = ax1.scatter(x, y, s=50, c='blue', alpha=0.6)
    ax1.set_title('Basic Scatter Plot')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Scatter plot with color and size mapping
    scatter2 = ax2.scatter(x, y, s=sizes, c=colors, alpha=0.6, cmap='viridis')
    ax2.set_title('Scatter Plot with Color and Size Mapping')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # Add colorbar
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    plt.close()

def bar_plots():
    # Create sample data
    categories = ['A', 'B', 'C', 'D', 'E']
    values1 = [10, 15, 7, 12, 8]
    values2 = [8, 12, 9, 14, 6]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Basic bar plot
    bars1 = ax1.bar(categories, values1, color='skyblue', alpha=0.7)
    ax1.set_title('Basic Bar Plot')
    ax1.set_ylabel('Values')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom')
    
    # Grouped bar plot
    x = np.arange(len(categories))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, values1, width, label='Group 1', alpha=0.7)
    bars3 = ax2.bar(x + width/2, values2, width, label='Group 2', alpha=0.7)
    
    ax2.set_title('Grouped Bar Plot')
    ax2.set_ylabel('Values')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    plt.close()

def histogram_and_distributions():
    # Create sample data
    np.random.seed(42)
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(2, 1.5, 1000)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Basic histogram
    axes[0, 0].hist(data1, bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('Basic Histogram')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # Histogram with density
    axes[0, 1].hist(data1, bins=30, density=True, alpha=0.7, color='green')
    axes[0, 1].set_title('Histogram with Density')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Density')
    
    # Multiple histograms
    axes[1, 0].hist(data1, bins=30, alpha=0.7, label='Data 1', color='blue')
    axes[1, 0].hist(data2, bins=30, alpha=0.7, label='Data 2', color='red')
    axes[1, 0].set_title('Multiple Histograms')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Box plot
    axes[1, 1].boxplot([data1, data2], labels=['Data 1', 'Data 2'])
    axes[1, 1].set_title('Box Plot')
    axes[1, 1].set_ylabel('Value')
    
    plt.tight_layout()
    plt.show()
    plt.close()

def seaborn_statistical_plots():
    # Create sample dataset
    np.random.seed(42)
    n_samples = 200
    
    df = pd.DataFrame({
        'x': np.random.randn(n_samples),
        'y': np.random.randn(n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'value': np.random.randn(n_samples)
    })
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Scatter plot with regression
    sns.regplot(data=df, x='x', y='y', ax=axes[0, 0])
    axes[0, 0].set_title('Scatter Plot with Regression')
    
    # Box plot
    sns.boxplot(data=df, x='category', y='value', ax=axes[0, 1])
    axes[0, 1].set_title('Box Plot by Category')
    
    # Violin plot
    sns.violinplot(data=df, x='category', y='value', ax=axes[1, 0])
    axes[1, 0].set_title('Violin Plot by Category')
    
    # KDE plot
    sns.kdeplot(data=df['value'], ax=axes[1, 1])
    axes[1, 1].set_title('Kernel Density Estimation')
    
    plt.tight_layout()
    plt.show()
    plt.close()

def subplots_and_layout():
    # Create sample data
    x = np.linspace(0, 2*np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Line plot
    axes[0, 0].plot(x, y1, 'b-', linewidth=2)
    axes[0, 0].set_title('Sine Function')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[0, 1].scatter(x[::5], y2[::5], c='red', alpha=0.6)
    axes[0, 1].set_title('Cosine Function (Scatter)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bar plot
    categories = ['A', 'B', 'C', 'D']
    values = [10, 15, 7, 12]
    axes[1, 0].bar(categories, values, color='green', alpha=0.7)
    axes[1, 0].set_title('Bar Plot')
    
    # Histogram
    data = np.random.normal(0, 1, 1000)
    axes[1, 1].hist(data, bins=30, alpha=0.7, color='orange')
    axes[1, 1].set_title('Histogram')
    
    plt.tight_layout()
    plt.show()
    plt.close()

def heatmap_example():
    # Create correlation matrix
    np.random.seed(42)
    data = np.random.randn(100, 5)
    df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
    correlation_matrix = df.corr()
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.show()
    plt.close()

def advanced_customization():
    # Create sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.exp(-x/5)
    
    # Create figure with custom styling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with custom styling
    ax.plot(x, y, 'r-', linewidth=3, label='Damped Sine Wave')
    
    # Customize appearance
    ax.set_title('Advanced Customization Example', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    
    # Customize ticks
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(fontsize=12, loc='upper right')
    
    # Add text annotation
    ax.text(5, 0.5, 'Peak Amplitude', fontsize=10, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Add arrow annotation
    ax.annotate('Decay', xy=(3, 0.3), xytext=(6, 0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    plt.tight_layout()
    plt.show()
    plt.close()

def practical_data_visualization():
    # Create sample dataset (simulating sales data)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    
    # Simulate seasonal sales data
    trend = np.linspace(100, 150, 365)  # Upward trend
    seasonal = 20 * np.sin(2 * np.pi * np.arange(365) / 365)  # Seasonal component
    noise = np.random.normal(0, 5, 365)  # Random noise
    sales = trend + seasonal + noise
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'month': dates.month,
        'day_of_week': dates.dayofweek
    })
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series plot
    axes[0, 0].plot(df['date'], df['sales'], 'b-', alpha=0.7)
    axes[0, 0].set_title('Daily Sales Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Sales')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Monthly box plot
    monthly_data = [df[df['month'] == month]['sales'].values for month in range(1, 13)]
    axes[0, 1].boxplot(monthly_data, labels=[f'Month {i}' for i in range(1, 13)])
    axes[0, 1].set_title('Sales Distribution by Month')
    axes[0, 1].set_ylabel('Sales')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Day of week analysis
    dow_means = df.groupby('day_of_week')['sales'].mean()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    axes[1, 0].bar(day_names, dow_means.values, color='green', alpha=0.7)
    axes[1, 0].set_title('Average Sales by Day of Week')
    axes[1, 0].set_ylabel('Average Sales')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Sales distribution histogram
    axes[1, 1].hist(df['sales'], bins=30, alpha=0.7, color='orange', density=True)
    axes[1, 1].set_title('Sales Distribution')
    axes[1, 1].set_xlabel('Sales')
    axes[1, 1].set_ylabel('Density')
    
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    print("Plotting Examples and Demonstrations")
    print("=" * 40)
    
    basic_line_plots()
    scatter_plots()
    bar_plots()
    histogram_and_distributions()
    seaborn_statistical_plots()
    subplots_and_layout()
    heatmap_example()
    advanced_customization()
    practical_data_visualization()
    
    print("\n" + "=" * 40)
    print("All plotting demonstrations completed!") 
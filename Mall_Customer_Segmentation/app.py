import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import io

# Set page configuration
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🛍️ Mall Customer Segmentation</p>', unsafe_allow_html=True)
st.markdown("**Built with K-Means Clustering**")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Mall_Customers.csv')
    return df

df = load_data()

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Section 1: Dataset Overview
st.markdown('<p class="section-header">📊 Dataset Overview</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", len(df))
with col2:
    st.metric("Male Customers", len(df[df['Gender'] == 'Male']))
with col3:
    st.metric("Female Customers", len(df[df['Gender'] == 'Female']))
with col4:
    st.metric("Features", df.shape[1] - 1)

# Show dataset preview
with st.expander("📋 View Dataset Preview"):
    st.dataframe(df.head(10), use_container_width=True)

# Statistical summary
with st.expander("📈 Statistical Summary"):
    st.dataframe(df.describe(), use_container_width=True)

# Data info
with st.expander("ℹ️ Dataset Information"):
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

# Section 2: Feature Selection
st.markdown('<p class="section-header">🎯 Feature Selection for Clustering</p>', unsafe_allow_html=True)

st.info("💡 **Clustering will be performed on Annual Income and Spending Score**")

# Prepare data for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Section 3: Elbow Method
st.markdown('<p class="section-header">📉 Elbow Method Analysis</p>', unsafe_allow_html=True)

st.markdown("""
The **Elbow Method** helps determine the optimal number of clusters by plotting 
the inertia (within-cluster sum of squares) for different values of K.
""")

# Calculate inertia for different K values
@st.cache_data
def calculate_inertia(X, max_k=10):
    inertias = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    return list(K_range), inertias

K_range, inertias = calculate_inertia(X)

# Find optimal K using KneeLocator
kl = KneeLocator(K_range, inertias, curve='convex', direction='decreasing')
optimal_k = kl.elbow

col1, col2 = st.columns([2, 1])

with col1:
    # Plot elbow curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    
    if optimal_k:
        ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, 
                   label=f'Optimal K = {optimal_k}')
        ax.plot(optimal_k, inertias[optimal_k-1], 'ro', markersize=12)
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inertia', fontsize=12, fontweight='bold')
    ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("### 🎯 Recommended K")
    if optimal_k:
        st.success(f"**Optimal number of clusters: {optimal_k}**")
        st.markdown(f"""
        The elbow point suggests **{optimal_k} clusters** as the optimal choice.
        
        **Why {optimal_k}?**
        - Maximum variance explained
        - Minimal within-cluster variance
        - Clear elbow point detected
        """)
    else:
        st.warning("No clear elbow point detected. Try manual selection.")

# Section 4: K-Means Clustering
st.markdown('<p class="section-header">🔄 K-Means Clustering</p>', unsafe_allow_html=True)

# K value selector
st.sidebar.markdown("### 🎛️ Cluster Configuration")
k_value = st.sidebar.slider(
    "Select Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=optimal_k if optimal_k else 5,
    help="Adjust the slider to change the number of clusters"
)

st.sidebar.markdown("---")
if optimal_k:
    st.sidebar.info(f"💡 Suggested K: **{optimal_k}**")

# Perform K-Means clustering
@st.cache_data
def perform_clustering(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    return kmeans, clusters

kmeans, clusters = perform_clustering(X, k_value)

# Add cluster labels to dataframe
df_clustered = df.copy()
df_clustered['Cluster'] = clusters

# Section 5: Cluster Visualization
st.markdown('<p class="section-header">📊 Cluster Visualization</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    # Main scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define color palette
    colors = plt.cm.Set3(np.linspace(0, 1, k_value))
    
    # Plot each cluster
    for i in range(k_value):
        cluster_data = df_clustered[df_clustered['Cluster'] == i]
        ax.scatter(
            cluster_data['Annual Income (k$)'],
            cluster_data['Spending Score (1-100)'],
            c=[colors[i]],
            label=f'Cluster {i}',
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Plot centroids
    centroids = kmeans.cluster_centers_
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c='red',
        s=300,
        alpha=0.9,
        marker='*',
        edgecolors='black',
        linewidth=2,
        label='Centroids'
    )
    
    ax.set_xlabel('Annual Income (k$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spending Score (1-100)', fontsize=12, fontweight='bold')
    ax.set_title(f'Customer Segments - K={k_value}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("### 📍 Cluster Centers")
    centroid_df = pd.DataFrame(
        centroids,
        columns=['Income (k$)', 'Spending Score'],
        index=[f'Cluster {i}' for i in range(k_value)]
    )
    st.dataframe(centroid_df.round(2), use_container_width=True)
    
    st.markdown("### 👥 Cluster Sizes")
    cluster_sizes = df_clustered['Cluster'].value_counts().sort_index()
    size_df = pd.DataFrame({
        'Cluster': [f'Cluster {i}' for i in cluster_sizes.index],
        'Customers': cluster_sizes.values
    })
    st.dataframe(size_df, use_container_width=True)

# Section 6: Cluster Profiles
st.markdown('<p class="section-header">📋 Cluster Profiles</p>', unsafe_allow_html=True)

# Calculate cluster statistics
cluster_profile = df_clustered.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'CustomerID': 'count'
}).round(2)

cluster_profile.columns = ['Avg Age', 'Avg Income (k$)', 'Avg Spending Score', 'Customer Count']
cluster_profile.index = [f'Cluster {i}' for i in cluster_profile.index]

st.dataframe(cluster_profile, use_container_width=True)

# Cluster interpretation
st.markdown("### 💡 Cluster Insights")

insights_cols = st.columns(min(k_value, 3))
for i in range(k_value):
    with insights_cols[i % 3]:
        cluster_data = df_clustered[df_clustered['Cluster'] == i]
        avg_income = cluster_data['Annual Income (k$)'].mean()
        avg_spending = cluster_data['Spending Score (1-100)'].mean()
        
        # Categorize cluster
        if avg_income > 60 and avg_spending > 60:
            category = "🌟 High Value"
            description = "High income, high spending - Premium customers"
        elif avg_income < 40 and avg_spending < 40:
            category = "💤 Low Value"
            description = "Low income, low spending - Budget conscious"
        elif avg_income > 60 and avg_spending < 40:
            category = "💰 High Income, Low Spend"
            description = "Potential to convert to premium"
        elif avg_income < 40 and avg_spending > 60:
            category = "⚠️ Low Income, High Spend"
            description = "Risk group - overspending"
        else:
            category = "📊 Moderate"
            description = "Average income and spending"
        
        st.info(f"""
        **Cluster {i}: {category}**
        
        {description}
        
        - Size: {len(cluster_data)} customers
        - Avg Income: ${avg_income:.1f}k
        - Avg Spending: {avg_spending:.1f}/100
        """)

# Section 7: Gender Distribution per Cluster
st.markdown('<p class="section-header">👥 Gender Distribution by Cluster</p>', unsafe_allow_html=True)

# Calculate gender distribution
gender_dist = df_clustered.groupby(['Cluster', 'Gender']).size().unstack(fill_value=0)

col1, col2 = st.columns([2, 1])

with col1:
    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    gender_dist.plot(kind='bar', stacked=True, ax=ax, color=['#FF69B4', '#4169E1'])
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    ax.set_title('Gender Distribution Across Clusters', fontsize=14, fontweight='bold')
    ax.set_xticklabels([f'Cluster {i}' for i in range(k_value)], rotation=0)
    ax.legend(title='Gender', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("### 📊 Gender Breakdown")
    gender_pct = df_clustered.groupby('Cluster')['Gender'].value_counts(normalize=True).unstack(fill_value=0) * 100
    gender_pct.columns = ['Female %', 'Male %']
    gender_pct.index = [f'Cluster {i}' for i in gender_pct.index]
    st.dataframe(gender_pct.round(1), use_container_width=True)

# Section 8: Download Clustered Data
st.markdown('<p class="section-header">💾 Export Results</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    # Download clustered dataset
    csv = df_clustered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Clustered Dataset (CSV)",
        data=csv,
        file_name=f'mall_customers_clustered_k{k_value}.csv',
        mime='text/csv',
        help="Download the complete dataset with cluster assignments"
    )

with col2:
    # Download cluster profiles
    profile_csv = cluster_profile.to_csv().encode('utf-8')
    st.download_button(
        label="📊 Download Cluster Profiles (CSV)",
        data=profile_csv,
        file_name=f'cluster_profiles_k{k_value}.csv',
        mime='text/csv',
        help="Download summary statistics for each cluster"
    )

with col3:
    # Download gender distribution
    gender_csv = gender_dist.to_csv().encode('utf-8')
    st.download_button(
        label="👥 Download Gender Distribution (CSV)",
        data=gender_csv,
        file_name=f'gender_distribution_k{k_value}.csv',
        mime='text/csv',
        help="Download gender breakdown by cluster"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
    <p><strong>Mall Customer Segmentation App</strong></p>
    <p>Built with Streamlit • K-Means Clustering • Python</p>
</div>
""", unsafe_allow_html=True)

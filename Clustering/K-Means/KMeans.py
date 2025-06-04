# Save as app.py and run with: streamlit run app.py

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Interactive K-Means Clustering", layout="wide")

st.markdown("""
# 🚀 Interactive K-Means Clustering
Upload your CSV dataset, pick 2 or 3 features for clustering, and visualize in 2D and 3D!
""")

uploaded_file = st.file_uploader("📁 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📝 Dataset Preview", df.head())

    # Sidebar for clustering settings
    st.sidebar.header("🔧 Clustering Settings")
    selected_columns = st.sidebar.multiselect("Select 2 or 3 features for clustering", df.columns.tolist(), default=df.columns.tolist()[:3])

    if len(selected_columns) in [2, 3]:
        k = st.sidebar.slider("🔢 Number of clusters (K)", min_value=2, max_value=10, value=3)

        X = df[selected_columns]
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        st.success("✅ Clustering complete!")

        # Show dataset
        st.write("### 📊 Dataset with Clusters", df)

        # Always show 2D Visualization (user manually picks 2 features)
        st.markdown("### 🌈 2D Clusters Visualization")
        x_axis = st.selectbox("Select X-axis", selected_columns, index=0)
        y_axis = st.selectbox("Select Y-axis", [col for col in selected_columns if col != x_axis], index=0)

        fig_2d = px.scatter(df, x=x_axis, y=y_axis,
                             color=df['Cluster'].astype(str),
                             title="2D Cluster Visualization",
                             labels={'color': 'Cluster'},
                             color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig_2d, use_container_width=True)

        # 3D Visualization if 3 features
        if len(selected_columns) == 3:
            st.markdown("### 🌈 3D Clusters Visualization")
            fig_3d = px.scatter_3d(df, x=selected_columns[0], y=selected_columns[1], z=selected_columns[2],
                                   color=df['Cluster'].astype(str),
                                   title="3D Cluster Visualization",
                                   labels={'color': 'Cluster'},
                                   color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(fig_3d, use_container_width=True)

        # Predict for new customer
        st.markdown("### 🔍 Predict Cluster for a New Customer")
        new_data = []
        cols = st.columns(len(selected_columns))
        for i, col in enumerate(cols):
            value = col.number_input(f"Enter {selected_columns[i]}", float(df[selected_columns[i]].min()), float(df[selected_columns[i]].max()))
            new_data.append(value)

        if st.button("🔮 Predict Cluster"):
            cluster_pred = kmeans.predict([new_data])[0]
            st.success(f"🌟 The new customer belongs to Cluster: {cluster_pred}")

        # Download clustered dataset
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Clustered Dataset",
            data=csv,
            file_name='clustered_data.csv',
            mime='text/csv'
        )
    else:
        st.warning("⚠️ Please select exactly 2 or 3 features for clustering & prediction.")
else:
    st.info("👆 Upload a CSV file to start.")

st.markdown("""
---
Created with ❤️ by Your Mahesh Babu
""")

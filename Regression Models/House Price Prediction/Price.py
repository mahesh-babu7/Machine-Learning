import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load trained model
with open("HousePricePredictionModel.pkl", "rb") as file:
    model = pickle.load(file)

# Page configuration
st.set_page_config(page_title="🏠 House Price Wizard", layout="centered")
st.title("🏠 House Price Wizard")
st.markdown("✨ *Your magic window into real estate pricing!*")

# Sidebar info
with st.sidebar:
    st.header("🔍 About")
    st.write("""
        This app predicts house prices based on **square footage** using a Linear Regression model.
        Trained on real housing data.
    """)
    st.markdown("---")
    st.write("Built with ❤️ using Streamlit")

# Main interface
with st.expander("🔧 Enter House Details", expanded=True):
    sqft = st.slider("📏 Square Footage", min_value=100, max_value=10000, value=1500, step=50)

if st.button("🔮 Predict My House Price"):
    predicted_price = model.predict(np.array([[sqft]]))[0]
    st.success("✅ Prediction complete!")
    st.balloons()

    # Show price as a metric
    st.metric(label="💰 Estimated House Price", value=f"${predicted_price:,.2f}")

    # Visualization
    with st.expander("📊 Show Price Trends"):
        x_vals = np.linspace(0, 10000, 100).reshape(-1, 1)
        y_vals = model.predict(x_vals)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_vals, y_vals, color='blue', linewidth=2, label="Regression Line")
        ax.scatter(sqft, predicted_price, color='red', s=100, edgecolors='black', label="Your Estimate")
        ax.set_xlabel("Square Footage")
        ax.set_ylabel("Price")
        ax.set_title("📈 Price vs. Square Footage")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("🧙‍♂️ *Empowering real estate insights, one click at a time!*")

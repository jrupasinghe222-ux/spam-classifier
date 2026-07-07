import streamlit as st
from app.predictor import predict_message

st.set_page_config(page_title="Spam Message Classifier", page_icon="📩")

st.title("📩 Spam Message Classifier")
st.write("Enter a message below to check whether it is **Spam** or **Ham**.")

message = st.text_area("Message", height=200, placeholder="Type or paste your message here...")

if st.button("Predict"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        result = predict_message(message)

        prediction = result["prediction"]
        probability =  result["probability"]

        st.divider()
        st.subheader("Prediction")

        if prediction == "spam":
            st.error(f"🚨 Spam")
        else:
            st.success(f"✅ Ham")

        st.write(f"Probability: {probability:.2%}")

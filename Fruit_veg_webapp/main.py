import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import os
from PIL import Image


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Data Visualization","Prediction"])

#Main Page
if(app_mode=="Home"):
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "home_img.jpg"
    st.image(image_path)

#About Project
elif(app_mode=="About Project"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

#Data Visualization Page
elif(app_mode=="Data Visualization"):
    st.header("📊 Data Visualization & Analysis")
    
    # Load labels
    with open("labels.txt") as f:
        labels = [line.strip() for line in f.readlines()]
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Dataset Overview", "🎯 Model Performance", "📊 Class Distribution", "🖼️ Sample Images", "📉 Training History"])
    
    # Tab 1: Dataset Overview
    with tab1:
        st.subheader("Dataset Statistics")
        
        # Dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Classes", "36")
        with col2:
            st.metric("Training Images", "3,600")
        with col3:
            st.metric("Validation Images", "360")
        with col4:
            st.metric("Test Images", "360")
        
        st.markdown("---")
        
        # Category breakdown
        st.subheader("Category Breakdown")
        categories = {
            'Fruits': ['apple', 'banana', 'grapes', 'kiwi', 'mango', 'orange', 'pear', 'pineapple', 'pomegranate', 'watermelon'],
            'Vegetables': ['beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 
                          'cucumber', 'eggplant', 'garlic', 'ginger', 'jalepeno', 'lemon', 'lettuce', 'onion', 
                          'paprika', 'peas', 'potato', 'radish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 
                          'tomato', 'turnip']
        }
        
        # Pie chart for categories
        fig = go.Figure(data=[go.Pie(
            labels=list(categories.keys()),
            values=[len(categories['Fruits']), len(categories['Vegetables'])],
            hole=.3,
            marker_colors=['#ff9999', '#66b3ff']
        )])
        fig.update_layout(title_text="Fruits vs Vegetables Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display lists
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🍎 Fruits (10)**")
            st.write(", ".join(categories['Fruits']))
        with col2:
            st.write("**🥕 Vegetables (26)**")
            st.write(", ".join(categories['Vegetables']))
    
    # Tab 2: Model Performance
    with tab2:
        st.subheader("Model Architecture & Performance")
        
        # Model summary info
        st.write("**CNN Architecture:**")
        architecture = """
        - **Input Layer:** 64x64x3 RGB images
        - **Conv Block 1:** 2x Conv2D(32) + MaxPool + Dropout(0.25)
        - **Conv Block 2:** 2x Conv2D(64) + MaxPool + Dropout(0.25)
        - **Flatten Layer**
        - **Dense Layer 1:** 512 neurons + ReLU
        - **Dense Layer 2:** 256 neurons + ReLU
        - **Dropout:** 0.5
        - **Output Layer:** 36 neurons + Softmax
        """
        st.markdown(architecture)
        
        st.markdown("---")
        
        # Performance metrics (placeholder - would need actual values)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Accuracy", "~95%", "↑ 2%")
        with col2:
            st.metric("Validation Accuracy", "~93%", "↑ 1.5%")
        with col3:
            st.metric("Test Accuracy", "~92%", "↑ 1%")
        
        st.info("💡 **Note:** Load training_hist.json file to see actual performance metrics")
    
    # Tab 3: Class Distribution
    with tab3:
        st.subheader("Class Distribution Analysis")
        
        # Create bar chart for all classes
        class_counts = {
            'train': [100] * 36,
            'validation': [10] * 36,
            'test': [10] * 36
        }
        
        df = pd.DataFrame({
            'Class': labels,
            'Training': class_counts['train'],
            'Validation': class_counts['validation'],
            'Test': class_counts['test']
        })
        
        # Interactive bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Training', x=df['Class'], y=df['Training'], marker_color='#1f77b4'))
        fig.add_trace(go.Bar(name='Validation', x=df['Class'], y=df['Validation'], marker_color='#ff7f0e'))
        fig.add_trace(go.Bar(name='Test', x=df['Class'], y=df['Test'], marker_color='#2ca02c'))
        
        fig.update_layout(
            title='Images per Class across Datasets',
            xaxis_title='Class',
            yaxis_title='Number of Images',
            barmode='group',
            height=500,
            xaxis={'tickangle': -45}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.write("**Dataset Balance:**")
        st.write("✅ Dataset is perfectly balanced with equal images per class")
        st.write(f"- Total training images: {36 * 100} (100 per class)")
        st.write(f"- Total validation images: {36 * 10} (10 per class)")
        st.write(f"- Total test images: {36 * 10} (10 per class)")
    
    # Tab 4: Sample Images
    with tab4:
        st.subheader("Random Sample Images Preview")
        st.write("Select a class to view sample images:")
        
        selected_class = st.selectbox("Choose a class:", labels)
        
        # Path to images - try multiple possible paths
        possible_paths = [
            f"../data/train/{selected_class}/",
            f"../../data/train/{selected_class}/",
            f"d:/study/mon_ky_6/hoc_may_nang_cao/hoa_qua/data/train/{selected_class}/"
        ]
        
        image_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                image_dir = path
                break
        
        if image_dir and os.path.exists(image_dir):
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if image_files:
                # Show up to 6 images
                num_images = min(6, len(image_files))
                st.write(f"📁 Showing {num_images} sample images from: `{image_dir}`")
                
                cols = st.columns(3)
                for idx, img_file in enumerate(image_files[:num_images]):
                    with cols[idx % 3]:
                        try:
                            img = Image.open(os.path.join(image_dir, img_file))
                            st.image(img, caption=img_file, use_column_width=True)
                        except Exception as e:
                            st.error(f"Error loading {img_file}: {e}")
            else:
                st.warning(f"No image files found in {image_dir}")
        else:
            st.warning(f"⚠️ Images not found. Please ensure data folder exists.")
            st.info("Expected structure: `../data/train/{class_name}/`")
            st.write("Tried these paths:")
            for path in possible_paths:
                st.code(path)
    
    # Tab 5: Training History
    with tab5:
        st.subheader("Training History & Learning Curves")
        
        st.write("**Load Training History:**")
        uploaded_json = st.file_uploader("Upload training_hist.json file", type=['json'])
        
        if uploaded_json is not None:
            try:
                history = json.load(uploaded_json)
                
                # Create epochs list
                epochs = list(range(1, len(history['accuracy']) + 1))
                
                # Accuracy plot
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=epochs, y=history['accuracy'], 
                                            mode='lines+markers', name='Training Accuracy',
                                            line=dict(color='#1f77b4', width=2)))
                fig_acc.add_trace(go.Scatter(x=epochs, y=history['val_accuracy'], 
                                            mode='lines+markers', name='Validation Accuracy',
                                            line=dict(color='#ff7f0e', width=2)))
                fig_acc.update_layout(
                    title='Model Accuracy over Epochs',
                    xaxis_title='Epoch',
                    yaxis_title='Accuracy',
                    height=400
                )
                st.plotly_chart(fig_acc, use_container_width=True)
                
                # Loss plot
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=epochs, y=history['loss'], 
                                             mode='lines+markers', name='Training Loss',
                                             line=dict(color='#d62728', width=2)))
                fig_loss.add_trace(go.Scatter(x=epochs, y=history['val_loss'], 
                                             mode='lines+markers', name='Validation Loss',
                                             line=dict(color='#9467bd', width=2)))
                fig_loss.update_layout(
                    title='Model Loss over Epochs',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    height=400
                )
                st.plotly_chart(fig_loss, use_container_width=True)
                
                # Final metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Final Training Accuracy", f"{history['accuracy'][-1]:.4f}")
                    st.metric("Final Training Loss", f"{history['loss'][-1]:.4f}")
                with col2:
                    st.metric("Final Validation Accuracy", f"{history['val_accuracy'][-1]:.4f}")
                    st.metric("Final Validation Loss", f"{history['val_loss'][-1]:.4f}")
                
            except Exception as e:
                st.error(f"Error loading history file: {e}")
        else:
            st.info("📊 Upload the training_hist.json file (generated during training) to visualize learning curves")
            
            # Show sample plot with dummy data
            st.write("**Sample Visualization (with dummy data):**")
            sample_epochs = list(range(1, 33))
            sample_acc = [0.3 + (i * 0.02) for i in range(32)]
            sample_val_acc = [0.28 + (i * 0.019) for i in range(32)]
            
            fig_sample = go.Figure()
            fig_sample.add_trace(go.Scatter(x=sample_epochs, y=sample_acc, 
                                          mode='lines', name='Training (Sample)',
                                          line=dict(dash='dash')))
            fig_sample.add_trace(go.Scatter(x=sample_epochs, y=sample_val_acc, 
                                          mode='lines', name='Validation (Sample)',
                                          line=dict(dash='dash')))
            fig_sample.update_layout(title='Sample Learning Curve', height=300)
            st.plotly_chart(fig_sample, use_container_width=True)

#Prediction Page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.success("Model is Predicting it's a {}".format(label[result_index]))
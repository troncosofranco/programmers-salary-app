#Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle



#Loading model from pickle file
def load_model():
    with open("saved_steps.pkl", "rb") as file:
        data = pickle.load(file)
    return data

data =  load_model()

#Encoding categorical data from model
regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

#Binning countries categories
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

#Binning education categories
def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    elif 'Master’s degree' in x:
        return 'Master’s degree'
    elif 'Professional degree' in x or 'Other doctoral degree' in x:
        return 'Post grad'
    else:
        return 'Less than Bachelor’s degree'
    
#Binning experience categories
def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

#@st.cache -> data remains in cache
@st.cache
#Load and clean data based on previous functions
def load_data():
    df = pd.read_csv("programers_salary_2021.zip", compression="zip")	
    df= df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedCompYearly"]] #Target data
    df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)

    df = df[df["Salary"].notnull()]
    df = df.dropna()
    df = df[df["Employment"] == "Employed full-time"]
    df = df.drop("Employment", axis=1)
    
    country_map = shorten_categories(df.Country.value_counts(), 400)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["Salary"] <= 250000]
    df = df[df["Salary"] >= 10000]
    df = df[df["Country"] != "Other"]

    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)
    df = df.rename({"ConvertedComp": "Salary"}, axis=1)
    return df

df = load_data()

#data input function
def show_explore_page():
    st.title("Software Developer Salary Explorer")
    st.write("This app predicts the **Software Developer Salary** based on Stack Overflow survey data.")

    video_file = open('logo.mp4', 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes, start_time=0)

    data = df['Country'].value_counts() 
    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.0f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.subheader("Data by country")
    
    
    st.pyplot(fig1)
    st.write(
        """
    #### Mean Salary Based On Country (U$D)
    """
    )
    
    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.subheader("Mean Salary (U$D) Vs Experience Years (All countries)")

    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)

#Selectors vectors
country = ('Sweden', 'Spain', 'Germany', 'Turkey', 'Canada', 'France',
       'Switzerland','United Kingdom of Great Britain and Northern Ireland',
       'Russian Federation', 'Israel', 'United States of America',
       'Brazil', 'Italy', 'Netherlands', 'Poland', 'Australia', 'India',
       'Norway')

education = ('Less than Bachelor’s degree','Bachelor’s degree', 'Master’s degree', 'Post grad')



#side buttons
st.sidebar.title("Input parameters")

country_sel = st.sidebar.selectbox('Country:', country)
education_sel = st.sidebar.selectbox('Education level:', education)
experience_sel = st.sidebar.slider('Years of experience:', 1, 40, 1)

ok = st.sidebar.selectbox('explore or predict salary:', ('explore', 'predict'))
if ok == 'predict':
    X = np.array([[country_sel, education_sel, experience_sel]])
    X[:, 0] = le_country.transform(X[:,0])
    X[:, 1] = le_education.transform(X[:,1])
    X = X.astype(float)

    salary = regressor.predict(X) #numpy array
    st.title("Software Developer Salary Prediction App")
    st.subheader(f'Your annual salary is ${salary[0]:.2f}') #two decimals
else: 
    show_explore_page()
    

st.write("---")

st.markdown("""
Credits:
* **Data Provided by:** [Stack overflow Survey 2021](https://insights.stackoverflow.com/survey/2021)
* **Code contribution:** [Python Engineer] (https://github.com/python-engineer/python-fun)
""")

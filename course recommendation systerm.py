import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt

# -------------------------
# SIMPLE CUSTOM AUTHENTICATION
# -------------------------

# Demo credentials
users = {
    "alice": "123",
    "bob": "456"
}

# Create login state if not exists
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# If not logged in, show login form
if not st.session_state.logged_in:
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()  # Stop loading dashboard until login is done
    


# ---------------------------------------------------------
# MAIN APPLICATION (NO AUTHENTICATION)
# ---------------------------------------------------------

st.title("Course Recommendation System")

# -------------------------
# LOAD DATASET
# -------------------------
df = pd.read_excel(
    r"C:\Users\shrey\Desktop\instructo Wipro\Project\archive\educaion_database.xlsx"
)
df.columns = [c.strip().replace('\n', ' ') for c in df.columns]

if 'title' not in df.columns:
    st.error("Dataset missing required column: title")
    st.stop()

st.subheader('Browse & Search Courses')

# Search bar
subj = st.text_input('Enter preferred subject or tag:')

if subj:
    filt = (
        df['subject'].str.contains(subj, case=False, na=False) |
        df['tags'].fillna('').str.contains(subj, case=False, na=False)
    )
    st.write(df[filt][['title', 'subject', 'difficulty', 'tags']])

st.markdown("---")
st.subheader('Personalized Recommendations')

# -------------------------
# CONTENT-BASED FILTERING
# -------------------------
selected_title = st.selectbox(
    'Select a course you liked:', df['title'].drop_duplicates()
)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(
    df['tags'].fillna('') + ' ' + df['subject'].fillna('')
)

idx = df.index[df['title'] == selected_title][0]

cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
sim_scores = list(enumerate(cosine_sim))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

rec_indices = [i for i, score in sim_scores[1:6]]

st.write(df.iloc[rec_indices][['title', 'subject', 'tags']])

# -------------------------
# SIMPLE FAKE RECOMMENDATION FOR DEMO
# -------------------------
st.subheader(" Collaborative Recommendations")

demo_users = ['alice', 'bob', 'charlie']
np.random.seed(42)

ratings = pd.DataFrame({
    'user': np.random.choice(demo_users, size=100),
    'title': np.random.choice(df['title'], size=100),
    'rating': np.random.randint(1, 6, size=100)
})

# Choose user manually now that authentication is removed
username = st.selectbox("Choose a demo user:", demo_users)

user_ratings = ratings[ratings['user'] == username]['title'].values
unseen = df[~df.title.isin(user_ratings)]

unseen['pred_score'] = np.random.random(len(unseen))

top5 = unseen.sort_values('pred_score', ascending=False).head(5)

st.write(top5[['title', 'subject', 'pred_score']])

# -------------------------
# VISUALIZATIONS
# -------------------------
st.subheader("Course Subject Distribution")
fig, ax = plt.subplots()
df['subject'].value_counts().plot(kind='bar', ax=ax)
st.pyplot(fig)

st.subheader("Difficulty Pie Chart")
fig2, ax2 = plt.subplots()
df['difficulty'].value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%')
st.pyplot(fig2)

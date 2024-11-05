import streamlit as st
import pymongo


conn = st.connection('postgresql', type='sql')
df = conn.query('SELECT * FROM news_count', ttl=600)

st.write(df)


st.divider()

@st.cache_resource
def init_connection():
    return pymongo.MongoClient(**st.secrets["mongo"])

client = init_connection()

@st.cache_data(ttl=600)
def get_data():
    db = client.news
    items = db.news_id.find()
    items = list(items)
    return items

st.write(get_data())
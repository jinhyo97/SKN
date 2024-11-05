import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime


# st.header('9. 파이썬 모듈')
# st.subheader('9.1 모듈과 import', anchor='6e9363cc')
# st.subheader('9.2 표준 모듈의 활용', anchor='6e9363cc')
# st.subheader('9.3 외부 모듈의 활용', anchor='6e9363cc')

# st.header('10. 파이썬 파일의 입출력')
# st.subheader('10.1 파일 입출력의 개요', anchor='fa732dff')
# st.subheader('10.2 파일 출력(output)', anchor='fa732dff')
# st.subheader('10.3 파일 입력(input)', anchor='fa732dff')

# st.divider()


# if st.button('click here'):
#     st.write('hahaha')
# else:
#     st.write('hohoho')


# st.download_button(
#     'download',
#     '1234',
#     file_name='dafult.txt'
# )

# st.link_button(
#     'naver',
#     url='https://www.naver.com',
# )

# st.checkbox('is_decimal')

# check = st.toggle('is_decimal')
# if check:
#     st.write('option is selected')

# option = st.radio(
#     'options',
#     ['apple', 'banana', 'kiwi'],
#     index=0,
# )


# option = st.selectbox(
#     'options',
#     ['apple', 'banana', 'kiwi'],
#     index=0,
# )
# st.write(option)

# option = st.multiselect(
#     'options',
#     ['apple', 'banana', 'kiwi'],
#     default=['apple'],
# )
# st.write(option)
# # st.write(option[1])

# color = st.select_slider(
#     'select color',
#     options=[
#         'red',
#         'orange',
#         'yellow',
#         'green',
#         'blue',
#         'indigo',
#         'violet',
#     ]
# )
# st.write(f'color: {color}')

# user_input_number = st.number_input(
#     'input number'
# )
# st.write(user_input_number)


# age = st.slider(
#     'select number in a given range',
#     0,
#     100,
#     50,
# )
# st.write(age)

# date = st.date_input(
#     'input_date',
#     datetime.date(2024, 8, 12),
# )
# st.write(date)

col1, col2 = st.columns([0.3, 0.7])

with col1:
    st.header('Col1')
    st.subheader('Col1-1')

with col2:
    st.header('Col2')
    st.subheader('Col2-1')

    col2_1, col2_2 = st.columns(2)

    with col2_1:
        st.header('Col2-1')
        st.subheader('Col2-2')

    with col2_2:
        st.header('Col2-2')
        st.subheader('Col2-2')

with st.container(border=1):
    st.write('hello world')
    st.write('hello world')
    st.write('hello world')
    st.write('hello world')
    st.write('hello world')
    st.write('hello world')
    st.write('hello world')

with st.popover('click here'):
    st.write('hello world')
    st.write('hello world')
    st.write('hello world')
    st.write('hello world')


tab1, tab2 = st.tabs(['apple', 'banana'])

with tab1:
    st.write('tab1')
    st.write('tab1')
with tab2:
    st.write('tab2')
    st.write('tab2')
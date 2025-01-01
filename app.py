from langgraph.prebuilt import create_react_agent

from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI
from typing import List

from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.documents import Document


from typing import List, Optional
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph, START
from langchain_core.messages import HumanMessage, trim_messages

import functools

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough

from typing import Literal
    
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai.chat_models import ChatOpenAI


from typing import Annotated, List, Dict
from langchain_core.tools import tool


from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional

# from langchain_experimental.utilities import PythonREPL
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, trim_messages, AIMessage

from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

from langchain.vectorstores import Chroma
import plotly.graph_objects as go

openai_api_key="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"

open_ai_small_path = "C:/Users/HP/Desktop/streamlit_trial/deneme_files/chroma/"
embed_model = OpenAIEmbeddings(openai_api_key="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", model="text-embedding-3-small")   


import streamlit as st
import pandas as pd


from langgraph.graph.message import add_messages
from typing import Annotated, Literal, Sequence, TypedDict

from listing_graph import listing_graph, parent_config


# inp = {"words":["mama sweatshrt"],
# "listing_count":2,
# "character_limit":125,
# "specified_criterias": "None",
# "undesired_words":"None",
# "desired_words":"mom",
# "product_name":"embroidered sweatshirt"}
st.set_page_config(layout="wide")


st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 250px;
    }
    .stTextInput, .stFileUploader {
        font-size: 10px;
        height: 75px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Amazon Listing Generator")

# options = ["Default"]

# options_ = st.sidebar.selectbox("please select a listing strategy:", options)
uploaded_file = st.sidebar.file_uploader("upload your hellium 10 keyword excel file", type=["xlsx", "xls"])
product_name = st.sidebar.text_input("please enter your product name",value="embroidered sweatshirt")
title_chracter_limit = st.sidebar.number_input("please enter the title character limit",value=125)
listing_count = st.sidebar.number_input("please enter desired listing count", value=3)
desired_words = st.sidebar.text_input("please enter your comma seperated specificly desired words", value="mom")
undesired_words = st.sidebar.text_input("please enter your comma seperated specificly undesired words", value="cozy")

# listing_count = 1

if uploaded_file is not None:
    try:
        kws = pd.read_excel(uploaded_file)
        # st.success("Dosya başarıyla yüklendi!")
        
        # DataFrame'i göster
        # st.dataframe(df)
    except Exception as e:
        st.error(f"Dosya okunurken bir hata oluştu: {e}")

# else:
#     st.info("Lütfen bir Excel dosyası yükleyin.")
# import numpy as np
# data = {
#     'A': np.random.randint(1, 100, 20),
#     'B': np.random.randint(1, 100, 20),
#     'C': np.random.randint(1, 100, 20),
#     'D': np.random.randint(1, 100, 20),
#     'E': np.random.randint(1, 100, 20)
# }

from io import BytesIO

# @st.cache_data
def load_data():
    # df = pd.DataFrame(data)
    return kws


def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data


# @st.cache_data
def plot_bar(x,y):
    fig = go.Figure(data=[go.Bar(x=x, y=y)])

    # Başlık ve etiketler ekleme
    fig.update_layout(
        title='Search Volume Barplot',
        xaxis_title='titles',
        yaxis_title='Seach Volume')
    fig.update_layout(width=600, height=400)
    return fig




if st.button("generate"):
    # Tüm gerekli girdilerin sağlandığını kontrol edin
    if uploaded_file and desired_words and undesired_words and title_chracter_limit and product_name:
        # if 'df' not in st.session_state:
        #     st.session_state.df = load_data()
            
            # st.write(st.session_state.df)
        # df = st.session_state.df

        # if 'fig' not in st.session_state:
        #     df = st.session_state.df
        #     st.session_state.fig = plot_bar(df["A"].tolist(), df["B"].tolist())
        words = kws.sort_values("Search Volume",ascending=False)["Keyword Phrase"].tolist()[:10]
        sv_scores = {i:j for i,j in kws[["Keyword Phrase","Search Volume"]].values}

        inp = {"words":words,
        "listing_count":listing_count,
        "character_limit":title_chracter_limit,
        "specified_criterias": "None",
        "undesired_words":undesired_words,
        "desired_words":desired_words,
        "product_name":product_name}
            
        listing = listing_graph.invoke({**inp}, config=parent_config)


        chunk_size = 6
        chunks = [listing["desc_list"][i:i + chunk_size] for i in range(0, len(listing), chunk_size)]
        df_list = []
        for i in range(listing_count):
            df_list.append(pd.DataFrame([listing["title_list"][i].content] + [a.content for a in chunks[i]]).T)

        final_df = pd.concat(df_list)

        final_df.columns = ["title"] + ["Bullet" + str(i) for i in range(1,6)] + ["Description"]


        title_scores = {}
        for title in [i.lower() for i in final_df["title"].tolist()]:
            for word, score in sv_scores.items():
                if title not in title_scores.keys() and word in title:
                    title_scores[title] = score
                    continue
                if  word in title:
                    title_scores[title] += score

        if 'fig' not in st.session_state:
            st.session_state.fig = plot_bar(["title: " + str(i) for i in range(1, len(title_scores)+1)], list(title_scores.values()))
        if 'df' not in st.session_state:
            st.session_state.df = final_df
            
        # df = st.session_state.df
        # fig = st.session_state.fig

        left_column, right_column = st.columns([3, 3])
        with left_column:
            # st.header("Generated Listings")
            # st.dataframe(final_df)
            st.dataframe(final_df, width=2000)

            # Excel indirme butonu
            st.download_button(
                label="Download Excel",
                data=convert_df_to_excel(final_df),
                file_name='listings.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )


        with right_column:
            st.header("Graphs")
            fig = plot_bar(["title: " + str(i) for i in range(1,len(title_scores)+1)], list(title_scores.values()))
            st.plotly_chart(fig, use_container_width=True)
            # fig = plot_bar(df["A"].tolist(), df["B"].tolist())
            # st.plotly_chart(fig, use_container_width=True)
        

        # st.success("Processing complete!")
    else:
        st.error("Please fill in all fields before submitting.")
















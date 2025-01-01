
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from io import BytesIO
from listing_graph import listing_graph, parent_config

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

openai_api_key = st.sidebar.text_input("please enter your open ai api key")
uploaded_file = st.sidebar.file_uploader("upload your hellium 10 keyword excel file", type=["xlsx", "xls"])
product_name = st.sidebar.text_input("please enter your product name",value="embroidered sweatshirt")
title_chracter_limit = st.sidebar.number_input("please enter the title character limit",value=125)
listing_count = st.sidebar.number_input("please enter desired listing count", value=3)
desired_words = st.sidebar.text_input("please enter your comma seperated specificly desired words", value="mom")
undesired_words = st.sidebar.text_input("please enter your comma seperated specificly undesired words", value="cozy")

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = openai_api_key

#openai_api_key = st.session_state["openai_api_key"]
print("openai_api_key: ", openai_api_key)

if uploaded_file is not None:
    try:
        kws = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Dosya okunurken bir hata oluştu: {e}")



if st.button("generate") and openai_api_key != "":
    print("controlll")
    if uploaded_file and desired_words and undesired_words and title_chracter_limit and product_name and openai_api_key:

        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
        words = kws.sort_values("Search Volume", ascending=False)["Keyword Phrase"].tolist()[:10]
        sv_scores = {i: j for i, j in kws[["Keyword Phrase", "Search Volume"]].values}

        inp = {"llm":llm,
                "words": words,
               "listing_count": listing_count,
               "character_limit": title_chracter_limit,
               "specified_criterias": "None",
               "undesired_words": undesired_words,
               "desired_words": desired_words,
               "product_name": product_name}

        listing = listing_graph.invoke({**inp}, config=parent_config)

        chunk_size = 6
        chunks = [listing["desc_list"][i:i + chunk_size] for i in range(0, len(listing), chunk_size)]
        print("lennn", len(chunks))
        print(chunks)

        df_list = []
        for i in range(listing_count):
            df_list.append(pd.DataFrame([listing["title_list"][i].content] + [a.content for a in chunks[i]]).T)

        final_df = pd.concat(df_list)

        final_df.columns = ["title"] + ["Bullet" + str(i) for i in range(1, 6)] + ["Description"]

        title_scores = {}
        for title in [i.lower() for i in final_df["title"].tolist()]:
            for word, score in sv_scores.items():
                if title not in title_scores.keys() and word in title:
                    title_scores[title] = score
                    continue
                if word in title:
                    title_scores[title] += score

        if 'fig' not in st.session_state:
            st.session_state.fig = plot_bar(["title: " + str(i) for i in range(1, len(title_scores) + 1)],
                                            list(title_scores.values()))
        if 'df' not in st.session_state:
            st.session_state.df = final_df

        left_column, right_column = st.columns([3, 3])
        with left_column:
            st.dataframe(final_df, width=2000)

            st.download_button(
                label="Download Excel",
                data=convert_df_to_excel(final_df),
                file_name='listings.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

        with right_column:
            st.header("Graphs")
            fig = plot_bar(["title: " + str(i) for i in range(1, len(title_scores) + 1)], list(title_scores.values()))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Please fill in all fields before submitting.")




elif openai_api_key == "":
    st.info("Please enter verified open ai api key")























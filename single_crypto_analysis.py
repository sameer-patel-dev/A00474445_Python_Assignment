import streamlit as st
from utils import getListofAllIDs, get_historical_data
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd


def show_page():
    st.header("Single Stock Analysis")

    coins_list = getListofAllIDs()
    if not coins_list.empty:
        coin_names = coins_list['id'].sort_values().tolist()
        selected_coin_name = st.selectbox("Select Stock", [""] + coin_names)
        
        if selected_coin_name:
            one_year_ago_date = datetime.now() - relativedelta(years=1)
            start_date_str = one_year_ago_date.strftime("%Y-%m-%d")
            end_date_str = datetime.now().strftime("%Y-%m-%d")
            df = get_historical_data(selected_coin_name, start_date_str, end_date_str)
            
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                max_price = df["price"].max()
                min_price = df["price"].min()
                max_date = df[df["price"] == max_price].index[0].strftime('%Y-%m-%d')
                min_date = df[df["price"] == min_price].index[0].strftime('%Y-%m-%d')


                df.index = df.index.strftime('%Y-%m-%d')
                st.line_chart(df["price"])
                st.write(f"The highest trading price was on {max_date} with a price of CAD {max_price}.")
                st.write(f"The lowest trading price was on {min_date} with a price of CAD {min_price}.")
    
    else:
        st.error("Could not load the coins list.")


if __name__ == "__main__":
    show_page()

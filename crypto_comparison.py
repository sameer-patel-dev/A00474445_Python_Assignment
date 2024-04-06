import streamlit as st
import matplotlib.pyplot as plt
from utils import getListofAllIDs, get_historical_data
from datetime import datetime, timedelta
import pandas as pd


def show_page():
    st.header("Stock Comparison")
    coins_list = getListofAllIDs()
    if not coins_list.empty:
        coin_names = coins_list['id'].sort_values().tolist()
        
        selected_coin1 = st.selectbox("Select Stock 1", [""] + coin_names)
        selected_coin2 = st.selectbox("Second Stock 2", [""] + coin_names)
        start_date = st.date_input("Start date", value=datetime.now() - timedelta(days=365))
        end_date = st.date_input("End date", value=datetime.now())

        if selected_coin1 and selected_coin2 and start_date < end_date:

            if selected_coin1 != selected_coin2:

                start_date_str = start_date.strftime("%Y-%m-%d")
                end_date_str = end_date.strftime("%Y-%m-%d")
                df1 = get_historical_data(selected_coin1, start_date_str, end_date_str)
                df2 = get_historical_data(selected_coin2, start_date_str, end_date_str)

                df1.rename(columns={'price': selected_coin1}, inplace=True)
                df2.rename(columns={'price': selected_coin2}, inplace=True)
                df = df1[[selected_coin1]].join(df2[[selected_coin2]], how='outer')
                df.index = pd.to_datetime(df.index)
                df.index = df.index.strftime('%Y-%m-%d')

                if not df.empty:
                    st.line_chart(df)

                else:
                    st.write("No data available for the selected coins or date range.")

            else:
                st.error("Ensure that both cryptocurrencies are not same")
        
        else:
            st.write("")
            st.write("")
            st.write("")
            st.info("Ensure that both cryptocurrencies are selected and the start date is before the end date.")
    
    else:
        st.error("Could not load the coins list.")


if __name__ == "__main__":
    show_page()

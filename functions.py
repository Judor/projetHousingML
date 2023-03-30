import base64
import io

import pandas as pd
import streamlit as st
import requests
import os
import json
import datetime


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), [' ', ' K', ' M', ' B', ' T'][magnitude])


def restrict_dataframe(df, start_range, end_range):
    df = df[(df['day'] >= pd.Timestamp(start_range)) & (df['day'] <= pd.Timestamp(end_range))]
    return df


def restrict_all_dataframes(df, start_range, end_range):
    df = restrict_dataframe(df, start_range, end_range)
    dfDB = getDFDB()
    df_sum_by_day = render_df_sum_by_day(df)
    df_sum_by_day = restrict_dataframe(df_sum_by_day, start_range, end_range)
    dfSumByDevice = render_df_sum_by_device(df)
    dfLowestCo2PerClick = render_df_lowest_co2_per_click(df)
    dfLowestCo2PerImp = render_df_lowest_co2_per_imp(df)
    dfSumBySiteDomain = render_df_sum_by_site_domain(df, dfDB)
    imps= df['imps'].sum()
    return df, df_sum_by_day, dfSumByDevice, dfLowestCo2PerClick, dfLowestCo2PerImp, dfSumBySiteDomain,imps

def human_formatCO2(co2):
    num = float('{:.3g}'.format(co2))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), [' g', ' Kg', ' T'][magnitude])


def get_dataframe(url):
    file_id = url.split('/')[-2]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url, stream=True)

    if response.status_code == 200:
        return pd.read_csv(response.raw)
    else:
        print("Error downloading the file.")
        return None


def load_inputs():
    if os.path.exists("inputs.json"):
        with open("inputs.json", "r") as file:
            data = json.load(file)
            inputs = {
                "CAMPAIGN_NAME": data['CAMPAIGN_NAME'],
                "CAMPAIGN_BUDGET": data['CAMPAIGN_BUDGET'],
                "MAIN_CTA": data['MAIN_CTA'],
                "AD_SIZE_IN_BYTES": data['AD_SIZE_IN_BYTES']
            }
            # Update the session state with the loaded inputs
            st.session_state.inputs = inputs
            return inputs
    else:
        inputs = {
            "CAMPAIGN_NAME": "Jellyfish X LeroyMerlin",
            "CAMPAIGN_BUDGET": 40000,
            "MAIN_CTA": "Click",
            "AD_SIZE_IN_BYTES": 3145728
        }
        # Update the session state with the default inputs
        st.session_state.inputs = inputs
        return inputs


def save_inputs(inputs):
    with open("inputs.json", "w") as file:
        json.dump(inputs, file)


def getCO2perByteDependingOnDevice(DeviceType, AdsSize, VideoCompletionRate, Imps):
    CARBON_INTENSITY_FR = 110
    adTotalSize = AdsSize * VideoCompletionRate * Imps
    dataCenterKWhPerByte = 9.41 * 10 ** -11
    dataCenterEnergy = dataCenterKWhPerByte * adTotalSize
    CellularkWhPerByte = 0.1 / 1073741824
    FixedkWhPerByte = 0.0315 / 1073741824

    fixed_mask = DeviceType.isin(["desktops & laptops", "media players", "set top box", "tv"])
    mobile_mask = DeviceType.isin(["mobile phones", "tablets"])

    DeviceTransmissionEnergy = pd.Series(FixedkWhPerByte * adTotalSize, index=DeviceType.index)
    DeviceTransmissionEnergy[mobile_mask] = 0.7 * CellularkWhPerByte * adTotalSize[
        mobile_mask] + 0.3 * FixedkWhPerByte * adTotalSize[mobile_mask]

    totalEnergyUsed = dataCenterEnergy + DeviceTransmissionEnergy
    return totalEnergyUsed * CARBON_INTENSITY_FR / 1000


def render_calendar(col, start_date, end_date):
    # Display a date input widget with a predefined date range
    selected_date_range = col.date_input("Filter the datas within this range:", [start_date, end_date],
                                         min_value=start_date, max_value=end_date)

    if selected_date_range:
        # If it's not empty, unpack it into start_range and end_range
        if isinstance(selected_date_range[0], datetime.date):
            # If it's a single date, set start_range and end_range to the same value
            start_range = selected_date_range[0]
            end_range = selected_date_range[0] if len(selected_date_range) == 1 else selected_date_range[1]
        else:
            # If it's a date range, unpack it into start_range and end_range
            start_range, end_range = selected_date_range


        return start_range, end_range
    else:
        col.warning("Please select a date range.")


def get_grade_from_number(number):
    if number > 80:
        return "A"
    elif number > 60:
        return "B"
    elif number > 40:
        return "C"
    elif number > 20:
        return "D"
    else:
        return "E"


def renderIndicators(df,imps):
    # Setting variables
    # Sum of various columns
    clicks = df["clicks"].sum()
    viewed_imps = df["viewed_imps"].sum()
    # Calculate percentage of viewable impressions
    percentage_viewed_imps = int(round(viewed_imps / imps * 100, 1))
    percentage_viewed_imps_str = f"{percentage_viewed_imps} % of viewable impressions"

    # Calculate CO2 total and budget paid
    co2_total = df["co2Total"].sum() * 1000
    budget_paid = df["curator_total_cost"].sum()

    # Calculate percentage of budget paid
    percentage_budget_paid = int(round(budget_paid / st.session_state.inputs['CAMPAIGN_BUDGET'] * 100, 1))
    percentage_budget_paid_str = f"{percentage_budget_paid} % of the allocated budget"

    # Calculate CO2 per main CTA
    co2_per_cta = co2_total / clicks
    co2_per_cta_str = f"{round(co2_per_cta, 1)} gCO2e per {st.session_state.inputs['MAIN_CTA']}"

    # Calculate CO2 wasted
    co2_wasted = round(co2_total - (df["Viewability Rate %"] * df["co2Total"]).sum() * 1000, 2)
    co2_wasted_percentage = round(((co2_wasted / co2_total) * 100), 1)
    co2_wasted_str = f"{co2_wasted_percentage} % of total CO2e"
    return clicks, viewed_imps, percentage_viewed_imps_str, co2_total, budget_paid, percentage_budget_paid_str, co2_per_cta_str, co2_wasted, co2_wasted_str


def read_css_file(file_path):
    """Read the contents of a CSS file."""
    with open(file_path) as css_file:
        return css_file.read()

def convert_image_to_base64(image):
    """Convert an image file to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def display_image_in_sidebar(image_base64, width, margin_bottom):
    """Display an image in the Streamlit sidebar with custom dimensions."""
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{image_base64}" style="width:{width}px; margin-bottom:{margin_bottom}px" />',
        unsafe_allow_html=True,
    )


def load_dataframes():
    # TODO optimize link
    # df = get_dataframe("https://drive.google.com/file/d/1a5NC05bBcERjV_hDqcqKC1awE1O9-AoJ/view?usp=share_link")
    df = pd.read_csv('Jellyfish.csv')
    df=df[["day","site_domain","device_type","imps","vie",'clicks',"curator_total_cost","video_completion_rate"]]
    # don't take the last 2 days of the campaign
    # df = df[df['day'] != '2023-03-03']
    df = df[df['day'] != '2023-03-02']
    df = df[df['day'] != '2023-03-01']
    df_db = pd.read_csv("assets/meanDomain.csv")
    return df, df_db





def preprocess_data(df, ad_size):
    df['day'] = pd.to_datetime(df['day'], format='%Y-%m-%d')
    df["co2Total"] = getCO2perByteDependingOnDevice(df['device_type'], ad_size, df["video_completion_rate"], df["imps"])
    df["co2_wasted"] = df["co2Total"] - (df["Viewability Rate %"] * df["co2Total"])
    return df


# Create a dataframe with the sum of co2 per day
def render_df_sum_by_day(df):
    df_sum_by_day = df.groupby('day').sum(numeric_only=True)
    df_sum_by_day.reset_index(inplace=True)
    df_sum_by_day['day'] = pd.to_datetime(df_sum_by_day['day'], format='%Y-%m-%d')
    df_sum_by_day = df_sum_by_day.sort_values(by='day')
    df_sum_by_day["co2PerImp"] = df_sum_by_day["co2Total"] / df_sum_by_day["imps"] * 1000
    df_sum_by_day["co2PerClick"] = df_sum_by_day["co2Total"] / df_sum_by_day["clicks"] * 1000
    df_sum_by_day["video_completions"] = df_sum_by_day["imps"] / df_sum_by_day["video_completion_rate"]
    df_sum_by_day["co2PerVideoCompletion"] = df_sum_by_day["co2Total"] / df_sum_by_day["video_completions"] * 1000
    df_sum_by_day["co2Per€Spent"] = df_sum_by_day["co2Total"] / df_sum_by_day["curator_total_cost"] * 1000
    return df_sum_by_day


# Creating a dataframe containing the 90% in terms of Impressions

def render_df_imps(df):
    df_imps = df.groupby('site_domain', as_index=False).sum(numeric_only=True)
    df_imps = df_imps.sort_values(by='imps', ascending=False)
    ImpsTotal = df_imps['imps'].sum()
    top90DF = df_imps[df_imps['imps'] / ImpsTotal * 100 > 0.1]
    return df_imps, top90DF


## Create a dataframe with the sum of co2 per device

def render_df_sum_by_device(df):
    df_sum_by_device = df.groupby('device_type').sum(numeric_only=True)
    df_sum_by_device.reset_index(inplace=True)
    df_sum_by_device = df_sum_by_device.sort_values(by='co2Total', ascending=False)
    return df_sum_by_device



# Create a dataframe with the sum of co2 per Site Domain

def render_df_sum_by_site_domain(df, dfDB):
    dfSumBySiteDomain = df.groupby('site_domain').sum(numeric_only=True)
    dfSumBySiteDomain.reset_index(inplace=True)
    dfSumBySiteDomain = dfSumBySiteDomain.sort_values(by='co2Total', ascending=False)
    dfSumBySiteDomain = dfSumBySiteDomain.head(int(len(dfSumBySiteDomain) * 0.1))
    dfSumBySiteDomain['score'] = dfSumBySiteDomain['site_domain'].map(dfDB.set_index('domain')['score'])
    dfSumBySiteDomain['grade'] = dfSumBySiteDomain['score'].apply(get_grade_from_number)
    return dfSumBySiteDomain


# Create a dataframe with the lowest co2 per Click per Site Domain

def render_df_lowest_co2_per_click(top90DF):
    dfLowestCo2PerClick = top90DF.sort_values(by='co2Total', ascending=True)
    dfLowestCo2PerClick = dfLowestCo2PerClick[dfLowestCo2PerClick['clicks'] != 0]
    dfLowestCo2PerClick["co2PerClick"] = dfLowestCo2PerClick["co2Total"] / dfLowestCo2PerClick["clicks"]
    dfLowestCo2PerClick = dfLowestCo2PerClick.sort_values(by='co2PerClick', ascending=True)
    dfLowestCo2PerClick = dfLowestCo2PerClick.head(10)
    return dfLowestCo2PerClick


# Create a dataframe with the lowest co2 per impression (not null)

def render_df_lowest_co2_per_imp(top90DF):
    dfLowestCo2PerImp = top90DF.sort_values(by='co2Total', ascending=True)
    dfLowestCo2PerImp = dfLowestCo2PerImp[dfLowestCo2PerImp['imps'] != 0]
    dfLowestCo2PerImp["co2PerImp"] = dfLowestCo2PerImp["co2Total"] / dfLowestCo2PerImp["imps"]
    dfLowestCo2PerImp = dfLowestCo2PerImp.sort_values(by='co2PerImp', ascending=True)
    dfLowestCo2PerImp = dfLowestCo2PerImp.head(10)
    return dfLowestCo2PerImp


def get_df_and_metrics(ad_size):
    df, dfDB = load_dataframes()
    df = preprocess_data(df, ad_size)
    df_sum_by_day = render_df_sum_by_day(df)
    df_imps, top90DF = render_df_imps(df)
    df_sum_by_device = render_df_sum_by_device(df)
    dfSumBySiteDomain = render_df_sum_by_site_domain(df, dfDB)
    dfLowestCo2PerClick = render_df_lowest_co2_per_click(top90DF)
    dfLowestCo2PerImp = render_df_lowest_co2_per_imp(top90DF)
    device_types, video_completion_rate, imps, first_day, last_day = get_metrics(df)
    return df, df_sum_by_day, df_imps, df_sum_by_device, dfSumBySiteDomain, dfLowestCo2PerClick, dfLowestCo2PerImp, device_types, video_completion_rate, imps, first_day, last_day


def get_variation_today_vs_yesterday(df):
    # get the last day
    last_day = df['day'].max()
    # timestamp to YYYY-MM-DD
    last_day = last_day.strftime('%Y-%m-%d')
    df_last_day = df[df['day'] == last_day]

    # Find the day before the last day
    day_before_last = pd.to_datetime(last_day) - pd.DateOffset(days=1)
    day_before_last_str = day_before_last.strftime('%Y-%m-%d')

    # get one dataframe with only the last day
    # get one dataframe with only the day before the last day
    df_almost_last_day = df[df['day'] == day_before_last_str]
    return df_last_day, df_almost_last_day


def imp_variation_percentage(df):
    today, yesterday = get_variation_today_vs_yesterday(df)
    denominator = yesterday['imps'].sum()
    return round(((today['imps'].sum() - denominator) / denominator * 100), 2) if denominator != 0 else None


def click_variation_percentage(df):
    today, yesterday = get_variation_today_vs_yesterday(df)
    denominator = yesterday['clicks'].sum()
    return (today['clicks'].sum() - denominator) / denominator * 100 if denominator != 0 else None


def video_completion_variation_percentage(df):
    today, yesterday = get_variation_today_vs_yesterday(df)
    denominator = yesterday['video_completions'].sum()
    return round(((today['video_completions'].sum() - denominator) / denominator * 100),
                 2) if denominator != 0 else None


def budget_variation_percentage(df):
    today, yesterday = get_variation_today_vs_yesterday(df)
    denominator = yesterday['curator_total_cost'].sum()
    return round(((today['curator_total_cost'].sum() - denominator) / denominator * 100),
                 2) if denominator != 0 else None


def co2_variation_percentage(df):
    today, yesterday = get_variation_today_vs_yesterday(df)
    denominator = yesterday['co2Total'].sum()
    return round(((today['co2Total'].sum() - denominator) / denominator * 100), 2) if denominator != 0 else None


def co2_wasted_variation_percentage(df):
    today, yesterday = get_variation_today_vs_yesterday(df)
    co2_wasted_today = round(
        today["co2Total"].sum() * 1000 - (today["Viewability Rate %"] * today["co2Total"]).sum() * 1000, 2)
    co2_wasted_yesterday = round(
        yesterday["co2Total"].sum() * 1000 - (yesterday["Viewability Rate %"] * yesterday["co2Total"]).sum() * 1000, 2)
    return round((co2_wasted_today - co2_wasted_yesterday) / co2_wasted_yesterday * 100,
                 2) if co2_wasted_yesterday != 0 else None


def getDFDB():
    return pd.read_csv("assets/meanDomain.csv")


def get_grade_in_db(domain, dfDB):
    # get the series of the domain grade in the dfDB
    dfDB = dfDB[dfDB['domain'] == domain]
    return dfDB['score'].values[0]


def get_df(ad_size):
    df = pd.read_csv('Jellyfish.csv')
    df = df[df['day'] != '2023-03-02']
    df = df[df['day'] != '2023-03-01']
    df = preprocess_data(df, ad_size)
    first_day = df['day'].min()
    last_day = df['day'].max()
    return df, first_day, last_day

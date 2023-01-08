from turtle import title
import streamlit as st
from PIL import Image
import pandas as pd
import os
import numpy as np
import plotly.express as px
import webbrowser

import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

# reading crawled CVE & CAPEC data saved in this repo
cve_url = 'https://github.com/plau51ble/ThesisChapters/blob/1700ab641690c3ac309c8326d0c5218771ac92eb/Chapter%205/data/CHIA_cve.csv'
capec_url = 'https://github.com/plau51ble/ThesisChapters/blob/1700ab641690c3ac309c8326d0c5218771ac92eb/Chapter%205/data/CHIA_capec.csv'

names = ['vendor_name', 'capec_count', 'tags', 'countermeasures']
dash_cve_df = pd.read_csv(cve_url, index_col=0)
dash_capec_df = pd.read_csv(capec_url, index_col=0)

vendor_names = dash_capec_df['vendor_name'].unique().tolist()
vendor_names.sort()

#print(vendor_names)

def substringSieve(string_list):
    string_list.sort(key=lambda s: len(s), reverse=True)
    out = []
    for s in string_list:
        if not any([s in o for o in out]):
            out.append(s)
    return out

from PIL import Image
import requests

img_url = 'https://github.com/plau51ble/cody-educates/blob/1f37fb3e51f5c02629bcdcffa34757c6818e41d7/logo.png?raw=true'
image = Image.open(requests.get(img_url, stream=True).raw)
st.image(image)

sidebar = st.sidebar
sidebar.markdown("Hello and welcome dear visitor,")
sidebar.markdown("Smart appliances like TV, lighting solutions, smart plugs, refrigerators and so on have been hacked in the past.")
sidebar.markdown("This page presents the cybersecurity history of brands to help you with buying decisions on their products.")
sidebar.markdown("Please use the menu below for brands that we have curated for you.")
location_selector = sidebar.selectbox(
    "Select a Brand",
    vendor_names
)
vendor_nm = location_selector
st.markdown(f"# How safe are {location_selector} products?")

line_inp = dash_cve_df[dash_cve_df.vendor_name == vendor_nm]
line_inp = line_inp.drop_duplicates(subset=['CVE', 'Published'])

plot_df = line_inp[['CVE', 'Published']]

plot_df['Count'] = plot_df.groupby('Published')['Published'].transform('count')
plot_df = plot_df.rename(columns={'Published': 'Year'})
plot_df.drop_duplicates(subset=['Year'], inplace = True)
count_max = plot_df['Count'].max()
count_min = plot_df['Count'].min()
acc_bar = px.bar(plot_df,
                    x = 'Year',
                    y = 'Count',
                    title = 'Cybersecurity incidents over the years',
                    color_discrete_sequence = px.colors.sequential.RdBu)
acc_bar.update_xaxes(type='category', categoryorder='category ascending')
acc_bar.update_yaxes(range=[count_min, count_max])
st.write(acc_bar)

st.markdown("---")

st.markdown("## Passwords are sometimes not enough to prevent a cyber attack.")
st.markdown("- A hacker may not require password to access your device/network")
st.markdown("- A hacker may need one or more attempts to access your device/network")
st.markdown("- Your passwords could have low, medium or high complexities")

st.markdown(f"### Among the hacked {location_selector} products:")

access_pie_col, password_req_col = st.columns(2)

pass_df = dash_cve_df[dash_cve_df.vendor_name == vendor_nm]
pass_df = pass_df[['CVE', 'access_complexity']]
pass_df.dropna(subset=['access_complexity'], inplace=True)
pass_cnt = pass_df['access_complexity'].value_counts()
pass_pie = px.pie(pass_cnt, values=pass_cnt.values, names=pass_cnt.index,
                     color_discrete_sequence = px.colors.sequential.RdBu)
pass_pie.update_layout(font_size = 8, margin=dict(l=0, r=450, t=20, b=20))

acc_df = dash_cve_df[dash_cve_df.vendor_name == vendor_nm]
acc_df = acc_df[['CVE', 'access_auth']]
acc_df.dropna(subset=['access_auth'], inplace=True)
acc_cnt = acc_df['access_auth'].value_counts()
acc_pie = px.pie(acc_cnt, values=acc_cnt.values, names=acc_cnt.index,
                     color_discrete_sequence = px.colors.sequential.RdBu)
acc_pie.update_layout(font_size = 8, margin=dict(l=0, r=450, t=20, b=20))

access_pie_col.subheader("The number of passwords needed by hacker to attack")
access_pie_col.write(acc_pie)
password_req_col.subheader("The complexity of hacked passwords set by the users")
password_req_col.write(pass_pie)

st.markdown("---")

st.markdown("## How do your devices get affected by a cyber attack?")
st.markdown("- The device may become unavailable for the intended use")
st.markdown("- You cannot trust the quality of the data stored on your device")
st.markdown("- You cannot trust the confidentiality of the data stored on your device")
st.markdown(f"#### The extent of availability loss among attacked {location_selector} products")

ava_df = dash_cve_df[dash_cve_df.vendor_name == vendor_nm]
ava_df = ava_df[['CVE', 'impact_confidentiality']]
ava_df.dropna(subset=['impact_confidentiality'], inplace=True)

ava_cnt = ava_df['impact_confidentiality'].value_counts()
ava_pie = px.pie(ava_cnt, values=ava_cnt.values, names=ava_cnt.index,
                     color_discrete_sequence = px.colors.sequential.RdBu)
st.write(ava_pie)

st.markdown("---")

st.markdown(f"## The severity of attacks on {location_selector} products overall:")

cvss_df = dash_cve_df[dash_cve_df.vendor_name == vendor_nm]
cvss_df = cvss_df[['CVE', 'CVSS']]
cvss_df.dropna(subset=['CVSS'], inplace=True)
cvss_df = cvss_df[cvss_df.CVSS > 0]
conditions = [
    (cvss_df['CVSS'] <= 3.9),
    (cvss_df['CVSS'] > 4) & (cvss_df['CVSS'] <= 6.9),
    (cvss_df['CVSS'] > 7) & (cvss_df['CVSS'] <= 8.9),
    (cvss_df['CVSS'] > 9)
    ]

# create a list of the values we want to assign for each condition
values = ['Low', 'Medium', 'High', 'Critical']

# create a new column and use np.select to assign values to it using our lists as arguments
cvss_df['Severity'] = np.select(conditions, values)
cvss_df = cvss_df[cvss_df['Severity'].isin(values)]

cvss_cnt = cvss_df['Severity'].value_counts()
cvss_pie = px.pie(cvss_cnt, values=cvss_cnt.values, names=cvss_cnt.index,
                     color_discrete_sequence = px.colors.sequential.RdBu)
st.write(cvss_pie)

st.markdown("---")
st.markdown("## What can you do to protect your devices and the network?")
st.markdown("- As a user you can prevent possible cyber attacks too")
st.markdown("- Simple measures can ensure that your network and connected devices are safe")

sols = dash_capec_df.loc[dash_capec_df['vendor_name'] == vendor_nm, 'countermeasures'].values[0]
if not sols:
    print ("")
else:
    sols = sols.strip("[\[\]")
    sols = sols.replace('\'', '')

    sols_list = sols.split(",")
    sols_list = [tg.lstrip() for tg in sols_list]
    sols_list = list(set(sols_list))
    sols_list = substringSieve(sols_list)

    st.markdown("##### You can either implement the following safeguards yourself, or seek help from an expert")
    for cmeasure in sols_list:
        st.write(f"* {cmeasure}")

with st.expander("Show me more"):
    st.write("""
    These are some searchable tags that we gathered to further your cybersecurity knowledge.""")

    tags = dash_capec_df.loc[dash_capec_df['vendor_name'] == vendor_nm, 'tags'].values[0]

    tags = tags.strip("[\[\]")
    tags = tags.replace('\'', '')
    tags_list = tags.split(",")
    tags_list = [tg.lstrip() for tg in tags_list]
    tags_list = list(set(tags_list))
    tags_list = substringSieve(tags_list)

    tag_query_dict = {}
    for tgs in tags_list:
        tag_query_dict[tgs] = "https://www.google.com/search?q=" + '+'.join(tgs.split(' '))

    for key in tag_query_dict:
        st.write(f"[{key}]({tag_query_dict[key]})")


st.markdown("---")
st.write("_We would now like to know your opinion about this page._")
if st.button('Sure! Take me to survey'):
    url = 'https://forms.gle/Cg1oVkCimphYMLAbA'
    webbrowser.open_new_tab(url)
    st.write('_Thank you!_')
else:
     st.write('_Goodbye!_')

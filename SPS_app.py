#|-------------------------------------|
#|----IMPORTS--------------------------|
#|-------------------------------------|

import pandas as pd
import numpy as np
import streamlit as st
import os, time, datetime as dt, random

import spotipy
from spotipy.oauth2 import SpotifyOAuth


#|-------------------------------------|
#|----Functions------------------------|
#|-------------------------------------|

# Sign in to Spotify
def app_sign_in():
    try: sp = spotipy.Spotify(auth=st.session_state["cached_token"])
    except Exception as e:
        st.write('Error during sign-in:')
        st.write(e)
    else:
        st.session_state["signed_in"] = True
        app_display_welcome()
        st.success("Sign in success!")
        
    return sp

# //Route one - not signed in yet//
def display_welcome_not_signed_in():
    # :-Dependencies----:
    welcome_msg = """
    This app aims to help create intuitive playlists from an existing song base! Current functions: 
    Sequencing a playlist based on target meta-features from a larger pool.
    
    Planned improvements: 
    Sequencing playlist based on user's liked songs.
    """
    
    link_html = " <a target=\"_self\" href=\"{url}\" >{msg}</a> ".format(
        url=auth_url,
        msg="AUTHENTICATE"
    )
    
    # :-Display---------:
    st.title("Spotify Playlist Sequencer")
    st.markdown(welcome_msg)
    st.write("No credentials found for this session. Please log in by clicking the link below.")
    st.markdown(link_html, unsafe_allow_html=True)

    
#|-------------------------------------|
#|----CODE-----------------------------|
#|-------------------------------------|

# Determining session cache state
if "signed_in" not in st.session_state:    st.session_state["signed_in"] = False
if "cached_token" not in st.session_state: st.session_state["cached_token"] = ""
if "code" not in st.session_state:         st.session_state["code"] = ""
if "oauth" not in st.session_state:        st.session_state["oauth"] = None

url_params = st.experimental_get_query_params()


    
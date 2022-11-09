# |==========================|
# |-Packages-----------------|
# |==========================|

import os, time, random, datetime as dt
import pandas as pd
import numpy as np
import streamlit as st

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# |==========================|
# |-Functions----------------|
# |==========================|

def get_token(oauth, code):

    token = oauth.get_access_token(code, as_dict=False, check_cache=False)
    # remove cached token saved in directory
    os.remove(".cache")
    
    return token

def sign_in(token):
    sp = spotipy.Spotify(auth=token)
    return sp

def app_get_token():
    try:
        token = get_token(st.session_state["oauth"], st.session_state["code"])
    
    # Error handling when getting the token
    except Exception as e:
        st.error("An error occurred during token retrieval!")
        st.write("The error is as follows:")
        st.write(e)
    else:
        st.session_state["cached_token"] = token

# Sign in to Spotify
def app_sign_in():
    try:
        sp = sign_in(st.session_state["cached_token"])
    except Exception as e:
        st.error("An error occurred during sign-in!")
        st.write("The error is as follows:")
        st.write(e)
    else:
        st.session_state["signed_in"] = True
        app_display_welcome()
        st.success("Sign in success!")
        
    return sp

def app_display_welcome():
    
    # import secrets from streamlit deployment
    cid = st.secrets["SPOTIPY_CLIENT_ID"]
    csecret = st.secrets["SPOTIPY_CLIENT_SECRET"]
    uri = st.secrets["SPOTIPY_REDIRECT_URI"]

    # set scope and establish connection
    scopes = " ".join(["user-read-private",
                       "playlist-read-private",
                       "playlist-modify-private",
                       "playlist-modify-public",
                       "user-read-recently-played"])

    # create oauth object
    oauth = SpotifyOAuth(scope=scopes,
                         redirect_uri=uri,
                         client_id=cid,
                         client_secret=csecret)
    # store oauth in session
    st.session_state["oauth"] = oauth

    # retrieve auth url
    auth_url = oauth.get_authorize_url()
    
    # this SHOULD open the link in the same tab when Streamlit Cloud is updated
    # via the "_self" target
    link_html = " <a target=\"_self\" href=\"{url}\" >{msg}</a> ".format(
        url=auth_url,
        msg="Click me to authenticate!"
    )
    
    # define welcome
    welcome_msg = """
    Welcome! :wave: This app uses the Spotify API to interact with general 
    music info and your playlists! In order to view and modify information 
    associated with your account, you must log in. You only need to do this 
    once.
    """
    
    # define temporary note
    # note_temp = """
    # _Note: Unfortunately, the current version of Streamlit will not allow for
    # staying on the same page, so the authorization and redirection will open in a 
    # new tab. This has already been addressed in a development release, so it should
    # be implemented in Streamlit Cloud soon!_
    # """

    st.title("Spotify Playlist Preserver")

    if not st.session_state["signed_in"]:
        st.markdown(welcome_msg)
        st.write(" ".join(["No tokens found for this session. Please log in by",
                          "clicking the link below."]))
        st.markdown(link_html, unsafe_allow_html=True)
        # st.markdown(note_temp)

# |==========================|
# |-Code---------------------|
# |==========================|

# Determining session cache state
if "signed_in" not in st.session_state:    st.session_state["signed_in"] = False
if "cached_token" not in st.session_state: st.session_state["cached_token"] = ""
if "code" not in st.session_state:         st.session_state["code"] = ""
if "oauth" not in st.session_state:        st.session_state["oauth"] = None
    
url_params = st.experimental_get_query_params()

# attempt sign in with cached token
if st.session_state["cached_token"] != "":
    sp = app_sign_in()
# if no token, but code in url, get code, parse token, and sign in
elif "code" in url_params:
    # all params stored as lists, see doc for explanation
    st.session_state["code"] = url_params["code"][0]
    app_get_token()
    sp = app_sign_in()
# otherwise, prompt for redirect
else:
    app_display_welcome()
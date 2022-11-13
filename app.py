import os, time, random, datetime as dt
import pandas as pd
import numpy as np
import streamlit as st

import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
#|--------------------|
# Initialisation State
#|--------------------|

# Initial States
if "signed_in" not in st.session_state:    st.session_state["signed_in"] = False
if "cached_token" not in st.session_state: st.session_state["cached_token"] = ""
if "code" not in st.session_state:         st.session_state["code"] = ""
if "oauth" not in st.session_state:        st.session_state["oauth"] = None

# import secrets from streamlit deployment
cid = st.secrets["cid"]
skey = st.secrets["skey"]
uri = st.secrets["red_uri"]

# set scope and establish connection
scopes = " ".join(["user-read-private",
                   "playlist-read-private",
                   "playlist-modify-private",
                   "playlist-modify-public",
                   # "user-read-recently-played"
                  ])

# 0Auth object definition
oauth = SpotifyOAuth(scope=scopes,
                     redirect_uri=uri,
                     client_id=cid,
                     client_secret=skey,
                     cache_path='.spotipyoauthcache')

# Save state of 0Auth
st.session_state['oauth'] = oauth

# Get cached login if not signed in
if st.session_state.signed_in == False:
    
    with st.spinner('Please wait...attempting login'):
        
        # Create printable that we can remove later
        login_text = st.empty()

        # Get cached token if any
        access_token = ""
        token_info = oauth.get_cached_token()

        # Check for cached token
        if token_info:
            st.write("Found cached token!")
            access_token = token_info['access_token']

        # If no cached token, get url response code
        else:
            st.write('No cached token, get url')
            url = oauth.get_authorize_url()
            code = oauth.parse_response_code(url)

            # Once you have the code, attempt to get access token
            if code:
                st.write('Found Spotify auth code in Request URL! Trying to get valid access token...')
                access_token = oauth.get_access_token(as_dict=False)
                # access_token = token_info['access_token']

        # Access token exists, then use access token to access Spotify
        if access_token:
            st.write("Access token available! Trying to get user information...")
            sp = spotipy.Spotify(access_token,auth_manager=oauth)
            user_details = sp.current_user()

        else:
            # Display link to login
            auth_url = oauth.get_authorize_url()
            link_html = " <a target=\"_self\" href=\"{a_url}\" >{msg}</a> ".format(
                a_url = auth_url,
                msg = "AUTHENTICATE"
            )
            st.markdown('Login to Spotify here:')
            st.markdown(link_html, unsafe_allow_html=True)
            st.stop()

        if user_details:
            st.session_state.username = user_details['display_name']
            st.session_state.userid = user_details['id']
            st.session_state.user_uri = user_details['uri']
            st.session_state.signed_in = True
            login_text.empty()
            
            

# Sidebar
def sidebar_params():
    
    with st.sidebar:
        with st.form('playlist_feature_targets'):
            # Meta-Features
            st.markdown('Meta-Features')
            impact = st.slider('Impact',min_value=0.0,max_value=1.0,value=0.5)
            hype = st.slider('Hype',min_value=0.0,max_value=1.0,value=0.5)
            vibe = st.slider('Vibe',min_value=0.0,max_value=1.0,value=0.5)

            # Feature targets
            st.markdown('Fine-Tuning:')
            instrumentalness = st.slider('Instrumentalness',min_value=0.0,max_value=1.0,value=0.5)
            # liveness = st.slider('Liveness',min_value=0.0,max_value=1.0,value=0.5)
            speechiness = st.slider('Speechiness',min_value=0.0,max_value=1.0,value=0.5)
            acousticness = st.slider('Acousticness',min_value=0.0,max_value=1.0,value=0.5)
            danceability = st.slider('Danceability',min_value=0.0,max_value=1.0,value=0.5)

            # # Limits
            # energy = st.slider('Energy',min_value=0.0,max_value=1.0,value=(0.20,0.75))
            # # Need to set to max/min of tempo of dataset
            # tempo = st.slider('Tempo',min_value=0.0,max_value=1.0,value=(0.20,0.75))
            # loudness = st.slider('Loudness',min_value=0.0,max_value=1.0,value=(0.20,0.75))

            #form button
            pl_feat_gen = st.form_submit_button('Generate!')

            # Some guiding sentences on the metrics
            feat_def_exp = st.expander('Some definitions to help you:')
            feat_def_exp.write('''  
                **Impact**:  
                How much 'oomph' you want in your music\n
                **Hype**:  
                When you want your heart rate to increase\n
                **Vibe**:  
                Common phrase said when vibing: 'this s*** slaps'\n
                **Instrumentalness**:  
                "Wait shouldn't a song have a singer?"\n
                **Speechiness**:  
                "Now you're just saying the lyrics" -rap hater\n
                **Acousticness**:  
                On a scale of Ed Sheeran to Skrillex\n
                **Danceability**:  
                Do your hips move involuntarily?  
                ''')

            if pl_feat_gen:
                # What happens after generate is pressed?
                pass
                
            
                
# if st.session_state.
#|---------------|
# Start up screen
#|---------------|

# Title
st.title('A More Intuitive Playlist Generator for Spotify')

# Subtext
st.markdown(f'''
Welcome {st.session_state.username}! This app aims to help create intuitive playlists 
from an existing song base. Right now, the app works by sequencing a playlist of set 
lengthbased on target meta-features from a pool, using a larger playlist as its base.
''')

# List of upcoming improvements
improvements = st.expander('Planned improvements:')            
improvements.markdown('''
    1. Sequencing playlist based on user's liked songs!
    2. Text-based hype/vibe assignment!
    3. Sequencing based on a song or artist seed!
    4. and more!
    ''')

# Playlist URL Input, and form submission
with st.form('playlist_input'):

    # Assign playlist_url null value to session state
    if 'playlist_url' not in st.session_state: st.session_state.playlist_url = ''

    url_tmp = st.text_input('To start, please input a playlist url:',
                            value=st.session_state.playlist_url)

    col1, col2, dummy = st.columns([1,1,6], gap='small')
    with col1: pl_submitted = st.form_submit_button('Submit')
    with col2: pl_clear = st.form_submit_button('Clear')

    if pl_submitted:

        #|------------------------------------|
        # KIV: Add playlist verification here!
        #|------------------------------------|

        st.session_state.playlist_url = url_tmp
        st.session_state.sidebar_state = 'expanded' # Set sidebar to open
        # Force an app rerun after switching the sidebar state.
        st.experimental_rerun() # Refresh app with sidebar open


    elif pl_clear:
        st.session_state.playlist_url = ''
        st.experimental_rerun()
        
# Initialise the sidebar
if ('playlist_url' in st.session_state) and (st.session_state.playlist_url != ''):
    sidebar_params()
    

import os, time, random, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import norm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials



#|--------------------|
# Initialisation State
#|--------------------|

# Initial States
if "signed_in" not in st.session_state:    st.session_state["signed_in"] = False
if "cached_token" not in st.session_state: st.session_state["cached_token"] = ""
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
            login_text.write("Found cached token!")
            access_token = token_info['access_token']

        # If no cached token, get url response code
        else:
#             st.write('No cached token, get url')
#             url = oauth.get_authorize_url()
#             code = oauth.parse_response_code(url)

#             # Once you have the code, attempt to get access token
#             if code:
            login_text.write('Found Spotify auth code in Request URL! Trying to get valid access token...')
            access_token = oauth.get_access_token(as_dict=False)
            login_text.write('Access Token retrieved')
            # access_token = token_info['access_token']

        # Access token exists, then use access token to access Spotify
        if access_token:
            login_text.write("Access token available! Trying to get user information...")
            sp = spotipy.Spotify(access_token,auth_manager=oauth)
            st.session_state.sp = sp
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
def sidebar_params(df):
    
    with st.sidebar:
        
        st.session_state.adv_switch = False
        st.session_state.adv_switch = st.checkbox('Advanced Layout',value=False)

        with st.form('playlist_feature_targets'):
            # Meta-Features
            st.markdown('Meta-Features')
            impact = st.slider('Oomph',min_value=0.0,max_value=1.0,value=0.5, # More understandable as 'oomph'
                              help="How much 'oomph' you want in your music")
            hype = st.slider('Hype',min_value=0.0,max_value=1.0,value=0.5,
                            help="When you want your heart rate to increase")
            vibes = st.slider('Vibes',min_value=0.0,max_value=1.0,value=0.5,
                             help="Common phrase said when vibing: 'this s*** slaps'")

            # Feature targets
            if st.session_state.adv_switch:
                st.markdown('Fine-Tuning:')
                instrumentalness = st.slider('Instrumentalness',min_value=0.0,max_value=1.0,value=0.5,
                                            help="'Wait shouldn't a song have a singer?'")
                liveness = st.slider('Liveness',min_value=0.0,max_value=1.0,value=0.5,
                                    help="More crowds cheering")
                speechiness = st.slider('Speechiness',min_value=0.0,max_value=1.0,value=0.5,
                                       help="'Now you're just saying the lyrics' -rap hater")
                acousticness = st.slider('Acousticness',min_value=0.0,max_value=1.0,value=0.5,
                                        help="On a scale of Ed Sheeran to Skrillex")
                danceability = st.slider('Danceability',min_value=0.0,max_value=1.0,value=0.5,
                                        help="Your hips don't lie")
                energy = st.slider('Energy',min_value=0.0,max_value=1.0,value=0.5,
                                  help="This 'feels' like a lot")
                tempo = st.slider('Tempo',min_value=0.0,max_value=1.0,value=0.5,
                                 help="Music, but faster")
                loudness = st.slider('Loudness',min_value=0.0,max_value=1.0,value=0.5,
                                help="How much sound do you want")
            
            # Other limits
            st.markdown('Other Considerations:')
            popularity = st.slider('Popularity',min_value=0.0,max_value=1.0,value=0.5,
                                  help="The higher the number, the more your friends might know the songs")


            
            # Number of tracks
            num_tracks = st.number_input(
                f'Number of Tracks (max tracks:{int(st.session_state.playlist_track_total)})',
                min_value = 1, 
                max_value = int(st.session_state.playlist_track_total),
                value = 50, # Default to 50 songs
                step=1)
            
            explicit = st.checkbox('explicit',value=True)

            #form button
            pl_feat_gen = st.form_submit_button('Generate!')

            if pl_feat_gen:
                # What happens after generate is pressed?                
                target_vector = {'track_uri':'target',
                                 'impact':max(0.01,min(impact,0.99)),
                                 'hype':max(0.01,min(hype,0.99)),
                                 'vibes':max(0.01,min(vibes,0.99)),
                                 'Popularity':max(0.01,min(popularity,0.99))
                                }
                if st.session_state.adv_switch:
                    target_vector_adv = {'instrumentalness':max(0.01,min(instrumentalness,0.99)),
                                         'liveness':max(0.01,min(liveness,0.99)),
                                         'speechiness':max(0.01,min(speechiness,0.99)),
                                         'acousticness':max(0.01,min(acousticness,0.99)),
                                         'danceability':max(0.01,min(danceability,0.99)),
                                         'energy':max(0.01,min(energy,0.99)),
                                         'tempo':max(0.01,min(tempo,0.99)),
                                         'loudness':max(0.01,min(loudness,0.99))
                                        }
                    
                    target_vector.update(target_vector_adv)
                
                
                # Drop any vectors that are around 0.5; in other words, neither -ve nor +ve vector
                list_drop = []
                for vec in [i for i in target_vector.keys() if i not in ['track_uri','vibes']]:
                    if round(target_vector[vec],1) == 0.5: list_drop = list_drop + [vec]
                for vec in list_drop: target_vector.pop(vec)
                
                
                
                # Save target_vector and explicit switch state
                st.session_state.target_vector = target_vector
                st.session_state.use_explicit = explicit
                
                
                
                # After every 'generate', re-treat the dataframe to get a 
                # new similarity matrix and display df
                df = st.session_state.df
                display_features = ['Name','Artist','Album','Popularity','Explicit','Genres']
                df_scaled = df_scaled_transform(df) # Transform and scale the dataframe slice
                
                # Perform cosine similarity and generate track list
                track_list = get_sim_list(df_scaled,target_vector) 
                
                # Generate df with only track list
                df_scaled = df.set_index('track_uri').reindex(
                    index = track_list).reset_index().head(num_tracks)
                
                # Update display df
                df_display = df_scaled[display_features].copy()
                # Reconvert back to lists
                for i in df_display.index:
                    for col in ['Artist','Genres']:
                        df_display.loc[i,[col]] = {col:listify(df_display.loc[i,col])}
                    
                    
                    
                st.session_state.track_list = track_list[:num_tracks]
                st.session_state.df_display = df_display
                
                st.experimental_rerun()
            
                
                


# Get track totals from URL
def get_track_total(url_tmp):
    try:
        playlist_tmp = url_tmp.split('/')[-1].split('?')[0] # Playlist URI
        track_total = sp.playlist(playlist_tmp)['tracks']['total']
        return track_total
    except:
        st.markdown('ERROR: Invalid playlist URL')
        st.stop()
        
        


# Format Genre Output
def format_genre(genre_dict,artists_uri):
    genre_set = []
    for j in range(len(artists_uri)):
        genre_tmp = [genre_dict[artists_uri[j][i]] for i in range(len(artists_uri[j]))]
        genre_tmp = np.unique([x for y in genre_tmp for x in y])
        genre_set.append(genre_tmp)
    
    return genre_set



#|----------------------------|
# Get track details in batches
#|----------------------------|
# Genre, Explicity, and Availability commented out until can fix spotify API
def get_track_details(source,sample,market=None):
    
    # |-Get track, artist, and album names-|
    track_name = [source['items'][track]['track']['name'] for track in range(sample)]
    album_name = [source['items'][track]['track']['album']['name'] for track in range(sample)]
    
    # |-Artists information----------------|
    artists_list = [source['items'][track]['track']['artists'] for track in range(sample)]
    artists_name = [[artists_list[track][artist]['name'] 
                     for artist in range(len(artists_list[track]))] for track in range(len(artists_list))]
    
    # |-Get corresponding URIs-------------|
    track_uri = [source['items'][track]['track']['uri'] for track in range(sample)]
    album_uri = [source['items'][track]['track']['album']['uri'] for track in range(sample)]
    
    # |-Artists URIs-----------------------|
    artists_uri = [[artists_list[track][artist]['uri'] 
                    for artist in range(len(artists_list[track]))] for track in range(len(artists_list))]

    # |-Other Metrics----------------------|
    track_pop = [source['items'][track]['track']['popularity'] for track in range(sample)] # Popularity Index
    track_expl = [source['items'][track]['track']['explicit'] for track in range(sample)] # Explicity Index
    if market != None: track_play = [source['items'][track]['track']['is_playable'] 
                                     for track in range(sample)] # Playability Index
    else: track_play = [True for track in range(sample)]

    # |-Get track audio features-----------|
    audio_feat_raw = sp.audio_features(track_uri)
    audio_features = [audio_feat_raw[i] for i in range(sample)]
    
    # |-Associated genres of artists-------|
    # 1. Get full list of artists in 1 continuous list
    artists_uri_condensed = [artists_list[i][j]['uri'] for i in range(len(artists_list)) 
                             for j in range(len(artists_list[i]))]
    
    # 2. Get details of associated artists based on the subset of tracks from scrape
    genre_raw = sp.artists(np.unique(artists_uri_condensed))
    
    # 3. Create dictionary of artist to associated genres
    genre_dict = {genre_raw['artists'][i]['uri']:genre_raw['artists'][i]['genres'] 
                  for i in range(len(genre_raw['artists']))}
    
    # 4. Combine each list of unique genres from each of associated artists
    artists_genres = format_genre(genre_dict,artists_uri)
    
    
    # |-Intermediary DataFrames------------|
    track_details = pd.DataFrame(list(zip(track_name,artists_name,album_name,
                                          artists_genres,
                                          track_pop,
                                          track_expl,track_play
                                         )), 
                                 columns=['Name','Artist','Album',
                                          'Genres',
                                          'Popularity',
                                          'Explicit','Available'
                                         ])
    audio_details = pd.DataFrame(audio_features)
    track_uris = pd.DataFrame(list(zip(track_uri,artists_uri,album_uri)), columns=['track_uri','artist_uri','album_uri'])

    # |-Combine dataframe outputs----------|
    return pd.concat((track_details,
                      audio_details[[col for col in audio_details.columns if col not in ['type','id','uri','track_href','analysis_url']]],
                      audio_details[['type','track_href','analysis_url']],
                      track_uris,
                     ),axis=1)
    
    


#|-----------------------------------------------------------------------------------|
# Compile track details, by batch size, for n number of tracks from a playlist source
#|-----------------------------------------------------------------------------------|
# Compile all track details for the given inputs
def compile_track_details(playlist_link,tracks=-1,sample=100,market=None):
    
    # Track scraping time
    start_time = time.time()
    elapsed = st.empty() # Updated text for elapsed time
    
    # Defining relevant variables
    playlist_URI = playlist_link.split('/')[-1].split('?')[0] # Playlist URI
    current = 0 # Track scrape counter
    
    # If tracks is -1, pull all songs in a playlist
    if tracks == -1: tracks = sp.playlist(playlist_URI,market=market)['tracks']['total']
    
    # Progress bar
    bar_songs = st.progress(0)
    
    # Empty dataframe to store all the track information
    df = pd.DataFrame()
    
    while current < tracks:
        
        # Limit the scrape to only the necessary number of pulls for the last query
        if tracks - current < sample: sample = tracks - current
        
        source = sp.playlist_tracks(playlist_URI,limit=sample,offset=current,market=market)
        df_tmp = get_track_details(source,sample,market=market)
        
        df = pd.concat((df,df_tmp)).reset_index(drop=True)
        current = len(df.index) # Set current track counter to the size of the dataframe
        
        # Milestone tracking
        # Elapsed time
        time_curr = time.time()
        hours = int((time_curr - start_time)/(60 ** 2))
        minutes = int(round(((time_curr - start_time)/60) % 60,0))
        seconds = int(round((time_curr - start_time) % 60,0))

        # Print Elapsed time and data volume
        # Update progress bar
        elapsed.write(f'''Elapsed time(h.m.s):{str(hours)}.{str(minutes)}.{str(seconds)}, Current Total Tracks: {len(df.index)}''')
        bar_songs.progress(len(df.index)/tracks)

    return df



#|-------------------------------------------------|
# Function to regenerate list from the .csv dataset
#|-------------------------------------------------|
def listify(x):
    x_tmp = str(x).replace('[','').replace(']','').replace("'","").split(',')
    x_tmp = [i.strip() for i in x_tmp]
    return x_tmp



#|--------------------------------|
# df scrape/pull .csv default file
#|--------------------------------|
# Defining as function so we can call it whenever the url is updated
def df_scrape(url):
# If not using default dataset
    if st.session_state.use_default == False:
        df = compile_track_details(url,
                                   tracks=int(st.session_state.tracks), 
                                   sample=25, # Sample set to 25 until function can handle artists
                                   market=None)

    # Using default - pulling data from pre-prepared .csv    
    elif st.session_state.use_default == True: 
        path = Path(__file__).parent/'Datasets/playlist_example.csv'
        df = pd.read_csv(path)        
        df = df.head(int(st.session_state.tracks))

    # Reconvert back to lists
    for i in df.index:
        for col in ['Artist','Genres','artist_uri']:
            df.loc[i,[col]] = {col:listify(df.loc[i,col])}
            
    return df



#|-----------------------|
# Data Cleaning Functions
#|-----------------------|

# 1. Remove true duplicates - not to be confused with covers of songs
def df_remove_true_dup(df,st_empty):
    st_empty.write('Removing true duplicates...')
    
    unique_tags = ['Popularity','Album','track_href','analysis_url','track_uri','artist_uri','album_uri']
    
    # df of song duplicates
    songs_dup = df[[col for col in df.columns if col not in unique_tags]
                  ].copy()[df.duplicated(subset='Name',keep=False)]
    
    songs_dup[['Artist','Genres']] = songs_dup[['Artist','Genres']].astype(str) # Convert lists into strings
    
    # Remove true duplicates
    df_no_dup = df.copy().drop(index=songs_dup[songs_dup.duplicated()].index, errors='ignore')
    df_no_dup = df_no_dup.reset_index(drop=True) 
    
    return df_no_dup



#|-------------------------|
# Data Processing Functions
#|-------------------------|

# 1. Drop columns
def df_drop_columns(df,st_empty):
    st_empty.write('Dropping unnecessary columns...')
    
    drop_col = ['Name','Album','key','mode','time_signature',
                'duration_ms','track_href','analysis_url','artist_uri',
                'album_uri','type','Available','Explicit']
    
    df_dropped = df[['track_uri']+[col for col in df.columns if col not in drop_col+['track_uri']]]
    
    return df_dropped



# 2. Vectorise Genres
def df_vectorize_genres(df, st_empty, ngram=(1,3)):
    st_empty.write('Vectorizing Genres...')
    
    df_str = df.copy()
    
    for col in ['Genres']:
        df_str[col] = [[i.strip() for i in j] for j in df[col]]
        df_str[col] = [','.join(lst) for lst in df_str[col]]
    
    # Count Vectorizer for Genres
    # Binary = True: We only care if it is something, not how many times it appears
    cvec = CountVectorizer(ngram_range=ngram, binary=True)
    cvec_array_gen = cvec.fit_transform(df_str['Genres'])

    # Temp df - coded this way in case we need to consider vectorising Artists in the future
    temp = pd.concat([
        # pd.DataFrame(cvec_array_gen.toarray(),columns=cvec.get_feature_names_out())
        #|-----------------------------------------------------------------------------------|
        # Depending on the version of CountVectorizer, may need to run this code instead.
        pd.DataFrame(cvec_array_gen.toarray(),columns=cvec.get_feature_names())
        #|-----------------------------------------------------------------------------------|  
    ],axis=1).fillna(0).astype(int)
    
    # Opted for Count-Vec + dropping columns <= 1 sum
    # Earlier tests suggest that the Tfidf vectorizer regularises too many genre tokens for the vectors to be useful
    temp = temp.drop(columns=temp.loc[:,temp.sum() <= 1].columns)
    
    # Exploiting .duplicated() to remove repeated columns
    temp_2 = temp.copy()
    temp_2.columns = [str(list(np.unique(i.split(' ')))).replace('[','').replace(']','').replace("'",'').replace(',','') 
                      for i in temp.columns]
    temp_2 = temp_2.loc[:,~temp_2.columns.duplicated()]
    
    # Merging back
    df_vectorized=pd.concat((
        df_str[[col for col in df_str.columns if col not in ['Artist','Genres']]],
        temp_2),
        axis=1)
    
    return df_vectorized



# 3. Scaling Numerical Features
def df_scale_numfeat(df, st_empty):
    st_empty.write('Scaling numerical features...')
    
    audio_features = ['danceability','energy','key','loudness','mode','speechiness','acousticness',
                      'instrumentalness','liveness','valence','tempo','duration_ms','time_signature']
    
    audio_feat_vect = ['Popularity']+[col for col in audio_features if col in df.columns]
    
    ss = StandardScaler()
    df_varray = ss.fit_transform(df[audio_feat_vect])
    df_scaled = df.copy()
    df_scaled[audio_feat_vect] = pd.DataFrame(df_varray,columns=audio_feat_vect)
    
    return df_scaled



# 4. Creating mood vectors
def df_mood_vectors(df, st_empty):
    st_empty.write('Scaling numerical features...')
    
    mood_markers=['danceability','energy','loudness','acousticness','valence','tempo']
    df_tmp = df.copy()
    
    for i in df_tmp.index:
        # Excitement - arbitrary weights for constituent vectors, based on domain knowledge
        df_tmp.loc[i,'impact'] = np.average(a=(df_tmp.loc[i,'danceability'],
                                               df_tmp.loc[i,'energy'],
                                               df_tmp.loc[i,'valence'],
                                               df_tmp.loc[i,'tempo'],
                                               df_tmp.loc[i,'acousticness'],
                                               df_tmp.loc[i,'loudness']),
                                            weights=[0,1,0,0.5,-0.5,1])

        # Hype - arbitrary weights for constituent vectors, based on domain knowledge
        df_tmp.loc[i,'hype'] = np.average(a=(df_tmp.loc[i,'danceability'],
                                             df_tmp.loc[i,'energy'],
                                             df_tmp.loc[i,'valence'],
                                             df_tmp.loc[i,'tempo'],
                                             df_tmp.loc[i,'acousticness'],
                                             df_tmp.loc[i,'loudness']),
                                          weights=[0.5,1,1,1,0,0.5])

        # Vibes - arbitrary weights for constituent vectors, based on domain knowledge
        df_tmp.loc[i,'vibes'] = np.average(a=(df_tmp.loc[i,'danceability'],
                                              df_tmp.loc[i,'energy'],
                                                # vibe is more to do with intensity, so +/- valence should be considered
                                              np.absolute(df_tmp.loc[i,'valence']), 
                                              df_tmp.loc[i,'tempo'],
                                              df_tmp.loc[i,'acousticness'],
                                              df_tmp.loc[i,'loudness']),
                                           weights=[1,0.2,0.2,0,0,-0.2])
        
        
    return df_tmp
    
    


# Consolidated function
def df_scaled_transform(df):
    st_empty = st.empty() # Progress tracker through the transformations
    
    # If later on there is a need to remove explicit songs during the solving stage
    if ('use_explicit' not in st.session_state) or (st.session_state.use_explicit==True): 
        df_tmp = df.copy() # Copy df to avoid any linked df problems
    elif st.session_state.use_explicit == False: # If don't want explicit, filter out
        df_tmp = df[df['Explicit'] == False].copy()
    
    df_tmp = df_remove_true_dup(df_tmp,st_empty) # Remove true duplicates
    df_tmp = df_drop_columns(df_tmp,st_empty) # Drop some unnecessary columns
    
    # # KIV Genre for later improvements
    # if st.session_state.df_genre==True: # if genres are considered, vectorize ngrams
    #     df_tmp = df_vectorize_genres(df_tmp, st_empty, ngram=st.session_state.df_genre_ngrams)
    # else:
    #     # If genre vectorisation is not done, need to remove these two columns
    df_tmp = df_tmp[[col for col in df_tmp.columns if col not in ['Artist','Genres']]]
    
    
    df_tmp = df_scale_numfeat(df_tmp, st_empty) # scale numerical features
    df_tmp = df_mood_vectors(df_tmp, st_empty) # create mood vectors
    
    st_empty.empty()
    return df_tmp
    
    


# 1. Cosine Similarity
def cos_sim(df):
    track_sim = pd.DataFrame(cosine_similarity(df.set_index('track_uri')),
                             columns=df.set_index('track_uri').index,
                             index=df.set_index('track_uri').index)
    
    return track_sim



def get_sim_list(df_scaled,target_vector):
    # All the target vectors are on a scale from 0 to 1; in other words, we can view these as being the 
    # probability density of the target point in the scaled df. Hence, we can scale them using an 
    # inverse-normal function.
    for col in [i for i in target_vector.keys() if i != 'track_uri']:
        target_vector[col] = round(norm.ppf(target_vector[col],loc=df_scaled[col].mean(),scale=df_scaled[col].std()),6)

    # Adding the target vector to the cosine similarity df
    target_df = pd.DataFrame(target_vector,index=[0])

    # concat, and fillna with 0 (for standard scaled vectors, 0 should be mean value)
    df_vector = pd.concat((df_scaled,target_df),ignore_index=True).fillna(0)

    # Remove all unnecessary columns
    df_vector = df_vector[target_vector.keys()]
    
    # Applying cosine similarity
    track_sim = cos_sim(df_vector)
    
    # Return list of track uris most similar to the target vector
    track_order = track_sim['target'].sort_values(ascending=False).drop(index='target').index
    return track_order

    


#|---------------|
# Start up screen
#|---------------|


sp = st.session_state.sp # Define spotify object in main flow
if "last_url" not in st.session_state:   st.session_state["last_url"] = "" # Setting up url cache for logic
if "df" not in st.session_state:         st.session_state["df"] = None # df dummy object
if "df_display" not in st.session_state: st.session_state["df_display"] = None # df_display dummy object
if "default_url" not in st.session_state: st.session_state["default_url"] = '''
https://open.spotify.com/playlist/0nsumiNqS4XUbc87OoMmd0?si=bd83e70c8e0e48b7&pt=1b5dd3875f6f7cba86f057dad6cde0f9
''' # Default playlist link - for reference

# Title
st.title('A More Intuitive Playlist Generator for Spotify')

# Subtext
st.markdown(f'''
Welcome **{st.session_state.username}**! 
This app aims to help create intuitive playlists from an existing song base. Right now, the 
app works by sequencing a playlist of set lengthbased on target meta-features from a pool, 
using a larger playlist as its base.
''')

# List of upcoming improvements
improvements = st.expander('Future Upgrades:')            
improvements.markdown('''
    1. Create a playlist that follows a 'mood curve'!
    2. Sequencing playlist based on user's liked songs!
    3. Text-based hype/vibe assignment!
    4. Sequencing based on a song or artist seed!
    5. Pull from more than one playlist!
    6. and more!
    ''')


#|-------------------|
# INPUT: Playlist URL
#|-------------------|
# Playlist URL Input, and form submission
with st.form('playlist_input'):

    # Assign playlist_url null value to session state
    if 'playlist_url' not in st.session_state: st.session_state.playlist_url = ''

    # INPUTS:
    # Playlist URL
    url_tmp = st.text_input('To start, please input a playlist url (Default will use my playlist!):',
                            value=st.session_state.playlist_url)
    
    # Formatting so that the buttons appear in the right spot
    col1, col2, col3, dummy = st.columns([1,0.8,1,5], gap='small')
    with col1: pl_submitted = st.form_submit_button('Submit')
    with col2: pl_clear = st.form_submit_button('Clear')
    with col3: pl_default = st.form_submit_button('Default')

    
    # Buttons - Submit, Clear, and Default
    if pl_submitted:
        
        st.session_state.playlist_url = url_tmp
        st.session_state.playlist_track_total = get_track_total(st.session_state.playlist_url)
        if st.session_state.playlist_url != st.session_state.default_url:
            st.session_state.use_default = False # Don't use default dataset
        else: st.session_state.use_default = True # Don't use default dataset
        # st.session_state.sidebar_state = 'expanded' # Set sidebar to open
        st.experimental_rerun()


    elif pl_clear:
        
        st.session_state.playlist_url = ''
        if 'df' in st.session_state: st.session_state.df = None
        st.session_state.use_default = False # Don't use default dataset
        st.experimental_rerun()
        
        
    elif pl_default:
        
        # Playlist that was used in the data exploration
        st.session_state.playlist_url = st.session_state.default_url
        st.session_state.playlist_track_total = get_track_total(st.session_state.playlist_url)
        st.session_state.use_default = True # Use default dataset - pull from .csv
        st.experimental_rerun()

    


#|-----------------------------------------------|
# INPUT: Number of tracks to scrape from the pool
#|-----------------------------------------------|
if ('playlist_url' in st.session_state) and (st.session_state.playlist_url != ''):
       
    # Number of Tracks
    with st.form('playlist_track_input'):
        tracks = st.number_input(
            f'Number of tracks to pool from (max tracks:{int(st.session_state.playlist_track_total)})',
            min_value = 0, 
            max_value = int(st.session_state.playlist_track_total),
            value = int(st.session_state.playlist_track_total),
            step=1)
        
        # # Toggle for whether to include genres in similarity solving
        # df_genre_but = st.checkbox('Include Genres?',value=True)
        # df_genre_ngrams = st.selectbox('Additional Functionality: Choose N-grams for genre sorting',
        #                                options=['1 word vectors','1-2 word vectors','1-3 word vectors'],
        #                                index=2)
        
        # Submit Button
        track_submitted = st.form_submit_button('Submit')
    
        # When submit is pressed
        if track_submitted:
            st.session_state.tracks = str(tracks)
            # st.session_state.df_genre = df_genre_but
            
            # if df_genre_ngrams == '1 word vectors': st.session_state.df_genre_ngrams = (1,1)
            # elif df_genre_ngrams == '1-2 word vectors': st.session_state.df_genre_ngrams = (1,2)
            # elif df_genre_ngrams == '1-3 word vectors': st.session_state.df_genre_ngrams = (1,3)
            
            # Generate scraped dataframe of playlist if not already 
            if (st.session_state.playlist_url != st.session_state.last_url) or (
                'df' not in st.session_state) or (st.session_state.df is None):
                
                # We will need 3 dataframes; 1) main (base), 2) formatted for display, 
                # and 3) transformed for similarity vector solving
                
                # 0) Scrape original playlist features
                
                st.spinner('Analysing similarities...') # Spinner to indicate start of transform
                url = st.session_state.playlist_url
                df = df_scrape(url)
                # Assigning df to session state
                st.session_state.df = df
                
                # 1) Formatted df for display

                display_features = ['Name','Artist','Album','Popularity','Explicit','Genres']
                st_empty = st.empty()
                df_display = df_remove_true_dup(df,st_empty) # Remove true duplicates
                df_display = df[display_features].copy()
                st.session_state.df_display = df_display
                st_empty.empty()

                st.session_state.last_url = url
            
            st.experimental_rerun()
            
            
    


#|----------------------|
# Initialise the sidebar
#|----------------------|
# Checks: playlist_url exists, and isn't blank, and playlist_tracks exists, and isn't blank
if ('playlist_url' in st.session_state) and ( # playlist_url exists
    st.session_state.playlist_url != '')and ( # playlist_url is not blank
    st.session_state.playlist_url == st.session_state.last_url) and ( # playlist_url and last_url are the same
    'tracks' in st.session_state) and ( # track exists
    st.session_state.tracks != int(0)) and ( # number of tracks is not 0
    st.session_state.df is not None): # df exists


    # See function for parameters
    sidebar_params(st.session_state.df)



    # Print the dataframe based on current target parameters
    if st.session_state.df is not None: 
        st.write('Resultant playlist!')
        st.dataframe(st.session_state.df_display)

        with st.form('playlist deployment'):

            # Text input for naming your playlist!
            playlist_name = st.text_input('Name your new playlist!')
            playlist_submit = st.form_submit_button('CREATE')

            if playlist_submit:
                sp = st.session_state.sp
                playlist_description = 'Created using the Spotify Playlist Generator!'

                # Creates an empty playlist
                pl_summary = sp.user_playlist_create(user=st.session_state.userid,
                                                     name=playlist_name,
                                                     description=playlist_description)

                # Update playlist with songs!
                for rep in range(int((len(st.session_state.track_list)-1)/100)+1):
                    # iterate since spotify's API only allows adding 100 songs at a time
                    sp.playlist_add_items(playlist_id=pl_summary['uri'],
                                          items=st.session_state.track_list[
                                              100*rep:min(100*(rep+1),len(st.session_state.track_list))])

                st.success('Playlist Created!')
                if dt.datetime.now().date().month >= 11: st.snow()
                else: st.balloons()

                

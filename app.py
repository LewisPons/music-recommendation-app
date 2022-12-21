
import streamlit as st
import pandas as pd
from PIL import Image
from utils import call_playlist, music_recomendation, get_songs_visuals, SONG_FEATURES, visualize_cover_art
from streamlit_extras.no_default_selectbox import selectbox
import codecs
import os

CLIENT_ID = "da82a340bd1341599bd3c590de6ad9fa"
CLIENT_SECRET = "2c8640b95bc14b3b8edeab8585bf13af"

st.set_page_config(page_title="Music Recomendation App")

# CLIENT_ID = os.environ['client_id']
# CLIENT_SECRET = os.environ['client_secret']

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

image = Image.open('spotify_logo.png')

st.sidebar.image(image , caption="Spotify Recomendation System",width = 256)
app_mode = st.sidebar.selectbox("Choose app mode", ["Run App", "Obtain your Spotify URI","About Me"])

@st.cache
def read_full_dataset(path: str = 'popular_music.parquet.gzip') -> pd.DataFrame:
    return pd.read_parquet(path)



if app_mode == 'Run App':
    st.title('Music Recomendation System')
    markdown = """
    This is a first-version Content Based Music Recommendation System.\n
    The model was trained based on this [dataset](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks).\n
    \n
    To start generating recomendations:
    1. Open Spotify.
    2. Select your favourite Playlist.
    3. Copy the playlist URL (It should be something like this (https://open.spotify.com/playlist/7sY3tKQa40BXziDEer3m3I?si=89f0cffef02e41ec).
    4. Paste the URL into the text box.
    5. Select the song you want recomendations for.
    6. Have fun!
    """
    
    st.markdown(markdown)

    playlist_link = st.text_input('**Input your playlist URI**')
    
    
    dataset = read_full_dataset()
    
    if (len(playlist_link) != 0) & ('playlist' in playlist_link):
        df = call_playlist(playlist_link, CLIENT_ID, CLIENT_SECRET)
        
        st.dataframe(df)
        songs_list = df['track_name'].to_list()
        selected_song = selectbox('Select a song to generate recomendations:', songs_list)
        
        if selected_song is not None:
            selected_song_df = df[df['track_name'] == selected_song]
            st.write('Selected Song:', selected_song_df)
            
            selected_song_track_id = selected_song_df['track_id'].values[0]
            selected_song_track_name = selected_song_df['track_name'].values[0]
            

            dataset = dataset[ ~(dataset['id'] == selected_song_track_id) & ~(dataset['name'] == selected_song_track_name)]            
            recomendations_df = music_recomendation(dataset, selected_song_df)
            recomendations_df = get_songs_visuals(recomendations_df, CLIENT_ID, CLIENT_SECRET)
            
            st.subheader('Recomendations generated:')
            st.write('Check out the top 3 songs for you!', recomendations_df) 
            
            with st.container():
                col1, col2 = st.columns([3,1])

                with col1:
                    st.image(recomendations_df['url'][0])
                with col2:
                    st.markdown(f"""**Song:** {recomendations_df['name'][0]}""")
                    st.markdown(f"""**Artists**: {recomendations_df['artists'][0].replace('[','').replace(']','').replace("'",'')}""")
                    
            with st.container():
                col1, col2 = st.columns([3,1])

                with col1:
                    st.image(recomendations_df['url'][1])
                with col2:
                    st.markdown(f"""**Song:** {recomendations_df['name'][1]}""")
                    st.markdown(f"""**Artists**: {recomendations_df['artists'][1].replace('[','').replace(']','').replace("'",'')}""")
                    
            with st.container():
                col1, col2 = st.columns([3,1])

                with col1:
                    st.image(recomendations_df['url'][2])
                with col2:
                    st.markdown(f"""**Song:** {recomendations_df['name'][2]}""")
                    st.markdown(f"""**Artists**: {recomendations_df['artists'][2].replace('[','').replace(']','').replace("'",'')}""")

            
    elif len(playlist_link) == 0:
        st.write('Please Insert a link :)')
        
    else:
        st.write('Please check the Playlist link is correct :)')
        
elif app_mode == "About Me":
    st.title('Music Recomendation System')
    mkdn = """
    Hi! My name is Luis Morales, I am a Data Engineer at Wizeline and B.Sc in Actuarial Sciences. \n
    I love working in Cloud and feel passionated about Machine Learning and AI!.
    """
    st.markdown(mkdn)
    st.success("Feel free to contacting me here 👇 ")

    col1,col2,col3,col4 = st.columns((2,1,2,1))
    col1.markdown('* [**LinkedIn**](https://www.linkedin.com/in/luis-morales-ponce/)')
    col1.markdown('* [**GitHub**](https://github.com/LewisPons)')
    image2 = Image.open('profile.jpeg')
    st.image(image2)

elif app_mode =="Obtain your Spotify URI":
    stepflow_file = codecs.open("SpotifyWorkflow.html", "r", "utf-8")
    st.markdown(stepflow_file.read(), unsafe_allow_html=True)


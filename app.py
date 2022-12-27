import streamlit as st
import pandas as pd
from PIL import Image
from utils import call_playlist, music_recomendation, get_songs_visuals, SONG_FEATURES, visualize_cover_art
from streamlit_extras.no_default_selectbox import selectbox
import plotly.express as px
import codecs
import os


st.set_page_config(page_title="Music Recomendation App")

# CLIENT_ID = os.environ['client_id']
# CLIENT_SECRET = os.environ['client_secret']
CLIENT_ID="da82a340bd1341599bd3c590de6ad9fa"
CLIENT_SECRET="2c8640b95bc14b3b8edeab8585bf13af"
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

image = Image.open('spotify_logo.png')

st.sidebar.image(image , caption="Spotify Recomendation System",width = 256)
app_mode = st.sidebar.selectbox("Choose app mode", ["Generate Recomendations", "About the Dataset & EDA","Obtain your Spotify URI","About Me"])

@st.cache
def read_full_dataset(path: str = 'popular_music.parquet.gzip') -> pd.DataFrame:
    return pd.read_parquet(path)



if app_mode == "Generate Recomendations":
    st.title('Music Recomendation System')
    markdown = """
    This is a first-version Content Based Music Recommendation System.\n
    The model was trained based on this [dataset](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks).\n
    \n
    To learn more about the model's training dataset, the EDA phase and the distribution of the features, click on the tab **About the Dataset**.
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
    
    if (len(playlist_link) != 0) : #& ('playlist' in playlist_link):
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
                    st.markdown(f"""**Artists**: [{recomendations_df['artists'][0].replace('[','').replace(']','').replace("'",'')}]({recomendations_df['artist_link'][0]})""")
                    
            with st.container():
                col1, col2 = st.columns([3,1])

                with col1:
                    st.image(recomendations_df['url'][1])
                with col2:
                    st.markdown(f"""**Song:** {recomendations_df['name'][1]}""")
                    st.markdown(f"""**Artists**: [{recomendations_df['artists'][1].replace('[','').replace(']','').replace("'",'')}]({recomendations_df['artist_link'][1]})""")
                    
            with st.container():
                col1, col2 = st.columns([3,1])

                with col1:
                    st.image(recomendations_df['url'][2])
                with col2:
                    st.markdown(f"""**Song:** {recomendations_df['name'][2]}""")
                    st.markdown(f"""**Artists**: [{recomendations_df['artists'][2].replace('[','').replace(']','').replace("'",'')}]({recomendations_df['artist_link'][2]})""")

            
    elif len(playlist_link) == 0:
        st.write('Please Insert a link :)')
        
    else:
        st.write('Please check the Playlist link is correct :)')
    
    
elif app_mode == "About the Dataset & EDA":
    tab0, tab1, tab2 = st.tabs(["EDA", "Interesting Findings", "Metadata Details"])

    with tab1:
        
        music_df = pd.read_csv('music_evolution.csv')
        fig = px.line(music_df, 
            x='release_date', 
            y=['acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'valence', 'speechiness']
        )
        fig.update_layout(
            title="Evolution of Music since 1920",
            xaxis_title="Value",
            yaxis_title="Release Year",
            legend_title="Feature"
        )
        
        st.plotly_chart(fig, theme="streamlit", use_conatiner_width=True)
        music_evolution_mkd = """
        He have many intersting insights about this the evolution of musical features through the years:
        - Note `acousticness` and `instrumentalness` has the largest fall through the years, this may be because the majority of recorded tracks in the 20s decade were clasical and jazz genres.
        - `danceability` and `energy` features have increased across the time, this should be cause of the rise of computers and new technologies in the musical industry, letting musics create new styles beats and rhythms without having to perform it themselves.
        - `valence` and `liveness` have matained the tendency in the time.
        """
        st.markdown(music_evolution_mkd)

    with tab0:
        st.header('Lets take a look to the EDA phase:')
        st.markdown("""
        The [dataset](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks?select=artists.csv) was cleaned before performing the analysis.\n
        Some of the prepecessing steps were:
        1. Parsing dates.
        2. Converting atring list into List objects.
        3. Renaming columns.
        4. Feature engineering between redundant columns.
        5. Drop Null values.
        """
        )
        st.markdown("After the cleansen, the dataframe contained this columns:")
        st.code("""Index(['id', 'name', 'popularity', 'duration_ms', 'explicit', 'artists',
       'id_artists', 'release_date', 'danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo', 'time_signature', 'followers',
       'genre_artist', 'name_artist', 'popularity_artist', 'duration_mins'],
      dtype='object')""")
        st.markdown("""---""")
        st.subheader("Distribution Analysis")
        
        popularity_image = Image.open('statistical_analysis/popularity.png')
        st.image(popularity_image)
        st.markdown(""" - `popularity`: The popularity of the artist. The value will be between 0 and 100, with 100 being the most popular. The artist's popularity is calculated from the popularity of all the artist's tracks. We can see there are more than 40,000 artists in the dataset that have the lowest popularity score. The mean is 27.5 and the median 27. \n""")
        st.markdown("""---""")
        
        
        duration_mins_image = Image.open('statistical_analysis/duration_mins.png')
        st.image(duration_mins_image)
        st.markdown("""
        - Notice the datset has a large range, as the minimum song observed it 0.05 mins and the maximum value 93.68 minutes long. That is why our dataset histogram has a large range
        - While our 3rd Quartile is ~4.39 mins.
        
        - In our dataset, we have ~ 95% of Non explicit songs, while < 5% are explicit
        \n
        """
        )
        st.markdown("""---""")
        
        
        st.markdown("""`danceability`: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.""")
        danceability_image = Image.open('statistical_analysis/danceability.png')
        st.image(danceability_image)
        st.markdown("""Looks like most of our sample is quite danceable!""")
        st.code("""danceability_IQR = complete_data['danceability'].quantile(.75) - complete_data['danceability'].quantile(.25)
        rank = complete_data['danceability'].max() - complete_data['danceability'].min()
        danceability_IQR, rank
        
        > (0.23300000000000004, 0.991)
        """)
        st.success("Note that Q2 is 0.45 and Q3 is 0.68, most of the songs in this dataset are danceable. The IQR is 0.233!")
        st.markdown("""---""")
        
        
        st.markdown("""- `energy`: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.""")
        energy_mins_image = Image.open('statistical_analysis/energy.png')
        st.image(energy_mins_image)
        st.markdown("This distribution is interesting! Note that the energy feature looks like a downward parabola!")        
        st.markdown("""---""")
        
        
        st.markdown("""- `key`: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.""")
        key_mins_image = Image.open('statistical_analysis/key.png')
        st.image(key_mins_image)
        st.markdown("Note that the most used keys in our sample are: 0 = C, 2 D, 7 = G and 9 = A")
        st.markdown("""---""")
        
        
        
        st.markdown("""- `loudness`: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.""")
        loudness_mins_image = Image.open('statistical_analysis/loudness.png')
        st.image(loudness_mins_image)
        st.markdown("This distribution is quite left skewed. Also in the boxplot we can note many outliers")
        st.markdown("""---""")
        

        st.markdown("""-  `acousticness`: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.""")
        acousticness_mins_image = Image.open('statistical_analysis/acousticness.png')
        st.image(acousticness_mins_image)
        st.markdown("""The distribution looks quite uniform between 0.09 and 0.9, white we have heavy tails en the extreme values!
                    
                    There are manys songs that are un-acoustic!""")
        st.markdown("""---""")
        
            
        st.markdown("""- `liveness`: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.""")
        liveness_mins_image = Image.open('statistical_analysis/liveness.png')
        st.image(liveness_mins_image)
        st.markdown("""Looks like most of our tracks were not perfomanced live.""")
        st.success("""There are 14312 that have a `liveness` value > 8, provides strong likelihood that the track is live.""")
        st.markdown("""---""")
        
        
        st.markdown("""-  `valence`: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).""")
        valence_mins_image = Image.open('statistical_analysis/valence.png')
        st.image(valence_mins_image)
        st.markdown("""There is a peak ot the valence feature, this is quite strange, as the entire variable is quite uniform across the possible values.
                    - Looks like there are few more tracks with a positive sound, than a negative one.""")
        st.markdown("""---""")
        
        
        st.markdown("""- `time_signature`: An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of "3/4", to "7/4".""")
        st.markdown("""We have: \n
                    - 85.8% of tracks in 4/4 signature
                    - 10.9% of tracks in 3/4 signature
                    - 1.9% of tracks in 5/4 signature
                    - 1.6% of tracks in with a rara signature, this may be untrusty data
        """)
        
        st.subheader("Lets see Linear relationship between variables")
        heatmap_image = Image.open('statistical_analysis/heatmap.png')
        st.image(heatmap_image)
        st.markdown("""Some important points here:
        - As `duration_mins` is a linear combiantion of `duration_ms`, we have a $\rho$ = 1. Remember  $duration_{mins} = \frac{duration_{ms}}{60000}$. Then, do not let this value makes noise.
        - Some variables are redundants, as one is defined similarly as other, notice `popularity` and `popularity_artist` have a high correlation, this is for how the variables are defined. This also happens with `followers` and `popularity_artist`
        """)
        st.markdown("Let's obtain all the pair-columns that have weak, moderate or strong correlation, either negative or positive")
        st.code("""interesting_correlations_df =  top_correlation[(top_correlation['top_correlations'] > 0.3) | (top_correlation['top_correlations'] < -0.3)]""")
        st.code("""interesting_correlations_df""")
        st.markdown("""|                                     |   top_correlations |\n|:------------------------------------|-------------------:|\n| ('duration_ms', 'duration_mins')    |           1        |\n| ('energy', 'loudness')              |           0.764735 |\n| ('popularity', 'popularity_artist') |           0.529741 |\n| ('danceability', 'valence')         |           0.52815  |\n| ('followers', 'popularity_artist')  |           0.423432 |\n| ('energy', 'valence')               |           0.372276 |\n| ('popularity', 'loudness')          |           0.327028 |\n| ('popularity', 'energy')            |           0.302315 |\n| ('loudness', 'instrumentalness')    |          -0.329306 |\n| ('popularity', 'acousticness')      |          -0.370882 |\n| ('loudness', 'acousticness')        |          -0.519423 |\n| ('energy', 'acousticness')          |          -0.715412 |""")
        
    with tab2:
        dataset_markdown = """
        The purpose of this project is to build a **songs recommendation system**, based on the user's music preference.\n
        
        This dataset contains different features of 600k+ tracks.
        1. `tracks.csv`: The audio features of tracks.
        2. `artists.csv`: The popularity metrics of artists.

        All data collected in this dataset was obtained through Spotify API by Yamac Eren .


        The dataset was obtained in [kaggle](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks?select=artists.csv)\n
        
            
        - `id`: The Spotify ID for the track.

        - `name`:The song's name.

        - `popularity`: The popularity of the artist. The value will be between 0 and 100, with 100 being the most popular. The artist's popularity is calculated from the popularity of all the artist's tracks.

        - `duration_ms`: The track length in milliseconds.

        - `explicit`: Whether or not the track has explicit lyrics ( true = yes it does; false = no it does not OR unknown).

        - `artists`: The artist's name.

        - `id_artists`: The Spotify artist ID for the track.

        - `release_date`: 

        - `danceability`: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.

        - `energy`: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.

        - `key`: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.

        - `loudness`: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.

        - `mode`: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.

        - `speechiness`: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.

        - `acousticness`: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.

        - `instrumentalness`: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.

        - `liveness`: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.

        - `valence`: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

        - `tempo`: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.

        - `time_signature`: An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of "3/4", to "7/4".

        For more in-depth information about audio features provided by Spotify: https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features
        
        
        
        """
        st.markdown(dataset_markdown)

elif app_mode == "About Me":
    st.title('About Me')
    mkdn = """
    Hi there! My name is **Luis Morales** and I am a Data Engineer with a B.Sc in Actuarial Sciences. I currently work at Wizeline, where I am responsible for designing, building, and maintaining data pipelines that move and transform data from various sources.

    I have a strong foundation in mathematics and statistical analysis, which has served me well in my career as a data engineer. I am also passionate about cloud computing and am always eager to learn more about machine learning and artificial intelligence.
    Thank you for taking the time to learn a little bit about me. 
    """
    st.markdown(mkdn)
    st.success("Feel free to contact me here 👇 ")

    col1,col2,col3,col4 = st.columns((2,1,2,1))
    col1.markdown('* [LinkedIn](https://www.linkedin.com/in/luis-morales-ponce/)')
    col1.markdown('* [GitHub](https://github.com/LewisPons)')
    image2 = Image.open('profile.jpeg')
    st.image(image2, width=400)


elif app_mode =="Obtain your Spotify URI":
    stepflow_file = codecs.open("SpotifyWorkflow.html", "r", "utf-8")
    st.markdown(stepflow_file.read(), unsafe_allow_html=True)


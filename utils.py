import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import pandas as pd
import numpy as np 
from skimage import io
import urllib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


SONG_FEATURES = ['danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo', 'time_signature']


def spotipy_authorize(cid: str, secret: str):
    """
        Get Spotify API Credentials manager
    """
    sp = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=cid, 
            client_secret=secret
        )
    )
    return sp


def call_playlist(playlist_uri : str, cid : str, client_secret : str, creator='spotify') -> pd.DataFrame:
    sp = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=cid, 
            client_secret=client_secret
        )
    )

    playlist_features_list = ['artist', 'album', 'track_name', 'track_id','url', 'artist_popularity', 'artist_genre', 'danceability',
                                'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness','instrumentalness', 'liveness', 
                                'valence', 'tempo', 'duration_ms', 'time_signature']
    
    playlist_df = pd.DataFrame(columns = playlist_features_list)
    artist_df = pd.DataFrame(columns=['artist_popularity', 'artist_genre'])
    
    playlist = sp.user_playlist_tracks(creator, playlist_uri)["items"]
    for track in playlist:
        playlist_features = {}
        artist_features = {}
        # Get metadata
        playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
        playlist_features["album"] = track["track"]["album"]["name"]
        playlist_features["track_name"] = track["track"]["name"]
        playlist_features["track_id"] = track["track"]["id"]
        playlist_features["url"] = track['track']['album']['images'][1]['url']
        
        artist_uri = track["track"]["artists"][0]["uri"]
        artist_info = sp.artist(artist_uri)
        
        
        artist_features["artist_popularity"] = artist_info["popularity"]
        artist_features["artist_genre"] = str(artist_info["genres"])
        popularity = (artist_info["popularity"])
        genres = str(artist_info["genres"])
        # Get audio features
        audio_features = sp.audio_features(playlist_features["track_id"])[0]
        for feature in playlist_features_list[7:]:
            playlist_features[feature] = audio_features[feature]
            
        # Concat the 
        artist_df = pd.DataFrame(artist_features, index=[0])
        track_df = pd.DataFrame(playlist_features, index = [0])
        
        artist_track_df = pd.concat([track_df, artist_df], axis=1)
        
        playlist_df = pd.concat([playlist_df, artist_track_df], ignore_index = True)
            
    return playlist_df


def music_recomendation(dataset: pd.DataFrame, song_df: pd.DataFrame):
    numerical_features_dataset = dataset[SONG_FEATURES]
    numerical_features_song = song_df[SONG_FEATURES]
    
    similarity = cosine_similarity(numerical_features_dataset, numerical_features_song)
    similarity_df = pd.DataFrame(similarity, columns=['cosine_similarity'])
    
    results = pd.concat([dataset, similarity_df], axis=1)
    
    return results.sort_values('cosine_similarity', ascending=False).reset_index(drop=True).head()


def get_songs_visuals(df: pd.DataFrame, cid: str, client_secret: str) -> pd.DataFrame:
    sp = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=cid, 
            client_secret=client_secret
        )
    )
    ids = df['id'].to_list()
    urls = []
    artist_links = []
    
    for id in ids:
        urn = f'spotify:track:{id}'
        track_url = sp.track(urn)['album']['images'][0]['url']
        artist_link  = sp.track(urn)['album']['artists'][0]['external_urls']['spotify']
    
        urls.append(track_url)
        artist_links.append(artist_link)
    
    df['url'] = urls
    df['artist_link'] = artist_links
    return df


def visualize_cover_art(playlist_df):
    temp = playlist_df['url'].values
    plt.figure(figsize=(15,int(0.625 * len(temp))))
    columns = 5
    
    for i, url in enumerate(temp):
        plt.subplot(int(len(temp) / columns + 1), columns, i + 1)

        image = io.imread(url)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        s='' 
        plt.xlabel(s.join(playlist_df['track_name'].values[i].split(' ')[:4]), fontsize = 10, fontweight='bold')
        plt.tight_layout(h_pad=0.8, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)
    
    plt.savefig('recommended_songs.png', dpi=400)
    
def download_image(df: pd.DataFrame):    
    urls = df['url'].to_list()
    songs = df['track_name'].to_list()
    for i in range(urls):
        urllib.request.urlretrieve(urls[i], f"./tmp/{songs[i]}-filename.jpg")


def summary_statistic_plot(df: pd.DataFrame, column: str, **kwargs):
    """
    Function to plot  a histogram, a boxplot and the descriptive statistics into one single plot
    
    Parameters:
    ----------
        df : pd.DataFrame
        column: str
            column that has a numeric data type

    """
    sns.set(style="darkgrid")
    
    splits = str(df[column].describe()).split()
    
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    sns.histplot(x = column, data = df, ax=ax_hist, **kwargs)
    sns.boxplot(x = column, data=df, ax=ax_box)
    
    plt.figtext(.95, .20, describe_helper(df[column])[0], {'multialignment':'left'})
    plt.figtext(1.05, .20, describe_helper(df[column])[1], {'multialignment':'right'})
    plt.show()
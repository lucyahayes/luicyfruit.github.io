# What is My Music Taste?

## An analysis using the Spotify API

Whenever anyone asks me, "so what type of music do you listen to?", my usual answer is "... uh everything?". Having grown up with a very intense classical music training, my music taste ranges from Dvorak Symphonies, to early 2000's punk rock, to jazz ensembles. This project aims to quantify what type of music I listen to according to my Spotify playlists, to see if there are any features or combinations of features that make a song a "Lucy" song, and to build a predictor model on how likely I would be to like a song based on those features. I will be using the **Spotify API** and the spotipy package to access my own user data, and to access public data. 

# 1. Data Acquisition

## 1.1 Import Libraries
```import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import simplejson as json
import time
import sys
import pandas as pd
from dateutil.parser import parse as parse_date
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
%matplotlib inline
sns.set_palette("husl")
```

## 1.1 Accessing the Spotify API Credentials Flow

Since I am accessing my own personal data, I used my credentials that are provided by your spotify web app. For different types of access, the API requires different Scopes to be used. This will require going through the authorization flow a few times in order to collect all relevant data. The first scope I will use is "playlist-read-collaborative", which will read in all of my playlists including those that are collaborative
```scope = 'playlist-read-collaborative'
token = util.prompt_for_user_token(username, scope, client_id = cid, client_secret = secret, redirect_uri = url)
sp = spotipy.Spotify(auth=token)
```
You are then prompted by your browser to enter the URL redirect (which I just set as 'http://localhost:8888') Once in, you can get started on reading in your library

## 1.3 Reading in Playlist Tracks

My method for reading in the different playlist tracks was to first make a list of all playlists, and then writing a function that takes in a playlist, and returns a dataframe of all of the corresponding tracks and their information. Next, I add all those to a main dataframe to start building out my library. The method was as follows:

```
playlists = sp.user_playlists(username)
playlist_list = []
for i, playlist in enumerate(playlists['items']):
    playlist_list.append(playlist['uri'])
```

```
# Function to read in the Tracks and some Attributes of each Playlist
def read_playlist(user, playlistID):
    c = ['uri', 'artist', 'name', 'popularity', 'release_date', 'added_at']
    playlist = sp.user_playlist(user, playlistID)
    tracks = playlist['tracks']['items']

    tracks_df = pd.DataFrame([(track['track']['uri'],
                           track['track']['artists'][0]['name'],
                           track['track']['name'],
                            track['track']['popularity'],
                           parse_date(track['track']['album']['release_date']) if track['track']['album']['release_date'] else None,
                           parse_date(track['added_at']))
                          for track in tracks], columns = c)
    return tracks_df
```

```
# Append all tracks into a singular Dataframe
playlist_df = pd.DataFrame(columns=['uri', 'artist', 'name', 'popularity', 'release_date', 'added_at'])
for p in playlist_list:
    y = read_playlist(username, p)
    playlist_df = playlist_df.append(y, ignore_index = True)
```
Now we have something that looks like this:
|    | uri                                  | artist         | name                                                          |   popularity | release_date        | added_at                  |
|---:|:-------------------------------------|:---------------|:--------------------------------------------------------------|-------------:|:--------------------|:--------------------------|
|  0 | spotify:track:6xEHCWUvalb0fNYuAo591v | Rob Araujo     | Nineteen                                                      |           35 | 2018-10-12 00:00:00 | 2019-07-09 20:22:42+00:00 |
|  1 | spotify:track:6v96ZIpQUtWMSUqlBlTif6 | Rob Araujo     | Hike                                                          |           31 | 2018-04-03 00:00:00 | 2019-07-10 14:12:50+00:00 |
|  2 | spotify:track:62VWmsNoDmqT0Mj9oHHFVh | Roy Hargrove   | Strasbourg / St. Denis                                        |           49 | 2008-01-01 00:00:00 | 2019-07-10 15:01:41+00:00 |
|  3 | spotify:track:1W97IZUEKOaIVxG7GKJkL6 | Anomalie       | Velours                                                       |           46 | 2017-06-23 00:00:00 | 2019-07-10 17:28:40+00:00 |
|  4 | spotify:track:0sCeNwt8xRCMR4NhKpMyBe | Herbie Hancock | Cantaloupe Island - Remastered 1999 / Rudy Van Gelder Edition |           61 | 1964-06-17 00:00:00 | 2019-07-10 17:30:22+00:00 |

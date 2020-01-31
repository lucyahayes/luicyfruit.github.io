---
layout: post
date: 2020-01-31
title: "What is my Music Taste?"
subtitle: "An analysis using the Spotify API"
header-img: "img/music.png"
---

Whenever anyone asks me, "so what type of music do you listen to?", my usual answer is "... uh everything?". Having grown up with a very intense classical music training, my music taste ranges from Dvorak Symphonies, to early 2000's punk rock, to jazz ensembles. This project aims to quantify what type of music I listen to according to my Spotify playlists, to see if there are any features or combinations of features that make a song a "Lucy" song, and to build a predictor model on how likely I would be to like a song based on those features. I will be using the **Spotify API** and the spotipy package to access my own user data, and to access public data. 

![Bassnectar](https://luicyfruit.github.io/img/bassnectar.jpg){:height="100px" width="100px"} ![SteelyDan](https://luicyfruit.github.io/img/steelydan.jpg){:height="100px" width="100px"} ![Deadmau5](https://luicyfruit.github.io/img/deadmau5.jpg){:height="100px" width="100px"}

# 1. Data Acquisition

## 1.1 Import Libraries
```python
import spotipy
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

## 1.2 Accessing the Spotify API Credentials Flow

Since I am accessing my own personal data, I used my credentials that are provided by your spotify web app. For different types of access, the API requires different Scopes to be used. This will require going through the authorization flow a few times in order to collect all relevant data. The first scope I will use is "playlist-read-collaborative", which will read in all of my playlists including those that are collaborative
```python
scope = 'playlist-read-collaborative'
token = util.prompt_for_user_token(username, scope, client_id = cid, client_secret = secret, redirect_uri = url)
sp = spotipy.Spotify(auth=token)
```
You are then prompted by your browser to enter the URL redirect (which I just set as 'http://localhost:8888') Once in, you can get started on reading in your library

## 1.3 Reading in Playlist Tracks

My method for reading in the different playlist tracks was to first make a list of all playlists, and then writing a function that takes in a playlist, and returns a dataframe of all of the corresponding tracks and their information. Next, I add all those to a main dataframe to start building out my library. The method was as follows:

```python
playlists = sp.user_playlists(username)
playlist_list = []
for i, playlist in enumerate(playlists['items']):
    playlist_list.append(playlist['uri'])
```

```python
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

```python
# Append all tracks into a singular Dataframe
playlist_df = pd.DataFrame(columns=['uri', 'artist', 'name', 'popularity', 'release_date', 'added_at'])
for p in playlist_list:
    y = read_playlist(username, p)
    playlist_df = playlist_df.append(y, ignore_index = True)
```
Now we have something that looks like this:

|    | uri                                  | artist         | name             | popularity | release_date| added_at   |
|----|--------------------------------------|----------------|------------------|------------|-------------|------------|
|  0 | spotify:track:6xEHCWUvalb0fNYuAo591v | Rob Araujo     | Nineteen         |         35 | 2018-10-12  | 2019-07-09 |
|  1 | spotify:track:6v96ZIpQUtWMSUqlBlTif6 | Rob Araujo     | Hike             |         31 | 2018-04-03  | 2019-07-10 |
|  2 | spotify:track:62VWmsNoDmqT0Mj9oHHFVh | Roy Hargrove   | Strasbourg       |         49 | 2008-01-01  | 2019-07-10 |
|  3 | spotify:track:1W97IZUEKOaIVxG7GKJkL6 | Anomalie       | Velours          |         46 | 2017-06-23  | 2019-07-10 |
|  4 | spotify:track:0sCeNwt8xRCMR4NhKpMyBe | Herbie Hancock | Cantaloupe Island|         61 | 1964-06-17  | 2019-07-10 |

Now that we have all of the basic track information, I wanted to obtain some of the audio features on each track. This uses the sp.audio_features function, which takes in a track uri and returns information such as energy, loudness, key, tempo, and more. I then wanted to take that information and store it in a dataframe, so I wrote a function that takes in a list of track uris, and returns a dataframe full of their relevant information

```python
def audio_features(uris):
    c = ['uri',
     'danceability', 
     'energy', 
     'key', 
     'loudness', 
     'mode', 
     'speechiness',
     'acousticness',
     'instrumentalness', 
     'liveness', 
     'valence', 
     'tempo', 
     'duration_ms', 
     'time_signature']

    df = pd.DataFrame(columns = c)   
    for uri in uris:
        x = sp.audio_features(uri)
        data = [{'uri': x[0]['uri'],
             'danceability': x[0]['danceability'],
             'energy': x[0]['energy'],
             'key': x[0]['key'],
             'loudness': x[0]['loudness'],
             'mode': x[0]['mode'],
             'speechiness': x[0]['speechiness'],
             'acousticness': x[0]['acousticness'],
             'instrumentalness': x[0]['instrumentalness'],
             'liveness': x[0]['liveness'],
             'valence': x[0]['valence'],
             'tempo': x[0]['tempo'],
             'duration_ms': x[0]['duration_ms'],
             'time_signature': x[0]['time_signature']
            }]
        df = df.append(data, ignore_index = True, sort = False)
    return df
```
When calling this on the track uris from the playlist data, we obtain a dataframe that looks like this:

|    | uri                                  |   danceability |   energy |   key |   loudness |   mode |   speechiness | ... | 
|----|--------------------------------------|----------------|----------|-------|------------|--------|---------------|-----|
|  0 | spotify:track:6xEHCWUvalb0fNYuAo591v |          0.487 |    0.787 |     5 |    -11.323 |      0 |        0.0945 |  ...| 
|  1 | spotify:track:6v96ZIpQUtWMSUqlBlTif6 |          0.567 |    0.377 |     0 |    -11.851 |      0 |        0.0783 |  ...| 
|  2 | spotify:track:62VWmsNoDmqT0Mj9oHHFVh |          0.701 |    0.445 |     1 |    -10.583 |      1 |        0.0834 |  ...| 
|  3 | spotify:track:1W97IZUEKOaIVxG7GKJkL6 |          0.541 |    0.465 |     1 |     -5.325 |      0 |        0.205  |  ...| 
|  4 | spotify:track:0sCeNwt8xRCMR4NhKpMyBe |          0.515 |    0.583 |     0 |     -8.182 |      0 |        0.0284 |  ...| 

Great! Now that we have the two separate sets of information (basic track info and audio features) for each track in my playlists, I will merge them into a larger dataframe on their uri.
```python
full = pd.merge(left = playlist_df, 
                right = features_df, 
                left_on = ['uri'], 
                right_on = ['uri'])
```
## 1.4 Reading in Top Tracks

This section reads in my top played tracks as I want to make sure that I'm not missing any music that I frequently listen to but isn't in my playlists. To start, this requires a new scope ('user-top-read') which requires me to go through the authorization flow again. 
```python
scope = 'user-top-read'
token = util.prompt_for_user_token(username, 
                                   scope, 
                                   client_id = cid, 
                                   client_secret = secret, 
                                   redirect_uri = url)
sp = spotipy.Spotify(auth=token)
```
The Top Tracks scope requires a time range, which are categorized as follows:
**Short Term** is the last four weeks, **Medium Term** is the last 6 months, and **Long Term** is over a few years of data. I used these time estimates as the "added_at" parameter to estimate when I first discovered that song. The API limit was 50 songs per time range, resulting in 150 songs added to the library

```python
# Possible Ranges
ranges = ['short_term', 'medium_term', 'long_term']

# List of URIs 
c = ['uri','added_at'] 
top_tracks = pd.DataFrame(columns = c)

# Get List of Top Tracks and add their estimated time
for r in ranges:
    if r == 'short_term': 
        d = parse_date('2020-01-01')
    elif r == 'medium_term':
        d = parse_date('2019-06-01')
    else: d = parse_date('2018-01-01')
        
    results = sp.current_user_top_tracks(time_range=r, limit=50)
    for i, item in enumerate(results['items']):
        data = [{'uri': item['uri'],
                'added_at': d}]
        top_tracks = top_tracks.append(data, ignore_index = True, sort = False)
```
Since the function I wrote previously took in a list of playlists, I have tweaked it to account for the dataframe of tracks instead. Since they are not in a playlist, I will use the date estimated above as the "added_at" date. 
```python
# Function to Read Tracks, rather than Playlists
def read_tracks(tracks_df):
    c = ['uri', 'artist', 'name', 'popularity', 'release_date', 'added_at']
    df = pd.DataFrame(columns = c)
    tracks = tracks_df['uri']
    for track in tracks:
        counter = 0
        t = sp.track(track)
        data = [{'uri': t['uri'],
        'artist': t['artists'][0]['name'],
        'name': t['name'],
        'popularity': t['popularity'],
        'release_date': parse_date(t['album']['release_date']) if t['album']['release_date'] else None,
        'added_at': tracks_df['added_at'][counter]}]
        df = df.append(data, ignore_index = True, sort = False)
        counter += 1
    return df
```
I call the read_tracks and audio_features functions on the top-tracks, merge those dataframes, and then append to the original playlist dataframe.

## 1.5 Read in Saved Tracks
Reading in saved tracks required a new scope again ('user_saved_tracks'). This followed the same steps as reading in my top tracks except this time there was an added_at date, so I didn't need to use estimates. I used the read_tracks and audio_features function on this data, and appended to the main dataframe. The API limit was 20 songs (even though I have many more saved!) so this didn't account for too many more songs, especially seeing how some were duplicates. 

Lastly, I removed those duplicates to get a dataframe with 570 entries, with the following variables (as defined in the [Spotify API Documentation](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/)):

- **Uri**: The Spotify URI for the track.
- **Artist**: The Artist who performed the track
- **Name**: The name of the Track
- **Popularity**:The popularity of the track. The value will be between 0 and 100, with 100 being the most popular.
The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are.
Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. 
- **Release Date**: Date of release
- **Added At**: Date saved, added to playlist, or estimated recently listening time
- **Danceability**: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
- **Energy**:Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
- **Key**: The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation 
- **Loudness**:	The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.
- **Mode**: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
- **Speechiness**:	Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
- **Acousticness**:	A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
- **Instrumentalness**: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. 
- **Liveness**:	Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
- **Valence**:	A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
- **Tempo**:The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
- **Duration**: The duration of the track in milliseconds.
- **Time Signature**: 	An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure)

It is important to note the scale of these features: Valence, Liveness, Instrumentalness, Acousticness, Speechiness, Energy, and Danceability are all on a scale from 0 to 1, where Time Signature, Tempo, Popularity and Loudness are not. It is also important to point out that although key is numerical in this case, the integers should be treated as categories, as they follow the notes of the scale rather than having any real numerical value. The Spotify API Documentation also provides typical distributions of these metrics, many of which are not distrubuted normally. 

# 2. Data Exploration and Analysis
I want to get a feel for this dataset. Since I will be looking into music taste over time, I looked into the amount of data I had for each year:

| Year | Count |
|------|-------|
| 2020 |   80  |
| 2019 |   318 |
| 2018 |   67  |
| 2017 |   105 |

I know I took a break from Spotify in 2018, so this data makes sense. I wanted to further look at seasonality in terms of months, however the population size for each month would be too small for any robust analysis. I decided to categorize months into (New York) seasons to explore if there was any weather related music taste differences. 
```python
seasons = []
for i in spotify_df.added_month:
    if i in (1,2,3,12):
        s = "Winter"
    if i in (4, 5, 6):
        s = "Spring"
    if i in (7,8,9):
        s = "Summer"
    if i in (10, 11):
        s = "Fall"
    seasons.append(s)
spotify_df['seasons'] = seasons
```

| Season | Count |
|--------|-------|
| Winter |   232 |
| Spring |   220 |
| Summer |   57  |
| Fall   |   61  |

Cool. Next, to get a general feel for the continuous variables, I used `Pandas Profiling` to look at quick distributions, correlation matrices, and zero instances. Overall, the songs in my library have low levels of acousticness, liveness, and speechiness, and high levels of danceability, enery, and loudness. Valence, key, mode, and popularity didn't show any real trend. There were warnings for high numbers of zeros in instrumentalness, key, and popularity. Since a 0 value for key is related to an actual note, it's okay to ignore that, but for instrumentalness and popularity I will be cautious moving forward

In terms of how these are correlated: 
-  **Acousticness** is negatively correlated with energy and loudness 
-  **Duration** is positively correlated with instrumentalness 
-  **Loudness** and **Energy** are very positively correlated 

I will explore this further when I look at these features more closely. 

## 2.1 Top Artists
I wanted to see who my top artists were, which came at a bit of a surprise! I did not realize I was such a Britney Spears and NYSYNC fan but I guess the data doesn't lie... 

|     | artist         |   uri |
|-----|----------------|-------|
|  33 | Bassnectar     |    22 |
| 301 | Steely Dan     |    21 |
| 285 | San Fermin     |    20 |
| 360 | deadmau5       |    13 |
| 121 | Frank Ocean    |     7 |
|  24 | Ariana Grande  |     7 |
|  50 | Britney Spears |     7 |
|  88 | Disclosure     |     6 |
|  89 | Djavan         |     6 |
|   0 | *NSYNC         |     5 |

## 2.2 Popularity
I thought the popularity metric was very interesting and decided to look further into it. The distribution of my library's popularity looked like this:

![Popularity](https://luicyfruit.github.io/img/popularity_hist.png)


Since there were a lot of zeros (missing data), I dropped them from the dataset temporarily to dive a little deeper into this metric. I wanted to see how the seasons affected my music taste, i.e. do I listen to different levels of popular music in different seasons? 

![Popularity_Boxplot](https://luicyfruit.github.io/img/popularity_box.png)

Since spring looked like it was different between the other seasons, I decided to run an ANOVA to see if this trend was significant. 
**Null Hypothesis**: Season has no affect on popularity of songs discovered
**Alternate Hypothesis:** Season affects popularity of songs discovered
```python
results = ols('popularity ~ C(seasons)', data=temp).fit()
results.summary()
```
Given the results of the ANOVA, (p value < F - Statistic, 1.7e-8 < 13.47), I can reject the null hypothesis. I used a tukey HSD test to see where this significance occurs more clearly, and I could reject the null at every pairing with Spring, and no other pairing (Winter-Fall, Winter-Summer, Fall-Summer). 

It is clear that I listen to significantly more popular music in the spring than in any other season! Something about the change in weather maybe

## 2.3 Audio Features
Let's look back at the audio features and continuous variables.

![AudioFeatures](https://luicyfruit.github.io/img/audio_features.png)

Let's break these down to how they compare with the Spotify API Documentation of typical distributions. 
- **Acousticness**: This typically sees a distribtion that's U shaped, with a high skew towards a 0 value that dips, and has a slight upturn at values approaching 1. My distribution is heavily skewed towards 0, signifying that I listen to **less** acoustic music than typically seen.
- **Danceability**: This typically has a normal distribution around a mean value of ~.6. My distribution is fairly normal, but slight skewed towards higher values. with a mean score of .67, signifying I listen to dancier music than normal.
- **Energy**: This typically has a distribution that's somewhat uniform, while being skewed closer to values of 1. My distribution lacks that uniform aspect and has more of a bell shape, but is still skewed closer to values of 1. Thus, I listen to fewer non-energetic songs than normal.
- **Instrumentalness, Liveness, Speechiness, Tempo,** and **Valence**: All look very similar to the distribtions provided by Spotify. 

Overall, it seems I listen to music that is less acoustic, more dance-y, and avoid non-energetic music than the distribution of music available on Spotify. 

Let's look into the variety of my music taste a little more. I want to compare the standard deviations of each feature to see if there are categories where I am more open-minded. Since all audio features above are on a scale from 0 - 1, we can look at them all simultaneously. Understanding instrumentalness had a high number of 0's to begin with, I removed them temporarily after looking at the standard deviation with and without them 

![FeaturesStdDev](https://luicyfruit.github.io/img/feature_stddev.png)

It seems I have a more narrow taste in music when it comes to speechiness, liveness, and danceability, than I do for other features such as instrumentalness, acousticness, or valence. It's interesting to note that Danceability, Acousticness, and Energy were features that distinguished my taste from the distribution of music on Spotify, however of those three, danceability had a low standard deviation. Thus, the danceability of a song seems to play a very important role in how I choose music. 

Adding back in features like duration, tempo, and popularity, next i checked how they are correlated:

![Heatmap](https://luicyfruit.github.io/img/features_heatmap.png)

Loudness and energy are highly correlated (keeping in mind for any model building in the future), which makes sense. Those were the only two variables with a correlation above .7. Acousticness and Energy were also negatively correlated (-0.63).

## 2.4 Key
The way Spotify API classifies keys is using Pitch Class notation, where C = 0, C# = 1, D = 2, all the way up to B = 11. The audio features also include mode, which indicates if the key is major or minor (1 and 0 respectively) - thus resulting in 22 possible keys. For readability, I added a categorical variable of major/minor. 

| Mode   | Count |
|--------|-------|
| Major  |   356 |
| Minor  |   214 |

It seems I listen to Major songs more than Minor. Knowing how I feel about New York Winters, and that major is associated with happy songs and minor sad, how does that look with seasonality?


![Mode](https://luicyfruit.github.io/img/mode_season.png)

In the spring it seems like I am listening to majority major songs (33% difference), where the remaining seasons get closer to an equal split (summer 23% difference, winter 21% difference, and fall 15% difference). 

Since valence has to do with the eomition evoked from a song, and major music tends to be happy while minor music tends to be sad, I wanted to see if there was any difference between my major and minor songs.

![Mode_valence](https://luicyfruit.github.io/img/mode_valence.png)

After looking at the boxplot (and a quick T-test), it was apparent that mode had nothing to do with valence of a song at least in the small biased sample of my music!



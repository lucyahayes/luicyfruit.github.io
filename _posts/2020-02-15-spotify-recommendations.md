---
layout: post
date: 2020-02-15
title: "How Does Spotify Determine your Discover Weekly?"
header-img: "img/music.png"
---

Tasking myself with trying to determine my music taste and potentially build my own Discover Weekly was no small task, after reading up on how Spotify determines your Discover weekly playlist. Turns out, they use a combination of three recommendation models:
- *Collaborative Filtering*
- *Natural Language Processing*
- *Audio* models

## Collaborative Filtering
Collaborative Filtering is **insert definition and picture**

The most commonly use form of collaborative filtering is thinking about how Netflix uses your watching history and the watching history of others to inform their recommendations. Spotify uses a similar mechanism, where it takes into account the stream counts of the music we listen to, as well as checking if the user saved the track to their playlist, or visited the artist page. Then, they look at users who have done the same thing as you, and the other music they like. Aka, if me and another user both liked the same bands and listened to some of the same tracks, I would be likely to also like the additional music that user listened to and vice-versa. Combining the entire userface results in seeing which songs are most similar to other songs, and using those to reccomend it to users.

## Natural Language Processing

NLP is the ability of a computer to understand human speech. ** insert definition and picture**

Spotify implements NLP by deploying web crawlers that look for articles, blogs, or any medium where there is written text about music. They then determine what is being said about the music, using frequent adjectives and type of language that is frequently used when talking about certain songs, and then which other songs or artists are also discussed within that context. Thus, each artist and song has "Top terms" associated with them, with the term and its corresponding weight (probability that someone will describe the song/artist as that term). Then, by looking at these vectors, NLP can determine which artist or song is most similar to another artist or song, and recommends based off that information

## Audio Models - Convolutional Neural Networks

** insert definition and picture**

I had the chance to explore some of the end result of the audio models when looking at the audio features in my user library. Spotify uses convolutional neural networks to analyze raw audio data over the length of time of the song, and can compute statistics of the features across this time. The output is a score for each feature, for things like tempo, danceability, acousticness, and energy levels. This model takes into account new songs, which by comparing those audio features with songs that have similar features, is why a new song can end up recommended to you (even if it only has very few listens). This helps new artist's music get discovered! 

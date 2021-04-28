# Using a Classifier to Predict the features of a Top Song w/Python
### CSCI 347 Final Project
### Kade Pitsch & Wesley Smith

# What Does the Data look like?
Initially the data contained about 170,000 rows and 19 columns. The data starts in 1921 and goes to 2020, just under 100 years of song data which is way too much for us. We reduced our training data-set to years 2018 and 2019 which still contains 5043 rows. Our set to verify the results is just the year 2020 with 4294 rows.

Some of the columns we decided to get rid of like the associated spotify ID.
That is not important to us or the overall song popularity. Then we split the `release_date` column into `month_release`, `day_release` and `weekday_release`.
The `release_year` is also not important for our training. 
Then we plotted some simple data just to get a look at things.

The plot below is not super clear but the X-axis is days of the week 0 being Monday, 1 being Tuesday, 2 is Wednesday...etc. Friday is the most popular day of release by a long shot! There is a reason for this, the chart week starts on Friday so for artists to get the maximum time for a chart week they release on Friday at 12am. This is also data taken on our untrimmed data so it is a lot more exaggerated.

![Plt 1](Images/plt1_thursday_releases.png)

December and January are the most popular months to release. This is probably to get the highest potential at #1 album of the year. More time on the charts the better your chances i suppose. 

![Plt 2](Images/plt2_release_month.png)

I don't think this graph means too much just kinda fun to look at the distribution of release days during the months. It does look like the first of the months and middle of the month have the highest values but there is no reason to release music on a certain day of the month, only important that they drop music first thing Friday.

![Plt 3](Images/plt3_release_day_of_month.png)

Unfortunately we think we are going to have to drop the artists column, due to how the data is structured from Spotify's API it looks like the artist
column is also includes featured artists and there is a lot of noise going
on here. We were able to just trim this column to get the first element included in the
list but we could not find too much information on how that list was structured so it
could actually be harming our accuracy because it could be attributing it to the wrong
artist. There is so much happening in this column specifically that we
decided it is not worth it. The top 'artist' is a workout playlist that does remixes of popular songs so this pollutes our data with these kinda pseudo-duplicates because they
are unique data points but they
are also super similar to their predecessor song. We will just remove these values from
the data set. We are glad we explored this portion of the data a little more and removed it because it cleaned a lot of things up.

![Plt 4](Images/plt4_popularArtists.png)


We started playing around with plotting certain features to see if they were highly
correlated. We got this image of the valence and the energy of the song which visually
showed some high correlation. After this we decided to actually calculate the
correlation of all the features to see if there is some data we should remove.

![Plt 5](Images/plt5_energyandValence.png)

Digging into the data more we used some pandas magic to find the covariance matrix then
we had to unstack the Series and sort it, getting the highest values of correlation
dropping the `NaN's` from the data in the process.

![Code 1](Images/correlation_code.png)

After running this code we got the output of

![Output 1](Images/Highly_correlated.png)

Looking at this we can see that Energy and loudness are the most highly correlated 
features, they do get output twice for x -> y and y -> x but we can just ignore every other row in the output as they are the same measurement. Then we get energy and acousticness, loudness and acousticness and popularity and explicit.
We have to drop the popularity for the training data so explicit feature can stay. 
Energy and acousticness appear twice in our highly correlated output we will drop both of those features for our final testing. 





# Methods of Classification
To classify our data we are going to use build a our own K-Nearest Neighbors. Compare
how ours does classifying to a built in method and also test other built-ins like
Naive-Bayes. Compare and contrast them all and see what method performs the best
provided with our data.
The main goal is to see what gives us the most accurate result. Cross-referencing this
with the provided `sklearn` versions to see how good our hand-written methods do.
The thing that we want to test is what features attribute to the most popular song.
Maybe the loudness of the song is the most important maybe it is the danceability or
something else. The point of our classification is to see which of
these attributes is the most important to the popularity.
Here are the features of the data-set, artists and names were dropped for K-Nearest Neighbors
``` python
['acousticness', 'artists', 'danceability', 'duration_ms', 'energy','explicit',
        'instrumentalness', 'liveness', 'loudness', 'mode','name', 'popularity',
        'release_date', 'speechiness', 'tempo', 'valence', 'year']
```

# K-Nearest Neighbors
We chose this method after logistic regression just ended up being too much to bite off. After several attempts at it we gave it up to focus more on the project rather than a special algorithm. 
So we built our own version of K-Nearest neighbors that performs pretty well...sometimes.

For the logistic regression to work the best it can we need to change up our data set. Unfortunately we will have to one-hot encode our categorical attributes which is going to make our data-set huge. We may have to trim down our data set to ~500 rows to train on the Logistic Regression model.



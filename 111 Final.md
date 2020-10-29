# Natural Language Generation Using Markov Chains and Scikit
## Introduction

After some time thinking of an idea for a final project, I came up with an idea to create a bot that someone could talk to that would reply as if it were a character from a movie or television show. For my own enjoyment, I chose Michael Scott from The Office. My idea was to take the input a speaker spoke to the bot (That's right! With a little twist of speech recognition.), find the lines which were most similar in the show, from any character, and generate a response based on Michael's response to these lines. This more or less worked. For more details on the process and my successes and fails, see below.


I did have some inspiration from students at Stanford University who created similar bot but using Neural Machine Translation, their project is linked [here](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15709728.pdf)!

## Data
Professor Barnwell helped me locate my [data](https://docs.google.com/spreadsheets/d/18wS5AAwOh8QO95RwHLS95POmSNKA2jjzdt0phrxeAE0/edit#gid=747974534), which was a document on Google Sheets which contained every line in the scripts in The Office attached to the name of the character it belonged to as well as season and episode. Here is my data set in the form of a Pandas dataframe:


```python
import pandas as pd
data = pd.read_csv("the-office-lines - scripts.csv")
data = data.loc[data['deleted'] == False]
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>season</th>
      <th>episode</th>
      <th>scene</th>
      <th>line_text</th>
      <th>speaker</th>
      <th>deleted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>All right Jim. Your quarterlies look very good...</td>
      <td>Michael</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Oh, I told you. I couldn't close it. So...</td>
      <td>Jim</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>So you've come to the master for guidance? Is ...</td>
      <td>Michael</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Actually, you called me in here, but yeah.</td>
      <td>Jim</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>All right. Well, let me show you how it's done.</td>
      <td>Michael</td>
      <td>False</td>
    </tr>
    <tr>
      <td>5</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>[on the phone] Yes, I'd like to speak to your ...</td>
      <td>Michael</td>
      <td>False</td>
    </tr>
    <tr>
      <td>6</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>I've, uh, I've been at Dunder Mifflin for 12 y...</td>
      <td>Michael</td>
      <td>False</td>
    </tr>
    <tr>
      <td>7</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>Well. I don't know.</td>
      <td>Pam</td>
      <td>False</td>
    </tr>
    <tr>
      <td>8</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>If you think she's cute now, you should have s...</td>
      <td>Michael</td>
      <td>False</td>
    </tr>
    <tr>
      <td>9</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>What?</td>
      <td>Pam</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



After looking at the line_text portion of the data frame, I decided I needed to clean up the text a bit more so my Markov generator to work better in the future. To do this I created new column in which all of the text in line_text was converted to lower case and all brackets and the text inside of the brackets were removed.


```python
def cleantext(text):
    words = text.split()
    words = [word.lower() for word in words]
    words = ' '.join([str(elem) for elem in words])
    return words

def cleanertext(text):
    ret = ''
    skip1c = 0
    skip2c = 0
    for i in text:
        if i == '[':
            skip1c += 1
        elif i == '(':
            skip2c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        elif i == ')'and skip2c > 0:
            skip2c -= 1
        elif skip1c == 0 and skip2c == 0:
            ret += i
    return ret

data['clean'] = data.line_text.apply(cleantext).apply(cleanertext)
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>season</th>
      <th>episode</th>
      <th>scene</th>
      <th>line_text</th>
      <th>speaker</th>
      <th>deleted</th>
      <th>clean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>All right Jim. Your quarterlies look very good...</td>
      <td>Michael</td>
      <td>False</td>
      <td>all right jim. your quarterlies look very good...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Oh, I told you. I couldn't close it. So...</td>
      <td>Jim</td>
      <td>False</td>
      <td>oh, i told you. i couldn't close it. so...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>So you've come to the master for guidance? Is ...</td>
      <td>Michael</td>
      <td>False</td>
      <td>so you've come to the master for guidance? is ...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Actually, you called me in here, but yeah.</td>
      <td>Jim</td>
      <td>False</td>
      <td>actually, you called me in here, but yeah.</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>All right. Well, let me show you how it's done.</td>
      <td>Michael</td>
      <td>False</td>
      <td>all right. well, let me show you how it's done.</td>
    </tr>
    <tr>
      <td>5</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>[on the phone] Yes, I'd like to speak to your ...</td>
      <td>Michael</td>
      <td>False</td>
      <td>yes, i'd like to speak to your office manager...</td>
    </tr>
    <tr>
      <td>6</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>I've, uh, I've been at Dunder Mifflin for 12 y...</td>
      <td>Michael</td>
      <td>False</td>
      <td>i've, uh, i've been at dunder mifflin for 12 y...</td>
    </tr>
    <tr>
      <td>7</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>Well. I don't know.</td>
      <td>Pam</td>
      <td>False</td>
      <td>well. i don't know.</td>
    </tr>
    <tr>
      <td>8</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>If you think she's cute now, you should have s...</td>
      <td>Michael</td>
      <td>False</td>
      <td>if you think she's cute now, you should have s...</td>
    </tr>
    <tr>
      <td>9</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>What?</td>
      <td>Pam</td>
      <td>False</td>
      <td>what?</td>
    </tr>
  </tbody>
</table>
</div>



## Methods
So with my original idea, having the text generate off Michael's responses was key to having the bot make logical conversation with the user. Unfortunately, I had a bit of trouble writing this function... It worked, but not as well or as consistently as I had hoped. I believe the problem has to do with the index of my data. I will show you my method here:


```python
import sklearn
import sklearn.feature_extraction.text
import sklearn.decomposition
import sklearn.neighbors

vect = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.8, max_features=2000, sublinear_tf=True)
vect.fit(data.clean)
features = vect.transform(data.clean)
svd = sklearn.decomposition.TruncatedSVD(n_components=2)
svd.fit(features)
nn = sklearn.neighbors.NearestNeighbors(metric='cosine')
nn.fit(features)
```




    NearestNeighbors(algorithm='auto', leaf_size=30, metric='cosine',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     radius=1.0)




```python
def findlinesx(query):
    query_features = vect.transform([query])
    dists, ixs = nn.kneighbors(query_features, n_neighbors=25)
    matches = data.iloc[ixs[0, 1:]].copy().id
    for match in matches:
        list = [matches.index]
    for match in list:
        replies = data.iloc[match + 1]
    for match in replies:
        return replies.loc[replies.speaker == "Dwight"]
                   
findlinesx("what day is today")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>season</th>
      <th>episode</th>
      <th>scene</th>
      <th>line_text</th>
      <th>speaker</th>
      <th>deleted</th>
      <th>clean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>17502</td>
      <td>17503</td>
      <td>4</td>
      <td>1</td>
      <td>85</td>
      <td>Question: Has anyone recently offended a Gypsy?</td>
      <td>Dwight</td>
      <td>False</td>
      <td>question: has anyone recently offended a gypsy?</td>
    </tr>
    <tr>
      <td>8633</td>
      <td>8634</td>
      <td>2</td>
      <td>20</td>
      <td>30</td>
      <td>Then I refuse.</td>
      <td>Dwight</td>
      <td>False</td>
      <td>then i refuse.</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, it did work... But not with the consistency or results that I wanted. So I decided to scratch that part and go back to just finding the most similar lines, which worked much better! For this method I needed to make a data frame which included just Michael's lines, then create a vectorizer and fit it to that data frame.


```python
michael = data.loc[data.speaker=="Michael"]
```


```python
import sklearn
import sklearn.feature_extraction.text

vect = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.8, max_features=2000, sublinear_tf=True)
vect.fit(michael.clean)
features = vect.transform(michael.clean)
svd = sklearn.decomposition.TruncatedSVD(n_components=2)
svd.fit(features)
nn = sklearn.neighbors.NearestNeighbors(metric='cosine')
nn.fit(features)
```




    NearestNeighbors(algorithm='auto', leaf_size=30, metric='cosine',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     radius=1.0)



Next I created a couple different functions. The first was a function to find ngrams I will use in my Markov generator later. The second was a function to find thirty lines similar to a given input. And the third was to choose a random start word to generate the Markov Chain with. The third function improved my project tremendously in terms of generating a natural output. Without it you would have to choose a word you would like Michael to reply with, and with that the response is less organic and the options for the text to follow are extremely limited. 


```python
import random
import nltk
tokenizer = nltk.tokenize.TweetTokenizer()

def find_ngrams(text, sizeOfNgram):
    bigrams = []
    for ngram in nltk.ngrams(tokenizer.tokenize(text), sizeOfNgram):
        tempDict = {
            "Size":    sizeOfNgram - 1,
            "Gram": ngram[:-1],
            "Word":  ngram[-1],
        }
        bigrams.append(tempDict)
    return bigrams
```


```python
def findthelines(query):
    in_list = [1, 2, 3, 4]
    new_list = []
    query_features = vect.transform([query])
    dists, ixs = nn.kneighbors(query_features, n_neighbors=30)
    matches = michael.iloc[ixs[0, 1:]].copy()
    for x in in_list:
        new_list = new_list + matches["clean"].apply(find_ngrams, sizeOfNgram=x).sum()
    my_data = pd.DataFrame.from_records(new_list)
    return my_data
```


```python
def startwords(text):
    query_features = vect.transform([text])
    dists, ixs = nn.kneighbors(query_features, n_neighbors=30)
    matches = michael.iloc[ixs[0, 1:]].copy()
    mtext = matches["clean"].tolist()
    mstr = ' '.join([str(elem) for elem in mtext])
    mylist = mstr.split()  
    bigrams = zip(mylist, mylist[1:])
    return(random.choice([b[1] for b in bigrams if b[0].endswith('.')]))
```

Next I created a function which would take an input, find the most common lines, find a random startword from those lines, and then generate a Markov Chain accordingly! It works! Well, most of the time... Depending on the input it may or may not get a bit stuck on itself...


```python
def newfunct(query):
    mylist = [startwords(query)]
    mydata = findthelines(query)
    grouped_grams = mydata.groupby(['Size', 'Gram', 'Word']).size()
    while mylist[-1] != ".":
        if grouped_grams.loc[1, (mylist[-1], )].sample(1, replace=True, weights=grouped_grams.loc[1, (mylist[-1], )]).index.size >= 3:
            mylist.append(grouped_grams.loc[1, (mylist[-1], )].sample(1, weights=grouped_grams.loc[1, (mylist[-1], )]).index[0])
        if grouped_grams.loc[1, (mylist[-1], )].sample(1, replace=True, weights=grouped_grams.loc[1, (mylist[-1], )]).index.size <= 3:
            mylist.append(grouped_grams.loc[1, (mylist[-1], )].sample(1, weights=grouped_grams.loc[1, (mylist[-1], )]).index[0])
    print(" ".join(mylist))
    
newfunct("do you have any paper")
```

    it's ju - they didn't have any more of those clips that is where the magic happens ! right over here , do ? do you know what that hold paper .


After that, I wrote some more code so my Markov generator was a bit more interactive.


```python
continue_dialogue = True
print("Hello, my name is Michael Scott:")
while(continue_dialogue == True):
    human_text = input()
    human_text = human_text.lower()
    print("Michael: ", end="")
    print(newfunct(human_text))
```

    Hello, my name is Michael Scott:



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    /opt/anaconda3/lib/python3.7/site-packages/ipykernel/kernelbase.py in _input_request(self, prompt, ident, parent, password)
        884             try:
    --> 885                 ident, reply = self.session.recv(self.stdin_socket, 0)
        886             except Exception:


    /opt/anaconda3/lib/python3.7/site-packages/jupyter_client/session.py in recv(self, socket, mode, content, copy)
        802         try:
    --> 803             msg_list = socket.recv_multipart(mode, copy=copy)
        804         except zmq.ZMQError as e:


    /opt/anaconda3/lib/python3.7/site-packages/zmq/sugar/socket.py in recv_multipart(self, flags, copy, track)
        474         """
    --> 475         parts = [self.recv(flags, copy=copy, track=track)]
        476         # have first part already, only loop while more to receive


    zmq/backend/cython/socket.pyx in zmq.backend.cython.socket.Socket.recv()


    zmq/backend/cython/socket.pyx in zmq.backend.cython.socket.Socket.recv()


    zmq/backend/cython/socket.pyx in zmq.backend.cython.socket._recv_copy()


    /opt/anaconda3/lib/python3.7/site-packages/zmq/backend/cython/checkrc.pxd in zmq.backend.cython.checkrc._check_rc()


    KeyboardInterrupt: 

    
    During handling of the above exception, another exception occurred:


    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-12-fad4591efde4> in <module>
          2 print("Hello, my name is Michael Scott:")
          3 while(continue_dialogue == True):
    ----> 4     human_text = input()
          5     human_text = human_text.lower()
          6     print("Michael: ", end="")


    /opt/anaconda3/lib/python3.7/site-packages/ipykernel/kernelbase.py in raw_input(self, prompt)
        858             self._parent_ident,
        859             self._parent_header,
    --> 860             password=False,
        861         )
        862 


    /opt/anaconda3/lib/python3.7/site-packages/ipykernel/kernelbase.py in _input_request(self, prompt, ident, parent, password)
        888             except KeyboardInterrupt:
        889                 # re-raise KeyboardInterrupt, to truncate traceback
    --> 890                 raise KeyboardInterrupt
        891             else:
        892                 break


    KeyboardInterrupt: 


Like my Markov generator, it works, but not all of the time. It tends to like certain inputs better than others. I wasn't too happy with it so I decided I wanted to try to improve it. I wrote some more functions and some more code which improved the bots ability to have a conversation a lot! Take a look below! 

(I often need to interrupt the kernel after running the line ahead of this one in order for the next functions to run, just a tip!)


```python
import string
import urllib.request
```


```python
mtext = michael["clean"].tolist()
mstr = ' '.join([str(elem) for elem in mtext])
msents = nltk.sent_tokenize(mstr)
mwords = nltk.word_tokenize(mstr)
```


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```


```python
wnlemmatizer = nltk.stem.WordNetLemmatizer()

def perform_lemmatization(tokens):
    return [wnlemmatizer.lemmatize(token) for token in tokens]

punctuation_removal = dict((ord(punctuation), None) for punctuation in string.punctuation)

def get_processed_text(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(punctuation_removal)))
```


```python
def generate_response(user_input):
    mresponse = ''
    msents.append(user_input)

    word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
    all_word_vectors = word_vectorizer.fit_transform(msents)
    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched == 0:
        mrobo = mresponse + "I am sorry, I could not understand you"
        return mresponse
    else:
        mresponse = mresponse + msents[similar_sentence_number]
        return mresponse
```


```python
greeting_inputs = ("hey", "good morning", "good evening", "morning", "evening", "hi", "whatsup")
greeting_responses = ["hey", "hey hows you?", "*nods*", "hello, how you doing", "hello", "Welcome, I am good and you"]

def generate_greeting_response(greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)
```


```python
continue_dialogue = True
print("Hello, I am your friend Michael. How are you today:")
while(continue_dialogue == True):
    human_text = input()
    human_text = human_text.lower()
    if human_text != 'bye':
        if human_text == 'thanks' or human_text == 'thank you very much' or human_text == 'thank you':
            continue_dialogue = False
            print("Michael: Most welcome")
        else:
            if generate_greeting_response(human_text) != None:
                print("Michael: " + generate_greeting_response(human_text))
            else:
                print("Michael: ", end="")
                print(generate_response(human_text))
                msents.remove(human_text)
    else:
        continue_dialogue = False
        print("Michael: Good bye and take care of yourself...")
```

    Hello, I am your friend Michael. How are you today:


     good


    Michael: 

    /opt/anaconda3/lib/python3.7/site-packages/sklearn/feature_extraction/text.py:300: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['ha', 'le', 'u', 'wa'] not in stop_words.
      'stop_words.' % sorted(inconsistent))


    good.


     what did you do today


    Michael: 

    /opt/anaconda3/lib/python3.7/site-packages/sklearn/feature_extraction/text.py:300: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['ha', 'le', 'u', 'wa'] not in stop_words.
      'stop_words.' % sorted(inconsistent))


    oh, what did you do today?


The dialogue doesn't make the most sense, but it is a start and it works! I honestly think that if I had gotten the bit at the beginning to work, where the function responded to the input with an output that was correalated with the response, this bot would have worked much better, making conversation that actually flowed. This, like you can see, more or less repeats the question or statement you say to it. But, despite that bit, I am pretty happy with it for now!

My next object to tackle was speech recognition. This proved to be the easiest portion of my project, and it makes it pretty fun to use! I chose to use Houndify as my oustide platform. 

(Once again, you often have to interrupt the kernel after the last bit of code for this to run.)


```python
import speech_recognition as sr
r = sr.Recognizer()
```


```python
houndify_client_id = "nUNnnQI4iNwkx37FwrzRRw=="
houndify_client_key = "pSqDYq1cKrL5IIvv6UmXtfE3d5bLWdUggz_qDMPxo2btetPORPzcI5nYGZm7Vy0oAIbH6q_dbThA6ShcpWDYEA=="
```


```python
mic = sr.Microphone(0)
```

Try speaking here!


```python
with mic as source:
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source, phrase_time_limit=8)

r.recognize_google(audio)
```

Now that the speech recognition is all set up, we can attach it to a function. I chose to attach it to newfunct. Go ahead and try it!


```python
def newfunct_audio(audio):
    text = r.recognize_houndify(audio, houndify_client_id, houndify_client_key)
    print("You said:", text)
    return newfunct(text)
```


```python
with mic as source:
    audio = r.listen(source, phrase_time_limit=5)

newfunct_audio(audio)
```

    You said: how are you doing today
    what are you doing , how we doing ? need anything ? need anything ? you doing ? need anything ? what are you doing , ernie , what are you ? you ? what are you doing ? need anything ? you doing ? you doing ? you doing ? you doing ? need anything ? you already there ? you doing , jan .


## Conclusion
Overall, I am pretty happy with how my project turned out. It definitely has room for improvement, and you might catch me during this quarantine trying to redo the portion with the replies so it doesn't haunt me forever. In the future I could explore using spacy in this more to make more coherent text, but this kind of works without it. It's inline with how Michael speaks anyways. 

I would really like to explore this again but with neural machine translation, and then compare that to Markov chains. Maybe I could make something more applicable to the real world. But, this was fun for now.


```python

```

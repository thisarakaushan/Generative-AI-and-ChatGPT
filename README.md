# Generative AI and ChatGPT Guide

## AGENDA
1. Introduction to ML/DL/AI, Generative Al and Tools used.
2. Python for Data Science
3. NLP
4. Introduction to Neural Network
5. Advanced Al technique
6. ChatGPT

**WHAT IS DATA SCIENCE?**
* To extract the hidden information/insights from the data through statistical analysis, comp tatjon, visualization; that help the
  organizations to take right business decisions.

**APPLICATION OF DATA SCIENCE**

<div style="text-align: center;">
<img align="center" width="400" alt="image" src="https://github.com/thisarakaushan/Generative-AI-and-ChatGPT/assets/125348115/6192cd44-2b41-4939-a359-f19b4093920d">
</div>

<br>

```
O  |  Obtain the data  |    Data collection
S  |  Scrub the data   |    Clean the data
E  |  Explore the data |    Visualization
M  |  Modeling         |    ML/AI model
N  |  iNterpretation   |    Infer the results
```

<br>

<center>
<img width="400" alt="image" src="https://github.com/thisarakaushan/Generative-AI-and-ChatGPT/assets/125348115/85ac3479-3857-4798-92e8-ee686ad5b180">
</center>

<br><br>

**Types Of AI**
1. Narrow AI
2. Strong AI
3. Super AI

### Narrow AI (Weak AI):

* Narrow Al, also known as Weak Al or Artificial Narrow Intelligence (ANI), refers to Al systems that are highly     
  specialized and designed to excel in a specific, well-defined task or a limited set of tasks.
* These systems lack general intelligence and cannot perform tasks outside their narrow domain.
* Examples:
    - _Speech Recognition_: Voice-activated virtual assistants like Apple's Siri and Amazon's Alexa are examples of Narrow           Al.They can understand and respond to spoken language but are limited to specific tasks like setting reminders or          answering questions.
    - _Image Recognition_: Applications that can identify objects or patterns in images, such as Google Photos' image       
      recognition feature, are instances ofNarrow Al. They are designed for a specific image analysis task.
      
      1. Chatbots:
          1. _Examples_: Customer support chatbots, Facebook Messenger bots
          2. _Capabilities_: Chatbots use NLP to engage in conversations with users, answering queries, providing         
             information, and assisting with tasks. They are commonly employed in customer service applications to handle                routine inquiries.
             
      2. Email Filtering and Spam Detection:
          1. _Examples_: Gmail's spam filter, Microsoft Outlook's Clutter feature
          2. _Capabilities_: Narrow Al algorithms analyze incoming emails, learning from user interactions to identify and               filter out spam or unwanted messages, improving the efficiency of email management.
             
      3. Language Translation:
          1. _Examples_: Google Translate, Microsoft Translator
          2. _Capabilities_: Narrow Al systems for language translation use machine learning techniques to analyze and                   translate text or speech from one language to another,facilitating communication across language barriers.

* Key Characteristics of Narrow Al:
  1. _Specialization_: Narrow Al is designed for specialized tasks, excelling within its I dedicated domain, and often surpassing human performance in that area.
  2. _Lack of Generalization_: Unlike AGI, Narrow Al is unable to transfer its knowledge or skills to unrelated tasks, remaining confined to its designated area of expertise.
  3. _Data-Driven Nature_: The performance of Narrow Al relies heavily on the quality and quantity of data it receives, making data preprocessing and curation critical for its efficacy.
  4. _Real-time Processing_: These Al systems excel at providing real-time responses, thanks to their specialized nature. This attribute is invaluable in applications requiring quick decision-making, such as autonomous vehicles—ant—medical diagnostics.

* some real-time examples of Narrow Al applications:
  1. Virtual Personal Assistants:
      1. _Examples_: Siri (Apple), Alexa (Amazon), Google Assistant
      2. _Capabilities_: These virtual assistants use natural language processing (NLP) to understand and respond to user            queries. They can perform tasks like setting reminders, sending messages, and providing information based on                predefined commands.

  2. Image and Speech Recognition:
      1. _Examples_: Google Photos, Microsoft Face ID, Speech-to-Text applications
      2. _Capabilities_: Narrow Al systems can analyze images to recognize faces, objects, and scenes. Speech recognition            applications convert spoken language into text, enabling hands-free operation and transcription services.
         
  3. Recommendation Systems:
      1. _Examples_: Netflix recommendations, Amazon product recommendations, Spotify playlists
      2. _Capabilities_: Narrow Al algorithms analyze user behavior.

### General AI (Strong AI):

* Strong Al, Also known as Artificial General Intelligence (AGI), Strong Al represents the pinnacle of Al development, with the potential to exhibit human-like cognitive abilities across a broad spectrum of domains.
* Unlike Narrow Al (ANI), which is specialized and task-focused, Strong Al aims to replicate human-level cognitive abilities.
* Characteristics: Strong Al exhibits the following key attributes:
    1. Versatility: The ability to perform a variety of tasks and adapt to new challenges.
    2. Learning and Adaptation: The capacity to learn from experience and transfer knowledge to different domains.
    3. Understanding Context: Understanding and interpreting context in a human-like manner.
    4. Consciousness: A level of self-awareness and consciousness akin to human cognition.

* Examples:
    1. IBM Watson:
        1. _Domain_: Healthcare, Finance, Natural Language Processing (NLP)
        2. _Capabilities_: Watson is a cognitive computing system that has demonstrated advanced natural language                      processing, machine learning, and data analysis capabilities. In healthcare, it has been used for medical                   research, diagnosis, and treatment recommendations.
           
    2. OpenAl's GPT-3 (Generative Pre-trained Transformer 3):
        1. _Domain_: Natural Language Processing (NLP)
        2. _Capabilities_: GPT-3 is a state-of-the-art language model that demonstrates remarkable language understanding             and generation abilities. It can generate human-like text, answer questions, and perform language-related tasks.            However, it lacks true generalization across diverse domains.
  
    3. Google's DeepMind AlphaGo:
        1. _Domain_: Board Games (Go)
        2. _Capabilities_: AlphaGo is an Al system that achieved superhuman performance in the ancient Chinese board game             Go. It uses a combination of deep neural networks and reinforcement learning to analyze and play the game at an             exceptionally high level.

    4. Boston Dynamics' Atlas Robot:
        1. _Domain_: Robotics, Computer Vision
        2. _Capabilities_: While not an AGI, the Atlas robot from Boston Dynamics showcases advanced robotics capabilities.           It can navigate complex environments, perform dynamic movements, and execute tasks with a degree of autonomy.
           
    5. Tesla Autopilot:
        1. _Domain_: Autonomous Vehicles
        2. _Capabilities_: Tesla's Autopilot system employs advanced Al algorithms for semi-autonomous driving. While it               falls short of AGI, it demonstrates machine learning capabilities for real-time navigation, object detection,               and decision-making.
      
### Super Al (Superintelligent AI):

* SuperintelligentAl refers to an advanced form of artificial intelligence that surpasses human intelligence across all domains and exhibits capabilities beyond the comprehension of human minds.
* Unlike Artificial General Intelligence (AGI), which aims to mimic human-level intelligence, Superintelligent Al signifies an intelligence explosion, where machines attain levels of intelligence surpassing the most
brilliant human minds.
* As of now, true SuperintelligentAl (Artificial Superintelligence or ASI) does not exist, and the development of Al systems that surpass human intelligence in all domains remains a theoretical concept.

* Characteristics of Superintelligent Al:

  1. Incomprehensible Cognitive Abilities:
      * Superintelligent Al would possess cognitive capacities that go beyond human understanding, enabling it to solve             complex problems, devise innovative solutions, and potentially outperform human experts in every field.
  2. Rapid Learning and Adaptation:
      * The ability to learn at an unprecedented pace and adapt to new information and challenges instantaneously,                  facilitating continuous self-improvement and evolution.
  3. Global Optimization:
      * Superintelligent Al could optimize global systems and processes, leading to advancements in science, technology,             and societal structures.

### Generative Models vs Discriminative Models:

```
Discriminative Models:

• Discriminative Models are a family of models that do not generate new data points but
  learn to model boundaries around classes in a dataset instead.
• Discriminative model makes predictions on unseen data based on conditional probability
  and can be used either for classification or regression problem statements.
  - Objective: Learn decision boundary for accurate classification 
  - Training Approach: Supervised Learning
  - Type of Learning: Discriminative Modeling
  - Data Generation: No inherent data generation capabilities
  - Decision Boundary: Learn explicit decision boundary between different classes
  

Generative Models:

• Generative Models are a fa ily of models that create new data points. They are generally
  used for unsupervised tasks.
• Generative Models use the joint probability distribution on a variable X and a target
  variable Y to model the data and perform inference by estimating the probability of
  the new data point belonging to any given class.
  - Objective: Model data distribution to generate new samples
  - Training Approach: Unsupervised Learning
  - Type of Learning: Probabilistic Modeling 
  - Data Generation: Can generate new samples resembling training data
  - Decision Boundary: Capture complex decision boundary indirectly

```

### GENERATIVE Al MODELS:

1. **Generative adversarial network:**
  * Generative Adversarial Networks (GANs) are a powerful cla of neural networks that are used for unsupervised learning. The goal of GANs is to generate new, synthetic data that resembles some known data distribution.
    
2. **(Large) language model:**
  * A (large) language model (LLM) refers to neural networks form eling and generating text data that typically combine three characteristics i.e. transformer with an attention mechanism, next-word prediction use of large-scale datasets of     text.
  
3. **Prompt learning:** Prompt learning is a method for LLMs.
   
4. **Diffusion probabilistic models:**
  * Diffusion probability models are a class of latent variable models tha are common for various tasks such as image generation.
   
5. **Reinforcement learning from humanfeedback:**
  * RLHF is used in conversational systems such as ChatGPT (OpenAl, 2022) for generating chat messages, such that new answers accommodate the previous chat dialogue and ensure that the answers are in alignment with predefined human preferences.
   
6. **seq2seq:**
  * sequence-to-sequence (seq2seq) refers to machine learning approaches where an input sequence is mapped onto an output quence.

**Examples:**

1. Transformer:
  * A transformer is a deep learning architecture that adopts t mechanism of self-attention which differentially weights the importance of each part of the input data.

2. Variational autoencoder:
  * A variational autoencoder (VAE) is a type of neural network th is trained to learn a low-dimensional representation of the input data.

3. ChatGPT:
  * OpenAl has developed a language model that is close to a conversation with humans. It learns from interactions and processes information based on its learnings.

4. DALL- E:
  * DALL-E and DALL-E 2 are deep learning models developed by OpenA to generate digital images from natural language descriptions, called "prompts".

5. LaMDA:
  * LaMDA Al is a conversational Large discourse Model (LLM) developed by Google to power dialogue-based systems that       
    generate natural-sounding human discourse.


## Natural Language Processing (Text Mining).............................................................................................

**What is TEXT MINING (TM)?**
  * The use of computational methods and techniques to extract high quality information from text.
  * A computational approach to the discovery of new, previously unknown information and/or knowledge through automated extraction of information from often large amounts of unstructured text.
  * ```Corpus``` is the collection of documents, and document is a sentence consisting with words.

#### Use cases of NLP:

  1. Document Classification
  2. Clustering / organizing documents
  3. Documents summerization
  4. Visualization od document space: often aimed to facilitating documnet search
  5. Making prediction: stock market prices prediction on the analysis of news araticles and financial reports
  6. Content-based recommender systems: for news articles, movies, books...

#### Application of NLP:

1. Virtual assastants (Alexa, Sau, Cortana, ...)
2. Customer support tools (chatbots, email routers/ classifiers, ...)
3. _Sentiment analysers_: based on the collection of customer feedbacks and reviews, we defied the sentiment and emotion of the customer whether the customer is satisfied with the product or not.
4. _Machine Translators_: translate the language one language to another
5. Document similarity systems.

#### Text Pre-processing:

* Remove puntuations like (. , ! $ () ' % @)
* Remowng URLs
* _Lower casing_ : To normalize the data
* _Remove Stop words_ : Stopwords are the most commonly occurring words in a text Which do not provde any valuable information. stopwords like they,
there. this, where, the, ...
* ```Tokenization``` : Tokenizing separates text into units such as sentences or words
* ```Stemming``` : Refers to the process of slicing the end or the beginning of words with the intention of removing affixes. It removes suffces. like "_ing_", "_ly_", "_s_", .... by a simple rule-based approach. It reduces the corpus of words but often the actual words get neglected.
    - _eg_: Entitling Entitled->Entitl
* ```Lemmatization``` : the objective of reductng a word to its base form and groupng together different forms of the same word. For example, the words "_running_", "_runs_" and "_ran_" are all forms of the word "**run**", so "run" is the lemma of all the prevous words.

##### Tokenization:

* Tokenization is one of the fundamental things to do in any text-processing activity.
* Tokenization is breaking the documents or sentences into chunks called tokens. Each token carries a semantic meaningg associated with it.
* Tokenization can be thought of as a segmentation technique wherein you are trying to break down larger pieces of text chunks into smaller meaningful ones.
* Tokens generally comprise words and numbers, but they can be extended to include punctuation marks, symbols, and, at times, understandable emoticons

* Eg:
```
Sentence = "The capital of China is Beijing"

Tokenization: 'The', 'capital', 'of, 'China', 'is', 'Beijing']
```

##### STOP-WORDS: 

* An alternative or a complementary way to eliminate words that are
(most probably) irrelevant for corpus analysis.
* Stop-words are those words that (on their own) do not bear any
information / meaning.
* It is estimated that they represent 20-30% of words in any corpus.
* There is no unique stop-words list. Frequently used lists are available at: ```http://www.ranks.nl/stopwords```
* Potential problems with stop-words removal:
    - the loss of original meaning and structure of text
    - examples:
      ```
      "this is not a good option" -> "option"
      "to be or not to be" -> null
      ```

##### LEMMATIZATION AND STEMMING:

* Two approaches to decreasing variability of words by reducing
different forms of words to their basic / root form.
* ```Stemming``` is a crude heuristic process that chops off the ends of
words without considering linguistic features of the words
* E.g., argue, argued, argues, arguing -> _argu_
<br>
* ```Lemmatization``` refers to the use of a vocabulary and morphological
analysis of words, aiming to return the base or dictionary form of a
word, which is known as the lemma
* E.g., argue, argued, argues, arguing -> _argue_

#### NLP: Feature Extraction / Vectorization

* When we have the colleted text data and do the pre-processing to clean the text data. That data convert to Numerical equivalent representation of the data (also known as Vector Representation) using _**Vectorization**_.

* Text data offers a very umque proposition by not providing any direct representation available for it in terms of numbers. Machines only understand numbers.
* Representing text using numbers is a challenge. At the same time, it is an opportunity to invent and try out approaches to represent text so that the maximum Information can be captured in the process.
* Steps toward transforming text data into mathematical data structures that will provide insights on how to actually represent text using numbers and, consequently, build Natural Language Processmg (NLP) models.
    - **Binary Weights** : ML-based
    - **Exploring the Bag-of-Words** (BoW) architecture (also known as Count Vectorization) : ML-based 
    - **TF-IDF vectors** : ML-based
    - N-gram

* There are Deep Learning based approaches like ```Word-Embedding```. It has 2 types:
    - Continuous Back of Word approach (CBoW)
    - Skip-gram

#### Binary Weight

* Weights take the value of O or 1, to reflect the presence (1) or absence (O) of the term in a particular document
* Example:
    - Docl: Text mining is to identify useful information.
    - Doc2: Useful information is mined from text.
    - Doc3: Apple is delicious.
      
      <img width="317" alt="image" src="https://github.com/thisarakaushan/Generative-AI-and-ChatGPT/assets/125348115/816f0972-1ad3-47d4-bf3a-74e8286be78f">

    - Here still didn't apply any pre-processing step, that's why there are some words like _mining_, _mined_, and _is_. When we apply pre-processing steps like Stemming or Lemmatization those _mining_, _mined_, word will converted to _mine_. Also Stop Word approach will remove the word _is_.
 
* _This approach is not considering any kind of special feature_.

#### Bag-of-Words architecture

* Each sentence can be represented as a vector. The length of this vector would be equal to the size of the vocabulary. Each entry in the vector would correspond to a term in the vocabulary, and the number in that particular entry would be the frequency of the term in the sentence under consideration. The lower limit for this number would be 0.
* Eg: if the paticular token present in the documents 2 times, then frequency id 2 and so on...
```
["We are reading about Natural Language Processing Ilere",
"Natur Language rocessmg ma - g computers comprehend language data",
"Th field of Natural Language Processing is evolving everyday"]

Processed data:
tread', 'natural', 'language', 'computers', 'everyday', 'data',
'evolve', 'field', 'process', 'comprehend', 'make']

BOW matrix:
array([[l., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0.]
      [0., 1., 2., 1., 0., 1., 0., 0., 1., 1., 1.]
      [0., 1., 1., 0., 1., 0., 0., 1., 1., 0., 0.]]
```

* _This approach is **only** consider the frequency of terms_.

#### TF-IDF vectors

* **Frequency** of every term + how **importance** the term is.

* TF-IDF stands for ```Term Frequency-Inverse Document Frequency```, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus.

* Typically, the tf-idf weight is composed by two terms: the first computes the normalized _Term Frequency_ (TF), the number of times a word appears in a document. divided by the total number of words in that document: the second term is the _Inverse Document Frequency_ (IDF). computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific word appears.

* **TF**: Term Frequency. which measures how frequently a term occurs in a document.
```
TF(t) = (Number Of times term t appears in a document) / (Total number Of terms in the document).
```

* **IDF**: Inverse Document Frequency, which measures how important a term is. While computing TE all terms are considered equally important. However it is known that certain terms, such as "is". "of". and "that". may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms While scale up the rare ones, by computing the following:
```
IDF(t) =log_e(Total number of documents / Number of documents with term t in it)
```

* Higher the tf-idf weight, more important the term is.
```
tf-idf weight = TF*IDF
```
* Example:
```
Consider document containing 100 words wherein the word(paticular word) cat appears 3 times.
The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03.

Now, assume we have IO million documents and the word cat appears in one thousand of these.
Then. the inverse document frequency (i.e., idf) is calculated as
log(10,000,000 / 1000) = 4.

Thus, the Tf•idf weight is the product Of these quantities:
tf-idf = tf*idf  0.03 • 4 = 0.12.
```

* _**Drawback**_: ```Context is missing```. Context is nothing but in which propective or in which context a paticular term is used in the sentence or feedback or the review, that part is missing in these approaches.

* This is been taking care by deep learning based approach which is the _Word-embedding_. 

* In python, there is two most properly used packeges and libraries which are dedicated for the text data only.
    1. ```nltk```: Natuarl Language Tool Kit.
    2. ```spacy```; used for handle text data.

<br>

# Deep Learning..................................................................................................

<img width="1000" height="250" alt="image" src="https://github.com/thisarakaushan/Generative-AI-and-ChatGPT/assets/125348115/8db1d82a-9c31-4237-b5e3-e632df321d98">
<br>

## Neural Networks

### Feed-Forward networks (Sequential NN):


<img width="461" alt="image" src="https://github.com/thisarakaushan/Generative-AI-and-ChatGPT/assets/125348115/5affea5d-db23-42f0-bf94-00414da10693">
<br><br>

1. Perceptron (simplest nn) are arranged in layers, with the first layer taking in inputs and the last layer producing outputs. The middle layers have no connection with the external world, and hence are called hidden layers.
2. Each perceptron in one layer is connected to every perceptron on the next layer. Hence information is constantly "fed forward" from one layer to the next., and this explains why these networks are called feed-forward networks.
3. There is no connection among perceptron in the same layer.
4. Except first layer, other every layers do two functionalities.

<br>
<img width="490" alt="image" src="https://github.com/thisarakaushan/Generative-AI-and-ChatGPT/assets/125348115/0e8fe879-7d04-49f4-834e-d8fa5d25b7fb">
<br><br>

* This directed network diagram defines a relationship between the input signals received by the dendrites (x variables), and the output signal (y variable). Just as with the biological neuron, each dendrite's signal is weighted (w values) according to its importance. The input signals are summed by the cell body and the signal is passed on according to an activation function denoted by f.

* A typical artificial neuron with n input dendrites can be represented by the formula shown above. The w weights allow each of the n inputs (denoted by xi) to contribute a greater or lesser amount to the sum of input signals. The net total is used by the activation function f(x), and the resulting signal, y(x), is the output axon.

### Activation Functions:

<br>
<img width="490" alt="image" src="https://github.com/thisarakaushan/Generative-AI-and-ChatGPT/assets/125348115/cd1b6566-e245-4184-8b92-aa870282e32c">
<br><br>

* Universal Approximation Theorem, states that any continuous functions on compact subsets of R^2 can be approximated by a neural network with at least one hidden layer.
* Gradient of the function is important for the training of the neural network.
* Let us now 100k at derivative Of sigmoid function

* Deep Learning has different kind of Neural Networks:
```
1. ANN (Artificial Neural Networks) :- Taken for _Regression_ and _Classification_ tasks

2. CNN (Convolutional Neural Networks) :- Taken for process images and videos analysis
    - R.CNN (Regression with CNN)
    - Fast CNN
    - Faster CNN
    - YOLO (You Only Look Once) : Object Detection Algorithm
   
3. RNN (Recurrent Neural Networks) : Taken for _sequential data analysis_
    - For LSTM (Long Short Term Memory)
    - For GRU ( Gated Recurrent Units)
        * like _time series forecasting_ and _text data analysis_

4. GAN (Generative Adversarial Networks)

5. VAE (Variational Autoencoders)

6. seq2seq model

7. Trnasformers (attention mechanism)
```

[see more](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e) for CNN algorithms...

* Mostly commonly used optimization technique in NN is **Gradient Descent**. We want to optimize the weight for minimum error. This optimizing technique try to adjust the value executing number of time in the backend.

* A NN is known as _**Dense NN**_ where we have the all connection available in between any of the two layers.
    - _Issue_ :- Problem of Overfitting (This happens when we _apply this on test data or live data_ but when we are building this model that will be performed very well)
    - _Solution_ :- randomly drop certain number of connection between any 2 layers (Dropout Layer)

* Broadly there are 3 steps to follow to design any kind of nn;
1. _N.N Architecture_ : specifies how many number of hidden layers you are going to use, number of nodes/nuerons and what is your activation fuction for each layer
   
2. _Compliation_ : Loss function (calculate the error) and select what is your optimizer (Adam, RMS Prop, Momentum, SGD)

3. _Execution / training_ :- parameters like _**Batch Size**_ and _**Epoch**_
    ```
    - Batch Size: when we have 100 records, We can defined whether we want to apply nn or train the nn
                   over all the 100 records or if we want to train nn over some part of the data an each and every execution
        eg: if my batch size = 50, then execution will done on first half of the data  and other half of the data
            repectively within 2 iteration.
    
    - Epoch: if my batch size = 50, epoch is 2, that means how many number of iteration it will take to cover the complete data;
             then it takes 2 iteration to cover all the data, also epoch is 2; overall is 2x2 mean all data need to cover 4 times
    ```

* Example: How to calculate hidden layer input value and output value;

<img width="566" alt="image" src="https://github.com/thisarakaushan/Generative-AI-and-ChatGPT/assets/125348115/e6ffe857-2b47-492f-83e1-971b3dadc742">

<img width="566" alt="image" src="https://github.com/thisarakaushan/Generative-AI-and-ChatGPT/assets/125348115/52155546-583a-40e0-9512-e85807e08745">
<br>

* According to the above figure the calculated output values for o1 = 0.513 and o2 = 0.7729 but actual values are o1 = 0.01 and o2 = 0.99. Now we want to calculate the error between actual value and calculated value.

<img width="566" alt="image" src="https://github.com/thisarakaushan/Generative-AI-and-ChatGPT/assets/125348115/bf0c71a4-b24d-48af-9a93-c5978c0ac05f">

* Now tuning the weight values optimizer try to minimize the error value.
  
<img width="566" alt="image" src="https://github.com/thisarakaushan/Generative-AI-and-ChatGPT/assets/125348115/d3eaf606-c916-43a4-a9be-da4fe5264196">

* For minimize the value we will use the most popular ```Gradient Descent Optimizer``` technique. When it comes to optimize the weight values; each and every optimizer will follow ```Backward propagation```.

<img width="566" alt="image" src="https://github.com/thisarakaushan/Generative-AI-and-ChatGPT/assets/125348115/4331d02f-6599-4b95-a930-e6dfdd5176ad">

<br><br>


### RECURRENT NEURAL NETWORKS.........................................................................................................................................

<br>
<img width="566" alt="image" src="https://github.com/thisarakaushan/Generative-AI-and-ChatGPT/assets/125348115/f62d3b4e-da83-483f-b1f9-7feb977a9b78">

<br><br>

* Uses for Sequence data analysis / sequential process.
* For hidden layer, it has own feedback layer and it works as a memory storage for hidden layer. That's why we are called that layer as _Recurrent_. ( means output of one hidden layer will be input to the next hidden layer node as well as current output layer)

```
1. Temporal lobe and ann: The temporal lobe might be well linked to the ANN with its main contribution.

2. Frontal lobe and rnn: RNN are like short-term memory. You will learn that they can remember things that just happen in a previous couple of observation and apply that knowledge in the going forward. For humans, short-term memory is one of the frontal lobe's funvtions.

3. Occipital lobe and cnn: The occipital lobe is linked to CNN. As we already know, CNNs are responsible for computer vision, recognition of images and objects, which makes them a perfect link to the occipital lobe.

4. Parietal lobe: The parietal lobe is responsible for sensation perception and constructing a spatial-coordination system to represent the world around us, and we are yet to create a neural network which will fit into this category.
```
#### RNN Applications:

1. **One to Many** : this the network with one input and multiple outputs.

<img width="213" alt="image" src="https://github.com/thisarakaushan/Generative-AI-and-ChatGPT/assets/125348115/3b09f58b-2e63-45e0-a6e3-59fda7ac0579">

<br>

2. **Many to One**  : best example for that is the _Sentiment Analysis_, when you have lots of text, such as customer's feedback or review, and you need to gauge what's the chance that this feedback is positive r how positive this comment actually is, or how negative it is.

3. **Many to Many** : network with multiple inputs and multiple outputs.
    - eg:- Machine translation (Google translator) 

<img width="420" alt="image" src="https://github.com/thisarakaushan/Generative-AI-and-ChatGPT/assets/125348115/ebadc7e0-2d46-4686-ac53-17981f8c2ef7">

<br><br>







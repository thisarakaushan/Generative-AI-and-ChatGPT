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
**Discriminative Models:** 

• Discriminative Models are a family of models that do not generate new data points but
  learn to model boundaries around classes in a dataset instead.
• Discriminative model makes predictions on unseen data based on conditional probability
  and can be used either for classification or regression problem statements.
  - Objective: Learn decision boundary for accurate classification 
  - Training Approach: Supervised Learning
  - Type of Learning: Discriminative Modeling
  - Data Generation: No inherent data generation capabilities
  - Decision Boundary: Learn explicit decision boundary between different classes
  

**Generative Models:**

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
  * Generative Adversarial Networks (GANs) are a powerful cla of neural networks that are used for unsupervised learning.       The goal of GANs is to generate new, synthetic data that resembles some known data distribution.
    
2. **(Large) language model:**
  * A (large) language model (LLM) refers to neural networks form eling and generating text data that typically combine         three characteristics i.e. transformer with an attention mechanism, next-word prediction use of large-scale datasets of     text.
  
3. **Prompt learning:** Prompt learning is a method for LLMs.
   
4. **Diffusion probabilistic models:**
  * Diffusion probability models are a class of latent variable models tha are common for various tasks such as image           generation.
   
5. **Reinforcement learning from humanfeedback:**
  * RLHF is used in conversational systems such as ChatGPT (OpenAl, 2022) for generating chat messages, such that new           answers accommodate the previous chat dialogue and ensure that the answers are in alignment with predefined human           preferences.
   
6. **seq2seq:**
  * sequence-to-sequence (seq2seq) refers to machine learning approaches where an input sequence is mapped onto an output       quence.

**Examples:**

1. Transformer:
  * A transformer is a deep learning architecture that adopts t mechanism of self-attention which differentially weights        the importance of each part of the input data.

2. Variational autoencoder:
  * A variational autoencoder (VAE) is a type of neural network th is trained to learn a low-dimensional representation of      the input data.

3. ChatGPT:
  * OpenAl has developed a language model that is close to a conversation with humans. It learns from interactions and          processes information based on its learnings.

4. DALL- E:
  * DALL-E and DALL-E 2 are deep learning models developed by OpenA to generate digital images from natural language            descriptions, called "prompts".

5. LaMDA:
  * LaMDA Al is a conversational Large discourse Model (LLM) developed by Google to power dialogue-based systems that       
    generate natural-sounding human discourse.







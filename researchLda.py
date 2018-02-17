from matplotlib import pyplot as plt
import sklearn as sk
import numpy as np
import os, lda, tokenLists, mainSurveyLda

def topicModel(surveyEntries):
    vectorizer = sk.feature_extraction.text.CountVectorizer()
    surveyMatrix = vectorizer.fit_transform(surveyEntries)
    vocabulary = vectorizer.get_feature_names()

    model_lda = lda.LDA(n_topics = 5, n_iter = 1000, random_state = 1)
    model_lda.fit(surveyMatrix)

    lda_estimate = model_lda.topic_word_
    top_words = 50

    results = os.path.join(os.path.dirname(__file__), "lda_results.txt")
    with open(results, "w") as output:
        for topic_rank, topic_distribution in enumerate(lda_estimate):
            topic_words = np.array(vocabulary)[np.argsort(topic_distribution)][:-(top_words+1):-1]
            output.write('Topic ' + str(topic_rank+1) + ': ' + ' '.join(topic_words)+ '\n\n')

    performance = [0,0,0,0,0]
    for ep in model_lda.doc_topic_:
        if np.argmax(ep) == 0:
            performance[0] += 1
        if np.argmax(ep) == 1:
            performance[1] += 1
        if np.argmax(ep) == 2:
            performance[2] += 1
        if np.argmax(ep) == 3:
            performance[3] += 1
        if np.argmax(ep) == 4:
            performance[4] += 1

    Topics = ('Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5')
    y_pos = np.arange(len(Topics))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, Topics)
    plt.ylabel('Frequency')
    plt.title('Frequency of Topic Models')
    plt.savefig('topicFreq.png', dpi=300, format='png')

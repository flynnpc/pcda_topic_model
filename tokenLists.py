from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
# Pass the survey responses in the form of a list of strings (setup for Spanish)
def cloudPrep(listForCloud):
    stop_words = set(stopwords.words('spanish'))
    stringCloud = ""
    for entry in listForCloud:
        token_list = word_tokenize(entry.lower())
        for word in token_list:
            if (word not in stop_words) & (word not in string.punctuation):
                stringCloud += (" " + word)
    # This returns a single string comprised of all of the respone's words
    # seperated by a space, filtered for stop words and punctuation.
    return stringCloud

def tokenLda(listForLda):
    stop_words = set(stopwords.words('spanish'))
    survey_response = []
    for entry in listForLda:
        token_list = word_tokenize(entry.lower())
        tokens_filtered = [item for item in token_list if (item not in stop_words)&(item not in string.punctuation)]
        survey_response.append(" ".join(tokens_filtered))
    # survey responses is a list of strings representing each survey response,
    # filtered for stop words and punctuation
    return survey_response

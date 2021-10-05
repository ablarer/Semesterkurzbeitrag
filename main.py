import random
import re
import xml.etree.ElementTree as ET
from collections import Counter

import nltk
import nltk.classify.util
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.util import bigrams, trigrams, ngrams


# Ziel der Funktion: 50 einzelne Query-XML-Dateien und 1'000 einzelne Collection-XML-Dateien machen
# Hier nur für die Queries.
def split_save_single_docs():
    # Variablen für das Splitting
    tree = ET.parse('ie1_queries.trec')
    root1 = tree.getroot()

    for doc in root1.findall('DOC'):
        # Get data
        recordId = doc.find('recordId').text
        text = doc.find('text').text
        # View result in console
        # print(recordId, text)
        # Write data
        # Variablen ür die einzelnen Dateien
        root = ET.Element("root")
        doc = ET.SubElement(root, "doc")
        ET.SubElement(doc, "recordId").text = recordId
        ET.SubElement(doc, "text").text = text
        tree = ET.ElementTree(root)
        tree.write(recordId + ".xml")

# Ziel der Funktion: Zum Lernen: Die Sample-Datei so parsen, damit die Funktion split_save_single_docs() einzelne
# Dateien machen kann.
def parsing_sample_XML_file():
    # Parsing for sample.xml
    tree = ET.parse('sample.xml')
    root = tree.getroot()
    print(root.tag)  # data
    print(root.attrib)  # {}
    for child in root:
        print(child.tag, child.attrib)
        # country {'name': 'Liechtenstein'}
        # country {'name': 'Singapore'}
        # country {'name': 'Panama'}
    print(root[0][0].text)  # 1 vom 1. country
    print(root[0][1].text)  # 2008 vom 1. country

    for country in root.findall('country'):
        rank = country.find('rank').text
        name = country.get('name')
        print(name, rank)
        # Liechtenstein 1
        # Singapore 4
        # Panama 68


# Ziel der Funktion: Query-Datei so parsen, damit die Funktion split_save_single_docs() einzelne Dateien machen kann.
def parsing_queries_trec_file():
    tree = ET.parse('ie1_queries.trec')
    root = tree.getroot()
    """
    print(root2.tag) # TREC
    print(root2.attrib) # {}
    for child in root2:
        print(child.tag, child.attrib) # DOC {}
    print(root2[0][0].text)  # 245 vom 1. DOC
    print(root2[0][1].text) # transistor phase splitting circuits vom 1. DOC
    """

    for doc in root.findall('DOC'):
        recordId = doc.find('recordId').text
        text = doc.find('text').text
        print(recordId, text)
        # 245 transistor phase splitting circuits
        # 891 the determination of the orbits of individual meteors by radio methods
        # etc. für alle Dokumente.

# Ziel der Funktion: Collection-Datei so parsen, damit die Funktion split_save_single_docs() einzelne Dateien machen kann.
def parsing_collection_trec_file():
    tree = ET.parse('ie1_collection.trec')
    root = tree.getroot()
    """
    print(root3.tag) # TREC
    print(root3.attrib) # {}
    for child in root3:
        print(child.tag, child.attrib) # DOC {}
    print(root3[0][0].text)  # Analog wie für 'ie1_queries.trec'
    print(root3[0][1].text) # Analog Analog wie für 'ie1_queries.trec'
    """

    for doc in root.findall('DOC'):
        recordId = doc.find('recordId').text
        text = doc.find('text').text
        print(recordId, text)
        # 'recordId' 'text' für alle Dokumente


# NLP Funktionen mit der nltk Bibliothek

def tokenizing():
    rawtext = 'This is a short example text that needs to be cleaned.'
    tokens = nltk.word_tokenize(rawtext)
    print(tokens)


# Folgenden Funktionen (Test 1 bis 4) sind von hier:
# https://towardsdatascience.com/getting-started-with-text-analysis-in-python-ca13590eb4f7
# Test 1 bis 4

def nltk_test_1():
    example_sent = "This is a sample sentence, showing off the stop words filtration."
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(example_sent)

    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    print(filtered_sentence)


def nltk_test_2():
    sen = "Dummy sentence to demonstrate bigrams Dummy sentence to demonstrate bigrams"
    nltk_tokens = word_tokenize(
        sen)  # using tokenize from NLKT and not split() because split() does not take into account punctuation
    print(nltk_tokens)
    # splitting sentence into bigrams and trigrams
    print(list(bigrams(nltk_tokens)))
    print(list(trigrams(nltk_tokens)))
    # creating a dictionary that shows occurances of n-grams in text
    n_gram = 2
    n_gram_dic = dict(Counter(ngrams(nltk_tokens, n_gram)))
    print(n_gram_dic)


def nlkt_test_3():
    # letters only
    raw_text = "this is a test. To demonstrate 2 regex expressions!!"
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)
    print(letters_only_text)
    # keep numbers
    letnum_text = re.sub("[^a-zA-Z0-9\s]+", " ", raw_text)
    print(letnum_text)


def nlkt_test_4():
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    word = "considering"
    stemmed_word = stemmer.stem(word)
    lemmatised_word = lemmatizer.lemmatize(word)
    print(stemmed_word)
    print(lemmatised_word)


# Unteres von https://pythonprogramming.net/text-classification-nltk-tutorial/
def nlkt_test_5():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    random.shuffle(documents)

    print(documents[1])

    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)
    print(all_words.most_common(15))
    print(all_words["stupid"])


if __name__ == '__main__':
    split_save_single_docs()
    # parsing_sample_XML_file()
    # parsing_queries_trec_file()
    # parsing_collection_trec_file()

    tokenizing()
    nltk_test_1()
    nltk_test_2()
    nlkt_test_3()
    nlkt_test_4()
    nlkt_test_5()

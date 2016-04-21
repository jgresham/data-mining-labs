#!/usr/bin/env python
import sys, os, pdb, collections
import nltk, json, re
from bs4 import BeautifulSoup

def main():
	
	articles_path = '/home/0/srini/WWW/674/public/reuters'
	output_path = '/home/7/gresham/5243/lab1/output_1.txt'
	
	feature_vectors = {}
	
	if(sys.argv[1] == 'all'):
		for filename in os.listdir(articles_path):
			articles = BeautifulSoup(open(articles_path+'/'+filename), 'html.parser')			
			for article in articles.findAll('reuters'):
				article_feat_vec = vectorizeArticle(article)
				feature_vectors[article.get('newid')] = article_feat_vec
	
	elif(sys.argv[1] == 'test'):
		filename = os.listdir(articles_path)[1]
		articles = BeautifulSoup(open(articles_path+'/'+filename), 'html.parser')
		for article in articles.findAll('reuters', limit=20):
			article_feat_vec = vectorizeArticle(article)
			feature_vectors[article.get('newid')] = article_feat_vec
		

	f_out = open(output_path, 'w')
	f_out.write('Word Frequency Feature Vector w/ title and topics\n')
	f_out.write('Article Id: (feature, feature value)...\n')
	for article_id, feature_vector in feature_vectors.iteritems():
		f_out.write(article_id + ':\t')
		f_out.write('(title,'+feature_vector['title'].rstrip()+')')
		if len(feature_vector['topics']) > 0:
			f_out.write('(topics,'+' '.join(feature_vector['topics'])+')')
		if len(feature_vector['places']) > 0:
			f_out.write('(places,'+' '.join(feature_vector['places'])+')')
		if feature_vector.get('word_counts') is not None:
			for word, word_count in feature_vector['word_counts'].iteritems():
				f_out.write('('+word+','+str(word_count)+')')
		f_out.write('\n')
	f_out.close()
	print(str(len(feature_vectors)) + ' feature vectors were created')

def vectorizeArticle(article):
	feature_vector = {}
	
	# add class labels and title labels
	feature_vector['topics'] = []
	for topic in article.find('topics').findAll('d'):
		feature_vector['topics'].append(topic.string)

	feature_vector['places'] = []
	for place in article.find('places').findAll('d'):
		feature_vector['places'].append(place.string)

	feature_vector['title'] = ''
	if article.find('title') is not None:
		feature_vector['title'] = article.find('title').string

	# no more features to add if there isn't a body
	if article.find('body') is None:
		return feature_vector

	# add article word count feature
	feature_vector['word_counts'] = {}
	stopwords = nltk.corpus.stopwords.words('english')
	article_text = article.find('body').string.lower()
	article_words = nltk.tokenize.word_tokenize(article_text)
	nonstop_article_words = [w for w in article_words if w not in stopwords]
	nonPunct = re.compile('.*[A-Za-z0-9].*')
	filtered_article_words = [w for w in nonstop_article_words if nonPunct.match(w)]
	filtered_word_counts = collections.Counter(filtered_article_words)
	for key in filtered_word_counts.keys():		# remove words that occur once
		if filtered_word_counts[key] < 2:
			del filtered_word_counts[key]
	feature_vector['word_counts'] = filtered_word_counts
	
	#return set of features
	return feature_vector


if __name__ == '__main__':
	main()

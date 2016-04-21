#!/usr/bin/env python
import sys, os, time, pdb, collections, math
import nltk, json, re
from bs4 import BeautifulSoup
import sklearn
import numpy as np
from minhash import MinHash

def main():

	articles_path = '/home/0/srini/WWW/674/public/reuters'
	feature_vectors = {}
	
	# default distance function is e.d., alternate is cosine simularity
	distance_func = 'ed'
	if len(sys.argv) > 2 and sys.argv[2] == 'cs':
		distance_func = 'cs'
		print 'using cosine simularity'
	
	# feature extraction
	print 'Extracting features from articles...'
	if(sys.argv[1] == 'all'):
		for filename in os.listdir(articles_path)[:6]:
			articles = BeautifulSoup(open(articles_path+'/'+filename), 'html.parser')
			for article in articles.findAll('reuters'):
				article_feat_vec = vectorizeArticle(article)
				if article_feat_vec:
					feature_vectors[article.get('newid')] = article_feat_vec
	
	elif(sys.argv[1] == 'test'):
		filename = os.listdir(articles_path)[1]
		articles = BeautifulSoup(open(articles_path+'/'+filename), 'html.parser')
		for article in articles.findAll('reuters', limit=2000):
			article_feat_vec = vectorizeArticle(article)
			if article_feat_vec:
				feature_vectors[article.get('newid')] = article_feat_vec
		
	# format feature vectors into arrays
	feature_vecs = []
	labels = []
	for article_id, feature_vector in feature_vectors.iteritems():
		feature_vec = []
		for word, count in feature_vector.get('word_counts').iteritems():
			feature_vec.append(word)
		feature_vecs.append(feature_vec)
		labels.append(feature_vector['topics'][0])
	
	print(str(len(feature_vecs)) + ' feature vectors were created')
	
	#'true' jaccard similarity
	start_true = time.clock()
	sets, true_jac = [], []
	for data in feature_vecs:
		sets.append(set(data))
	for i in range(0, len(feature_vecs)-1):
			for j in range(i, len(feature_vecs)-1):
				union = float(len(sets[i].union(sets[j])))
				act = 0
				if union > 0:
					act = float(len(sets[i].intersection(sets[j])))/union
				true_jac.append(act)
	end_true = time.clock()
	print 'true jaccard similarities (baseline) took ' + str(end_true-start_true) + 's to compute'
	
	k_values = [16, 32, 64, 128, 256]
	for k in k_values:

		print str(k)+" k-minhash being created"
		
		start_hash = time.clock()		
		hashes = []
		for data in feature_vecs:
			m = MinHash(num_perm=k)
			for d in data:
				m.update(d.encode('utf8'))
			hashes.append(m)
		
		est_jac = []
		for i in range(0, len(feature_vecs)-1):
			for j in range(i, len(feature_vecs)-1):
				est_jac.append(hashes[i].jaccard(hashes[j]))

		end_hash = time.clock()

		#calculate error
		start_err = time.clock()
		mse = 0
		for i in range(0, len(true_jac)-1):
				mse = mse + math.pow(est_jac[i]-true_jac[i], 2)
		mse = mse / len(true_jac)
		end_err = time.clock()

		print '\tk hash took time: ' + str(end_hash-start_hash) + 's with mse = ' + str(mse)


# returns None if the article doesn't have a label (topics) or text (body)
def vectorizeArticle(article):
	feature_vector = {}
    
	# add class labels and title labels
	feature_vector['topics'] = []
	if not article.find('topics').findAll('d'):
		return None
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
		return None

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
	
	# add lexical diversity feature
	feature_vector['diversity'] = float(len(filtered_word_counts)) / len(filtered_article_words)
	
	# add length of article by filtered word count (no stopwords, punct, etc.)
	feature_vector['num_of_words'] = len(filtered_article_words)

	#return set of features
	return feature_vector


if __name__ == '__main__':
	main()

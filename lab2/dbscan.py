#!/usr/bin/env python
import sys, os, time, pdb, collections, math
import nltk, json, re
from bs4 import BeautifulSoup
import sklearn
import numpy as np
from sklearn.cluster import KMeans

def main():

	articles_path = '/home/0/srini/WWW/674/public/reuters'
	feature_vectors = {}
	
	# default distance function is e.d., alternate is cosine simularity
	distance_func = 'ed'
	if len(sys.argv) > 2 and sys.argv[2] == 'cs':
		distance_func = 'cs'
		print 'using cosine simularity'
	
	start_all = time.clock()
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
		
	# finally, calculate td-idf feature, after all document words are counted
	num_docs_with_word = {}
	for article_id, feature_vector in feature_vectors.iteritems():
		if feature_vector.get('word_counts') is not None:
			for word, word_count in feature_vector['word_counts'].iteritems():
				num_docs_with_word[word] = 1 + num_docs_with_word.get(word, 0)
	for article_id, feature_vector in feature_vectors.iteritems():
		if feature_vector.get('word_counts') is not None:
			feature_vector['tf-idf'] = {}
			for word, word_count in feature_vector['word_counts'].iteritems():
				feature_vector['tf-idf'][word] = (float(word_count)/feature_vector['num_of_words']) * math.log(float(len(feature_vectors))/num_docs_with_word[word])
	
	# keep class labels
	feature_words = num_docs_with_word.keys()
	feature_vecs = []
	labels = []
	for article_id, feature_vector in feature_vectors.iteritems():
		feature_vec = []
		for word in feature_words:
			if feature_vector.get('tf-idf') is not None and feature_vector.get('tf-idf').get(word) is not None:
				feature_vec.append(feature_vector.get('tf-idf').get(word))
			else:
				feature_vec.append(0)
		feature_vec.append(feature_vector.get('diversity', 0))
		feature_vec.append(feature_vector.get('num_of_words', 0))
		feature_vecs.append(feature_vec)
		labels.append(feature_vector['topics'][0])
	print str(len(feature_vec)) + ' features per sample'
	print(str(len(feature_vecs)) + ' feature vectors were created')

	# normalize data
	data = np.array(feature_vecs)
	data = sklearn.preprocessing.normalize(data, axis=0)
	if distance_func == 'ed':
		dist_mat = sklearn.metrics.pairwise.pairwise_distances(data, metric='euclidean')
	else:
		dist_mat = sklearn.metrics.pairwise.pairwise_distances(data, metric='cosine')

	# run training algorithm using different parameters (n = # of clusters)
	eps_e = [0.2]
	min_points_m = [3]
	for e in eps_e:
		for m in min_points_m:
			print "dbscan with e=" + str(e) + " and min_points="+str(m)+' running...'

			start = time.clock()
			clusterer = sklearn.cluster.DBSCAN(eps=e, min_samples=m, metric='precomputed')
			cluster = clusterer.fit(dist_mat)
			end = time.clock()
				
			core_samples_mask = np.zeros_like(cluster.labels_, dtype=bool)
			core_samples_mask[cluster.core_sample_indices_] = True
			cluster_labels = cluster.labels_.tolist()
			n = len(cluster_labels)

			print "\tn=" + str(n) + " clusters created in "+str(end-start)+"s"
			print "\tsilhouette avg= " + str(sklearn.metrics.silhouette_score(data, np.array(cluster_labels)))
		
			# compute stats for each cluster
			cluster_entropys = []
			for cluster_n in range(n):
				cluster_size = cluster_labels.count(cluster_n)
				if cluster_size < 1:
					continue
				entropy = 0
				label_counts = {}
				label_indicies = [i for i, val in enumerate(cluster_labels) if val==cluster_n]
				print '\t'+str(cluster_n)+'th cluster: '+str(cluster_size)+' elements'
				# counting labels in each cluster
				for label_index in label_indicies:
					label = labels[label_index]
					label_counts[label] = 1 + label_counts.get(label, 0)
				# entropy
				for label, label_count in label_counts.iteritems():
					prob_label_in_cluster = float(label_count)/cluster_size
					entropy -= prob_label_in_cluster*math.log(prob_label_in_cluster)
					if label:
						print '\t\t'+label+' occurs '+str(label_count)+' times'
					else:
						print '\t\tunknown label occurs '+str(label_count)+' times'
				print '\tentropy= ' + str(entropy)
				cluster_entropys.append(entropy)
				# variance
				cluster_samples = [data[x] for x in label_indicies]
				if len(cluster_samples) < 2:
					var = 0
				else:
					var = np.var(cluster_samples)
				print '\tvariance(skew)= ' + str(var)

	print 'time to process & cluster:'+str(time.clock()-start_all)+'s'

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

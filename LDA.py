from __future__ import print_function                   
import nltk
import string
from string import *
from nltk.corpus import stopwords
import sys,os
import numpy as np
import operator
from sklearn.svm import SVC
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import math

def get_data(folder):
	traindir = [folder]
	# classes = ['business/' , 'entertainment/' , 'politics/' , 'sport/' , 'tech/']
	classes = ['entertainment/', 'sport/' , 'politics/']
	# classes = ['football/', 'rugby/','tennis/','athletics/', 'cricket/']
	# classes = ['football/', 'cricket/','athletics/']
	print("Reading Dataset")
	dataValues = []
	classCount = []
	for idir in traindir:
			for c in classes:
				print(c)
				count = 0
				listing = os.listdir(idir+c)
				for filename in listing:
					count += 1
					if count>100:
						break
					#f = open(idir+c+filename,'r')
					#t = f.read()
					# print(idir+c+filename)
					with open(idir+c+filename,'r') as inFile, open('outputFile.txt','w') as outFile:
						for line in inFile.readlines():
							print(" ".join([word for word in line.lower().translate(str.maketrans('', '', string.punctuation)).split() 
								if len(word) >=4 and word not in stopwords.words('english')]), file=outFile)
					f = open('outputFile.txt','r')
					t = f.read()
					t=nltk.word_tokenize(t)
					t=nltk.pos_tag(t)
					# print("hello")
					f.close()
					# print(t)
					nt=[]
					for word in t:
						if word[1]=='NN' or word[1]=='NNS' or word[1]=='NNP':
							nt.append(word[0])
					# nt=[word[0] if word[1]=='NN' or word[1]=='NNS' or word[1]=='NNP' for word in t]
					dataValues.append(nt)
				classCount.append(count-1)		
	#print(classCount)
	print("Reading Done")
	return(dataValues,classCount)

def get_testdata(folder):
	traindir = [folder]
	classes = ['business/' , 'entertainment/' , 'politics/']
	dataValues = []
	classCount = []
	for idir in traindir:
			for c in classes:
				print(c)
				count = 0
				listing = os.listdir(idir+c)
				for filename in listing:
					count += 1
					# if count>1000:
					# 	break
					#f = open(idir+c+filename,'r')
					#t = f.read()
					print(idir+c+filename)
					with open(idir+c+filename,'r') as inFile, open('outputFile.txt','w') as outFile:
						for line in inFile.readlines():
							print(" ".join([word for word in line.lower().translate(str.maketrans('', '', string.punctuation)).split() 
								if len(word) >=4 and word not in stopwords.words('english')]), file=outFile)
					f = open('outputFile.txt','r')
					t = f.read()
					f.close()
					t=nltk.word_tokenize(t)
					t=nltk.pos_tag(t)
					nt=[]
					for word in t:
						if word[1]=='NN' or word[1]=='NNS' or word[1]=='NNP':
							nt.append(word[0])
					# nt=[word[0] if word[1]=='NN' or word[1]=='NNS' or word[1]=='NNP' for word in t]
					dataValues.append(nt)
				classCount.append(count)		
	#print(classCount)
	return(dataValues,classCount)

def get_vocab(docs):
	vocab=list()
	for doc in docs:
		temp_vocab=np.unique(doc)
		for word in temp_vocab:
			if word not in vocab:
				vocab.append(word)
	return vocab

def get_wt(docs,topics,vocab):
	wt={}
	for topic in topics:
		wt[topic]={}
		for word in vocab:
			wt[topic][word]=0
	return wt

def get_ta(docs,topics):
	ta=[]
	for doc in docs:
		dic={}
		for word in doc:
			ti=np.random.randint(0,len(topics))
			dic[word]=topics[ti]
		ta.append(dic)
	return ta

##Generate initial word count and topic assignment matrices##
def generate_initial_matrices(docs,topics,vocab):
	wt=get_wt(docs,topics,vocab)
	ta=get_ta(docs,topics)
	i=0
	for doc in docs:
		for word in doc:
			topic=ta[i][word]
			wt[topic][word]+=1
		i+=1
	return wt,ta

##Generating a document topic count matrix##
def get_dt(docs,topics,wt,ta):
	i=0
	dt=[]
	for doc in docs:
		dic={}
		for topic in topics:
			dic[topic]=np.sum(ta[i][word]==topic for word in doc)
		dt.append(dic)
		i+=1
	return dt

def lda_model(alpha ,eta , n_iter, docs,topics,ta,wt,dt , vocab):
	topic_word = dict()
	for t in topics:
		topic_word[t] = np.sum(wt[t][w] for w in vocab)
	for x in range(n_iter):
		i=0
		for doc in docs:
			for word in doc:
				init_topic = ta[i][word]
				#Sampled
				wt[init_topic][word]-=1
				dt[i][init_topic]-=1
				tot_topic = len(doc)-1
				topic_word[init_topic] -=1
				prob_word = list()
				for t in topics:
					# Probability of the word in a topic
					# tot_word = np.sum(wt[t][w] for w in vocab)
					prob_w_t = (wt[t][word]+eta)/float(topic_word[t] + len(vocab)*eta )
					# Probability of the topic in a document
					prob_t_d = (dt[i][t]+alpha)/float(tot_topic + len(topics)*alpha)
					prob_word.append(prob_t_d*prob_w_t)
				#new_topic = np.random.choice(topics,1, prob_word)
				m = max(prob_word)
				for t in range(len(topics)):
					if(prob_word[t]==m):
						new_topic = topics[t]
				#new_topic = new_topic[0]
				wt[new_topic][word]+=1
				topic_word[new_topic] +=1
				dt[i][new_topic]+=1
				ta[i][word] = new_topic 
			i+=1
	return ta,wt,dt

def get_theta(docs , topics , dt , alpha):
	theta = list()
	i=0
	for doc in docs:
		theta.append(dict((t,(dt[i][t]+alpha)/float(len(doc) + (len(topics))*alpha)) for t in topics))
		i+=1
	return theta

def get_phi(topics ,vocab, wt , eta):
	phi = dict()
	i=0
	for t in topics:
		tot_word = np.sum(wt[t][w] for w in vocab)
		phi[t] =dict((w,(wt[t][w]+eta)/float(tot_word+eta)) for w in vocab)
		i+=1
	return phi

def print_topic(phi):
	for x in phi:
		y = sorted(phi[x].items(), key=operator.itemgetter(1) , reverse=True)
		print(x,end=' ')
		print("topics => ",end=' ')
		m = y[:20]
		mr = list(x  for x in m)
		for qw in mr:
			print(qw[0] , end=' ')
		print("")

def calculateAccuracy(y_predicted_test,y_test):
	correct = 0
	totalCount = len(y_test)
	for i in range(len(y_test)):
		if y_predicted_test[i] == y_test[i]:
			correct = correct + 1
	accuracy = float(correct)/float(totalCount)
	accuracy *= 100
	print("SVM Accuracy = " , end=' ')
	print(accuracy)

def trainSVM(docs,theta,classCount):
	X = []
	Y = []
	X_test = []
	Y_test = []
	X_train = []
	Y_train = []
	giveClass = 1
	for a in range(len(classCount)):
		for b in range(classCount[a]):
			Y.append(giveClass)
		giveClass += 1
	
	#test_giveClass = 1
	#for a in range(len(test_classCount)):
	#	for b in range(test_classCount[a]):
	#		Y_test.append(test_giveClass)
	#	test_giveClass+=1
	
	i=0
	for doc in docs:
		X.append(list(theta[i].values()))
		i+=1

	temp = 0
	# print(len(classCount))

	for a in range(len(classCount)):
		#for b in range(classCount[a]):
		#	print(temp)
		xtrain = X[int(temp):int(temp+(0.6*classCount[a]))]
		for label in xtrain:
			X_train.append(label)
		xtest = X[int((0.6*classCount[a])+temp):int(temp+classCount[a])]
		for label in xtest:
			X_test.append(label)
		ytrain = Y[int(temp):int(temp+(0.6*classCount[a]))]
		for label in ytrain:
			Y_train.append(label)
		ytest=Y[int((0.6*classCount[a])+temp):int(temp+classCount[a])]
		for label in ytest:
			Y_test.append(label)
		temp += classCount[a]

	# print(len(X_train),len(Y_train))
	# print(X_train)
	# print(Y_train)

	#j=0
	#for doc in test_docs:
	#	X_test.append(list(test_theta[j].values()))
	#	j+=1
	#X_test = X[250:]
	#X_train = X[:250]
	#Y_test = Y[250:]
	#for i in range(len(Y_test)):
	#	Y_test[i] = Y_test[i] - 5 #subtract_No_of_topics
	#Y_train = Y[:250]
	#print(len(X),len(Y))
	#print(len(X_train),len(Y_train))
	#print(X_train)
	#print(Y_train)
	clf=SVC( kernel = 'rbf')
	clf.fit(X_train,Y_train)
	y_predicted_test=clf.predict(X_train)
	# print(y_predicted_test)
	calculateAccuracy(y_predicted_test,Y_train)

def getInterIntraDistance(theta, classCount, docs , n_top):
	
	i=0
	X = []
	
	for doc in docs:
		X.append(list(theta[i].values()))
		i+=1
	
	j=0
	temp = 0
	base = 0
	Nooftopics = n_top
	meanVector=[]

	# Calculating Mean Vector
	for a in range(len(classCount)):
		meanVector.append([])
		for c in range(Nooftopics):
			meanVector[a].append(0)
			for b in range(classCount[a]):
				meanVector[a][c] += X[temp + b][c]
			meanVector[a][c] = meanVector[a][c]/(classCount[a])
		temp += classCount[a]

	# Calculating InterDistance within different classes
	InterDistance =  np.zeros(shape=(len(classCount),len(classCount))) 
	for a in range(len(classCount)):
		for b in range(len(classCount)):
			if(a!=b):
				for c in range(Nooftopics):
					InterDistance[a][b] += 	pow((meanVector[a][c]-meanVector[b][c]),2)
				InterDistance[a][b] = math.sqrt(InterDistance[a][b])

	# Calculating Intradistance within different classes
	temp2 = 0
	get_tempDistance = 0
	IntraClass = np.zeros(shape=(len(classCount),1))
	for a in range(len(classCount)):
		for b in range(classCount[a]):
			get_tempDistance = 0
			for c in range(Nooftopics):
				get_tempDistance += pow((X[temp2+b][c]-meanVector[a][c]),2)
			get_tempDistance = math.sqrt(get_tempDistance)
			#IntraClass[a] += abs(X[temp2+b][c]-meanVector[a][c])
			IntraClass[a] += get_tempDistance
		IntraClass[a] = abs((IntraClass[a]-(0.15*classCount[a]))/classCount[a])
		temp2 += classCount[a] 

	# Printing different classes
	print("===InterDistances===")
	print(InterDistance)
	print("===IntraDistances===")
	print(IntraClass)

if __name__=="__main__":	
	docs,classCount=get_data(sys.argv[1])
	#print(classCount)
	# print(classCount)	
	#test_docs,test_classcount=get_testdata(sys.argv[2])
	vocab = get_vocab(docs)
	#test_vocab = get_vocab(test_docs)
	# print("Number of topics Required")
	# n_top = 
	topics=['t1','t2','t3']
	# topics=['topic1','topic2']
	####Parameters for LDA
	print("Starting LDA Modelling")
	alpha=0.5
	eta=0.05
	itr=100
	wt,ta=generate_initial_matrices(docs,topics,vocab)
	dt=get_dt(docs,topics,wt,ta)
	ta,wt,dt = lda_model(alpha,eta,itr,docs,topics,ta,wt,dt,vocab)
	theta = get_theta(docs,topics,dt,alpha)
	phi = get_phi(topics,vocab,wt,eta)
	print("Created LDA Model")
	print("---------------Calculating Inter and Intra Distances---------------------")
	getInterIntraDistance(theta , classCount , docs , len(topics))
	# pca = PCA(n_components=3)
	new_data = list()
	for doc in theta:
		haha = list()
		for topic in topics:
			haha.append(doc[topic])
		new_data.append(haha)
	# pca.fit(new_data)
	# new_data = pca.transform(new_data)
	fig = pyplot.figure()
	ax = Axes3D(fig)
	labels = list()
	colours = ['g' , 'r' , 'b' , 'y' , 'purple']
	j=0
	for x in classCount:
		for i in range(x):
			labels.append(colours[j])
		j+=1

	j=0
	for doc in new_data:
		ax.scatter(doc[0] , doc[1] , doc[2], c = labels[j])
		j+=1
	print("PCA Plot")
	pyplot.show()
	print("PCA Plot Done")

	# test_wt,test_ta=generate_initial_matrices(test_docs,topics,test_vocab)
	# test_dt=get_dt(test_docs,topics,test_wt,test_ta)
	# test_ta,test_wt,test_dt = lda_model(alpha,eta,itr,test_docs,topics,test_ta,test_wt,test_dt,test_vocab)
	# test_theta = get_theta(test_docs,topics,test_dt,alpha)
	# test_phi = get_phi(topics,test_vocab,test_wt,eta)
	
	# trainSVM(docs,theta,classCount,test_docs,test_theta,test_classcount)
	trainSVM(docs,theta,classCount)
	# print("Theta aa rha hai")
	# print(theta)
	# print("Phi aa rha hai")
	print("-------------------Top 20 Words of each topic----------------------------")
	print_topic(phi)
	# print(phi)



# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from random import shuffle
# from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import io, sys
import math as calc
import re, string
import csv

def remove_punctuation(text):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub('', text)

def compile_pos_neg_words():
    pos_neg_words = []
    with open('PosNegWords.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            pos_neg_words.append(row)
    return pos_neg_words

def compile_stop_words():
    stopset = []
    with open('stopwords.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            stopset.append(row)     
    return stopset
            
def tokenize(text):
    text = remove_punctuation(text)
    text = text.lower()
    text = re.split("\W+", text)
    words = []
    stopset = compile_stop_words()
    #stop = stopwords.words('english')
    stop = stopset[0]
    pos_neg_words = compile_pos_neg_words()
    for word in text:
        if word not in stop and text.count(word) >= 15:
            words.append(word)
        elif [word] in pos_neg_words and text.count(word) >= 7:
            words.append(word)
    return np.unique(words)

class MaxEnt(Classifier):
    def get_model(self): return None
                 
    def set_model(self, model): pass

    model = property(get_model, set_model)
    
    def lambda_weight(self, label, feature):
        return self.parameters[self.labels[label], feature]

    def create_feature_vectors(self, instance):
        feature_vector = [self.features[feature] for feature in instance.features() if feature in self.features]
        feature_vector.append(0)
        return feature_vector

    def chop_up(self, train_instances, batch_size):
        shuffle(train_instances)
        mini_batches = [train_instances[x:(x+batch_size)] for x in range(0, len(train_instances), batch_size)]
        return mini_batches

    def compute_observation(self, mini_batch):
        observation = np.zeros((len(self.labels), len(self.features)))
        for instance in mini_batch:
            observation[self.labels[instance.label]][instance.feature_vector] += 1
        return observation

    def compute_estimation(self, mini_batch):
        estimation = np.zeros((len(self.labels), len(self.features)))
        for instance in mini_batch:
            for label in self.labels:
                probability = self.compute_probability(label, instance)
                estimation[self.labels[label]][instance.feature_vector] += probability
        return estimation

    def gradient(self, mini_batch):
        observation = self.compute_observation(mini_batch)
        estimation = self.compute_estimation(mini_batch)
        return observation - estimation

    def compute_probability(self, label, instance):
        top = sum(self.lambda_weight(label, instance.feature_vector))
        bottom = [sum(self.lambda_weight(label, instance.feature_vector)) for label in self.labels]
        posterior = calc.exp(top - sp.misc.logsumexp(bottom))
        return posterior
    
    def compute_negative_log_likelihood(self, batch):
        log_likelihood = sum([calc.log(self.compute_probability(instance.label, instance)) for instance in batch])
        return log_likelihood * -1
    
    def accuracy(self, test, verbose=sys.stderr):
        correct = [self.classify(x) == x.label for x in test]
#        if verbose:
#            print("%.2d%% " % (100 * sum(correct) / len(correct)), file=verbose)
        return float(sum(correct)) / len(correct)
    
    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""
        features = 1
        labels = 0
        features_dict = {"___BIAS___": 0}
        labels_dict = {}

        for instance in instances:
            text = " ".join(instance.features())
            features_tokenized = tokenize(text)
#            features_tokenized = instance.features()
            
            try:
                label_test = labels_dict[instance.label]
            except KeyError:
                labels_dict[instance.label] = labels
                labels += 1
                
            for feature in features_tokenized:
                try:
                    feature_test = features_dict[feature]
                except KeyError:
                    features_dict[feature] = features
                    features += 1
        
        print(features)
        
        self.parameters = np.zeros((labels, features))
        self.labels = labels_dict
        self.features = features_dict

        for instance in instances:
            instance.feature_vector = self.create_feature_vectors(instance)

        for instance in dev_instances:
            instance.feature_vector = self.create_feature_vectors(instance)        
                
        self.train_sgd(instances, dev_instances, 0.0001, 1)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient."""
        negative_log_likelihood = float("inf")
        decreasing = True
        no_changes = 0
        data_points = 1
        x_data = []
        y_data = []
        calculated_negative_log_likelihood = 0

        while decreasing:
            mini_batches = self.chop_up(train_instances, batch_size)

            for mini_batch in mini_batches:
                gradient = self.gradient(mini_batch)
                self.parameters += gradient * learning_rate
                calculated_negative_log_likelihood = self.compute_negative_log_likelihood(dev_instances)
                y_data.append(self.accuracy(dev_instances))
                x_data.append(data_points)
                data_points =+ 1
                
            if calculated_negative_log_likelihood >= negative_log_likelihood:
                no_changes += 1
                print("New Log Likelihood: " + str(calculated_negative_log_likelihood))
                if no_changes == 2:
                    decreasing == False
            else:
                no_changes = 0
                negative_log_likelihood = calculated_negative_log_likelihood
                print("Log Likelihood: " + str(negative_log_likelihood))
                
        with open('batchsize1.csv', 'wb') as writefile:
            csv_writer = csv.writer(writefile)
            for ii in range(len(x_data)):
                csv_writer.writerow([x_data[ii], y_data[ii]])
                
#        plt.plot(x_data, y_data)
#        plt.xlabel('Number of Updates')
#        plt.ylabel('Accuracy')
#        plt.savefig("batchsize1", format="png")

    def classify(self, instance):
        rating = ""
        probability = 0
        instance.feature_vector = self.create_feature_vectors(instance)
        for label in self.labels:
            calculated_probability = self.compute_probability(label, instance)
            if calculated_probability > probability:
                rating = label
                probability = calculated_probability
        return rating

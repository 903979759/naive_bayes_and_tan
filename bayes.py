# MOHAN RAO DIVATE KODANDARAMA
# divatekodand@wisc.edu
# CS USERID: divate-kodanda-rama
#CS760 HW4

# Homework Assignment #4
# Part 1 takes 50 points, Part 2 takes 25 points, and Part 3 takes 25 points. Part 3 also has 20 points extra credit.
# Part 1
# For this homework, you are to write a program that implements both naive Bayes and TAN (tree-augmented naive Bayes). Specifically, you should assume:
# Your code is intended for binary classification problems.
# All of the variables are discrete valued. Your program should be able to handle an arbitrary number of variables with possibly different numbers of values for each variable.
# Laplace estimates (pseudocounts of 1) are used when estimating all probabilities.
# For the TAN algorithm. Your program should:
#
# Use Prim's algorithm to find a maximal spanning tree (but choose maximal weight edges instead of minimal weight ones). Initialize this process by choosing the first variable in the input file for Vnew. If there are ties in selecting maximum weight edges, use the following preference criteria: (1) prefer edges emanating from variables listed earlier in the input file, (2) if there are multiple maximum weight edges emanating from the first such variable, prefer edges going to variables listed earlier in the input file.
# To root the maximal weight spanning tree, pick the first variable in the input file as the root.
# Your program should read files that are in the ARFF format. In this format, each instance is described on a single line. The variable values are separated by commas, and the last value on each line is the class label of the instance. Each ARFF file starts with a header section describing the variables and the class labels. Lines starting with '%' are comments. See the link above for a brief, but more detailed description of the ARFF format. Your program needs to handle only discrete variables, and simple ARFF files (i.e. don't worry about sparse ARFF files and instance weights). Example ARFF files are provided below. Your program can assume that the class variable is named 'class' and it is the last variable listed in the header section.
#
# The program should be called bayes and should accept four command-line arguments as follows:
# bayes <train-set-file> <test-set-file> <n|t>
# where the last argument is a single character (either 'n' or 't') that indicates whether to use naive Bayes or TAN.
#
# If you are using a language that is not compiled to machine code (e.g. Java), then you should make a small script called bayes that accepts the command-line arguments and invokes the appropriate source-code program and interpreter, as you did with the previous homeworks.
#
# Your program should determine the network structure (in the case of TAN) and estimate the model parameters using the given training set, and then classify the instances in the test set. Your program should output the following:
#
# The structure of the Bayes net by listing one line per variable in which you indicate (i) the name of the variable, (ii) the names of its parents in the Bayes net (for naive Bayes, this will simply be the 'class' variable for each other variable) separated by whitespace.
# One line for each instance in the test-set (in the same order as this file) indicating (i) the predicted class, (ii) the actual class, (iii) and the posterior probability of the predicted class (rounded to 12 digits after the decimal point).
# The number of the test-set examples that were correctly classified.
# You can test the correctness of your code using lymph_train.arff and lymph_test.arff, as well as vote_train.arff and vote_test.arff. This directory contains the outputs your code should produce for each data set.
#
# Part 2
# For this part, use stratified 10-fold cross validation on the chess-KingRookVKingPawn.arff data set to compare naive Bayes and TAN. Be sure to use the same partitioning of the data set for both algorithms. Report the accuracy the models achieve for each fold and then use a paired t-test to determine the statistical significance of the difference in accuracy. Report both the value of the t-statistic and the resulting p value.
# You can use a t-test calculator, such as this one for this exercise.
#
# Part 3 - Written Exercises
# This part consists of some written exercises. Download from here. You can use this latex template to write your solution.
# Submitting Your Work
# You should turn in your work electronically using the Canvas course management system. Turn in all source files and your runnable program as well as a file called hw4.pdf that shows your work for Part 2 and 3. All files should be compressed as one zip file named <Wisc username>_hw4.zip. Upload this zip file at the course Canvas site.


# For Python 2 / 3 compatability
from __future__ import print_function

import sys

from scipy.io import arff
import scipy
from io import StringIO
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
# import sklearn
# from sklearn import metrics
import pdb
# print("#CS760 HW4")




# #TODO:REMOVE THIS LATER
# import sklearn
#
# from scipy import stats



class Tan:
    ''' TAN Classifier - Stores the parameters of the Model'''

    def __init__(self, py, cpd_dict, parents, parents_id):
        self.py = py
        self.cpd_dict = cpd_dict
        self.parents = parents
        self.parents_id = parents_id

    def predict(self, Xy, attr, attr_types_ranges, out_labels):
        """
        posterior probability  = (likelihood * prior) / (total probability of the sample)
        :param Xy:
        :param attr:
        :param attr_types_ranges:
        :param out_labels:
        :return:
        """
        predicted_probs = []
        predicted_labels = []
        n_features = len(attr) - 1  # Last attribute is the class
        n_correct = 0
        for idx, example in enumerate(Xy):
            likelihood1 = 1.0
            likelihood0 = 1.0
            if example[-1] == attr_types_ranges[-1][-1][-1]:
                y = 1
            else:
                y = 0
            for ida, attribute in enumerate(example):
                # attribute_utf = attribute.decode('UTF-8')
                attribute_utf = attribute
                if ida == n_features:
                    continue
                else:
                    n = get_attr_val_idx(ida, attribute_utf, attr_types_ranges)
                    if ida == 0:
                        p = 0
                    else:
                        p_index = self.parents_id[ida][-1]
                        # p = get_attr_val_idx(p_index, example[p_index].decode('UTF-8'), attr_types_ranges)
                        p = get_attr_val_idx(p_index, example[p_index], attr_types_ranges)
                    likelihood0 *= self.cpd_dict[attr[ida]][0, p, n]
                    likelihood1 *= self.cpd_dict[attr[ida]][1, p, n]

            prior0 = self.py[0]
            prior1 = self.py[1]
            total_prob = likelihood0 * prior0 + likelihood1 * prior1
            p_y_x = (likelihood1 * prior1) / total_prob
            # print(idx, p_y1_x)
            # TODO: Check the threshold for equality
            if p_y_x >= 0.5:
                label = out_labels[-1][1]
            else:
                label = out_labels[-1][0]
                p_y_x = 1 - p_y_x

            # if label == example[-1].decode('UTF-8'):
            if label == example[-1]:
                n_correct += 1
            predicted_probs.append(p_y_x)
            predicted_labels.append(label)

        return predicted_probs, predicted_labels, n_correct


def get_attr_val_idx(ida, attribute, attr_types_ranges_train):
    attr_range = attr_types_ranges_train[ida][-1]
    idx = attr_range.index(attribute)
    return idx

def train_naive_bayes(Xy_train, attr_train, attr_types_ranges_train, out_labels):
    '''
    :param Xy_train:
    :param attr_train:
    :param attr_types_ranges_train:
    :param out_labels:
    :return: dictionary containing parameters of naive bayes classifier
    '''
    naive_bayes_freq = {}           # To store the count of values a parameter takes given the value of y. row number is y
    naive_bayes_params = {}         # P(xi | y) - parameters of the model, P(y)
    for idx, attr in enumerate(attr_train):
        #print(attr)
        n_vals = len(attr_types_ranges_train[idx][-1])    # Number of values a nomial feature can take
        #TODO: intitializing to one - Laplace estimates (pseudocounts of 1)
        naive_bayes_freq[attr] = np.zeros((2, n_vals), dtype=np.int)
        naive_bayes_params[attr] = np.zeros((2, n_vals), dtype=np.float)

    for id, example in enumerate(Xy_train):
        label = example[-1]
        # if label.decode('UTF-8') == out_labels[-1][0]:
        if label == out_labels[-1][0]:
            y_idx = 0
        else:
            y_idx = 1
        for ida, attribute in enumerate(example):
            #print(id, ida, attribute)
            # attr_utf = attribute.decode('UTF-8')
            attr_utf = attribute
            attr_val_idx = get_attr_val_idx(ida, attr_utf, attr_types_ranges_train)
            naive_bayes_freq[attr_train[ida]][y_idx][attr_val_idx] += 1

    n_y= [naive_bayes_freq[attr_train[-1]][0][0], naive_bayes_freq[attr_train[-1]][1][1]]
    n_examples = n_y[0] + n_y[1]
    p_y = [(1.0 * n_y[0] + 1) / (n_examples + 2), (1.0 * n_y[1] + 1) / (n_examples + 2)]        #Laplace Estimates

    for idx, attribute in enumerate(attr_train):
        n_attr_vals = len(attr_types_ranges_train[idx][-1])
        for y_idx in range(2):
            for feature_val in range(n_attr_vals):
                naive_bayes_params[attribute][y_idx][feature_val] = (1.0 * naive_bayes_freq[attribute][y_idx][feature_val] + 1.0) / (n_y[y_idx] + n_attr_vals)  # Laplace Estimates

    naive_bayes_params[attr_train[-1]][0][0] = p_y[0]
    naive_bayes_params[attr_train[-1]][1][1] = p_y[1]

    # print(naive_bayes_freq)
    # print(naive_bayes_params)

    return naive_bayes_params

def predict_n_bayes(naive_bayes_params, Xy, attr, attr_types_ranges, out_labels):
    """
    posterior probability  = (likelihood * prior) / (total probability of the sample)
    :param naive_bayes_params:
    :param Xy:
    :param attr:
    :param attr_types_ranges:
    :param out_labels:
    :return:
    """
    predicted_probs = []
    predicted_labels = []
    n_features = len(attr) - 1      # Last attribute is the class
    n_correct = 0
    for idx, example in enumerate(Xy):
        likelihood1 = 1.0
        likelihood0 = 1.0
        for ida, attribute in enumerate(example):
            # attribute_utf = attribute.decode('UTF-8')
            attribute_utf = attribute
            if attribute_utf == attr_types_ranges[-1][-1][0] or attribute_utf == attr_types_ranges[-1][-1][1]:
                continue
            else:
                attr_val_idx = get_attr_val_idx(ida, attribute_utf, attr_types_ranges)
                likelihood0 *= naive_bayes_params[attr[ida]][0][attr_val_idx]
                likelihood1 *= naive_bayes_params[attr[ida]][1][attr_val_idx]

        prior0 = naive_bayes_params[attr[-1]][0][0]
        prior1 = naive_bayes_params[attr[-1]][1][1]
        total_prob = likelihood0 * prior0 + likelihood1 * prior1
        p_y_x = (likelihood1 * prior1) / total_prob
        # print(idx, p_y1_x)
        #TODO: Check the threshold for equality
        if p_y_x >= 0.5:
            label = out_labels[-1][1]
        else:
            label = out_labels[-1][0]
            p_y_x = 1 - p_y_x
        #pdb.set_trace()
        # if label == example[-1].decode('UTF-8'):
        if label == example[-1]:
            n_correct += 1
        predicted_probs.append(p_y_x)
        predicted_labels.append(label)


    return predicted_probs, predicted_labels, n_correct

def log_2(num):
    return ((1.0 * math.log(num)) / math.log(2.0))

def compute_mi(pxi_y, pxj_y, p_y, pxij_y, pxij_cond_y):
    mi = 0
    y_size, i_size, j_size = pxij_y.shape
    for i in range(i_size):
        for j in range(j_size):
            for y in range(y_size):
                # mi += (1.0 * pxij_y[y][i][j] * p_y[y] * log_2( pxij_y[y][i][j] / (pxi_y[y][i] * pxj_y[y][j])))
                mi += (1.0 * pxij_y[y][i][j]  * log_2(pxij_cond_y[y][i][j] / (pxi_y[y][i] * pxj_y[y][j])))
    return mi

def compute_mi_graph(Xy, n_features, attr, attr_types_ranges, out_labels):
    n_examples = Xy.size
    n_bayes_params = train_naive_bayes(Xy, attr, attr_types_ranges, out_labels)
    p_y = []
    p_y.append(n_bayes_params[attr[-1]][0][0])
    p_y.append(n_bayes_params[attr[-1]][1][1])
    p_xi_xj_cond_y = {}
    p_xi_xj_y = {}                  # Joint conditional Probabilities
    freq_dict = {}                  # count of xi , xj | y
    for i in range(n_features - 1):
        i_size = len(attr_types_ranges[i][-1])
        for j in range(n_features - i - 1):
            j_s = i + j + 1
            j_size = len(attr_types_ranges[j_s][-1])
            freq_dict[(attr[i], attr[j_s])] = np.zeros((2, i_size, j_size), dtype=np.int)   # axis 0 -> y        axis 1 -> xi     axis 2 -> xj
            p_xi_xj_y[(attr[i], attr[j_s])] = np.zeros((2, i_size, j_size), dtype=np.float)
            p_xi_xj_cond_y[(attr[i], attr[j_s])] = np.zeros((2, i_size, j_size), dtype=np.float)
            for k in range(n_examples):
                # print("ijk : ", i, j_s , k)
                # i_index = get_attr_val_idx(i, Xy[k][i].decode('UTF-8'), attr_types_ranges)
                # j_index = get_attr_val_idx(i+j+1, Xy[k][j_s].decode('UTF-8'), attr_types_ranges)
                # y_index = get_attr_val_idx(n_features, Xy[k][n_features].decode('UTF-8'), attr_types_ranges)
                i_index = get_attr_val_idx(i, Xy[k][i], attr_types_ranges)
                j_index = get_attr_val_idx(i+j+1, Xy[k][j_s], attr_types_ranges)
                y_index = get_attr_val_idx(n_features, Xy[k][n_features], attr_types_ranges)
                freq_dict[(attr[i], attr[j_s])][y_index][i_index][j_index] += 1

    for key in freq_dict.keys():
        joint_count = freq_dict[key]
        for y in range(2):
            #TODO : Changing p(xi, xj|y) to p(xi, xj, y). Laplace estimate makes a difference in mutual information computation
            total_count_cond = np.sum(freq_dict[key][y])
            total_count = n_examples
            # p_xi_xj_y[key][y,:,:] = (joint_count[y] + 1) / (total_count * 1.0 + joint_count[y].size)
            # p_xi_xj_cond_y[key][y,:,:] = (joint_count[y] + 1) / (total_count_cond * 1.0 + joint_count[y].size)
            p_xi_xj_y[key][y,:,:] = (joint_count[y] + 1) / (total_count * 1.0 + joint_count.size)
            p_xi_xj_cond_y[key][y,:,:] = (joint_count[y] + 1) / (total_count_cond * 1.0 + joint_count[y].size)

    mi_graph = np.zeros((n_features, n_features), dtype=np.float)
    for i in range(n_features - 1):
        for j in range(i+1, n_features):
            mi_graph[i][j] = compute_mi(n_bayes_params[attr[i]], n_bayes_params[attr[j]], p_y, p_xi_xj_y[(attr[i], attr[j])], p_xi_xj_cond_y[(attr[i], attr[j])])

    # print(freq_dict)
    # print(p_xi_xj_y)
    # print(mi_graph)

    return mi_graph, p_y

def get_max_edge(v_new, v_remaining, mi_graph):
    # print(v_new, v_remaining)
    max_mi = - float('inf')
    max_v_n = sorted(v_new)[0]
    max_v_r = sorted(v_remaining)[0]
    for v_n in sorted(v_new):
        for v_r in sorted(v_remaining):
            if v_n > v_r:
                x_index, y_index = v_r, v_n
            else:
                x_index, y_index = v_n, v_r
            if mi_graph[x_index][y_index] > max_mi:
                max_mi = mi_graph[x_index][y_index]
                max_v_n = v_n
                max_v_r = v_r

    return max_v_n, max_v_r


def find_max_st(mi_graph):
    count = 0
    max_st = np.zeros_like(mi_graph)
    n_features = max_st.shape[0]
    v = set(range(n_features))
    v_new = {0}
    v_remaining = v.copy()
    v_remaining.remove(0)
    while v_new != v:
        x, y = get_max_edge(v_new, v_remaining, mi_graph)
        assert x in v_new and y in v_remaining
        if x > y:
            x_new, y_new = y, x
            count += 1
        else:
            x_new, y_new = x, y
        max_st[x_new][y_new] = mi_graph[x_new][y_new]
        v_remaining.remove(y)
        v_new.add(y)

    # print(count)
    # print(max_st)
    return max_st

def get_childrens(current_childrens, v_new, max_st):
    new_childrens = []
    edges = []
    n_features = max_st.shape[0]
    for child in current_childrens:
        for i in range(n_features):
            if max_st[child, i] > 0:
                if i not in v_new:
                    new_childrens.append(i)
                    edges.append((child, i))
    return new_childrens, edges

def train_tan(Xy_train, attr, attr_types_ranges, out_labels):

    n_examples = Xy_train.shape[0]
    n_features = len(Xy_train[0]) - 1
    mi_graph, py = compute_mi_graph(Xy_train, n_features, attr, attr_types_ranges, out_labels)
    max_st = find_max_st(mi_graph)
    for i in range(n_features):
        for j in range(n_features):
            if i > j:
                max_st[i, j] = max_st[j, i]
    directed_mst = np.zeros_like(max_st)

    # Root is 0th feature
    current_childrens = [0]
    v_new = {0}
    v = set(range(n_features))
    # directed_mst[:1, :] = max_st[:1, :]
    while v_new != v:
        new_childrens, edges = get_childrens(current_childrens, v_new, max_st)
        for edge in edges:
            directed_mst[edge[0], edge[1]] = max_st[edge[0], edge[1]]
            v_new.add(edge[1])
        current_childrens = new_childrens

    bayes_net = np.zeros((n_features+1, n_features+1), dtype=np.float)
    bayes_net[0:n_features, 0:n_features] = directed_mst
    #Last row of bayes_net is y
    for i in range(n_features):
        bayes_net[-1, i] = 1

    # print(directed_mst)
    # print(bayes_net)

    parents = []
    parents_id = []
    parents.append((attr[0], "No_parent"))
    parents_id.append((0, -1))
    for j in range(n_features):
        for i in range(n_features):
            if directed_mst[i][j] > 0:
                parents.append((attr[j], attr[i]))
                parents_id.append((j, i))

    cpd_freq = {}
    cpd = {}
    for idx, feature in enumerate(attr):
        if idx == n_features:
            continue
        assert idx == parents_id[idx][0]
        if idx == 0:
            n_parent_vals = 1
        else:
            parent_index = parents_id[idx][-1]
            n_parent_vals = len(attr_types_ranges[parent_index][-1])
        cpd_freq[feature] = np.zeros((2, n_parent_vals, len(attr_types_ranges[idx][-1])), dtype=int)
        cpd[feature] = np.zeros((2, n_parent_vals, len(attr_types_ranges[idx][-1])), dtype=float)
        for k in range(n_examples):
            # print("feature, k : ", idx, k)
            # node = get_attr_val_idx(idx, Xy_train[k][idx].decode('UTF-8'), attr_types_ranges)
            node = get_attr_val_idx(idx, Xy_train[k][idx], attr_types_ranges)
            if idx == 0:
                parent = 0
            else:
                # parent = get_attr_val_idx(parent_index, Xy_train[k][parent_index].decode('UTF-8'), attr_types_ranges)
                parent = get_attr_val_idx(parent_index, Xy_train[k][parent_index], attr_types_ranges)
            # y_index = get_attr_val_idx(n_features, Xy_train[k][n_features].decode('UTF-8'), attr_types_ranges)
            y_index = get_attr_val_idx(n_features, Xy_train[k][n_features], attr_types_ranges)
            cpd_freq[feature][y_index, parent, node] += 1

    for idx, key in enumerate(attr):
        if idx == n_features:
            continue
        if idx == 0:
            n_parent_vals = 1
        else:
            parent_idx = parents_id[idx][-1]
            n_parent_vals = len(attr_types_ranges[parent_idx][-1])
        joint_count = cpd_freq[key]
        for y in range(2):
            for p in range(n_parent_vals):
                total_count = np.sum(joint_count[y, p])
                cpd[key][y, p, :] = (joint_count[y, p, :] + 1) / (total_count * 1.0 + joint_count[y, p, :].size)

    tan = Tan(py, cpd, parents, parents_id)

    return tan



def data_loader(file_name):
    '''

    :param file_name:
    :return: X and Y matrix, attributes, attribute types and ranges
    '''

    file = open(file_name, 'r')
    data_set, meta_data = arff.loadarff(file)
    num_examples = len(data_set)
    num_features = len(data_set[1]) - 1

    data_attr = meta_data.names()  # Returns a list
    #data_attr_types = meta_data.types()  # Returns a list . data_attr_types_ranges already has type information
    data_attr_types_ranges = []
    for attr in data_attr:
        data_attr_types_ranges.append(meta_data.__getitem__(attr))

    output_labels = meta_data.__getitem__(data_attr[-1])
    # neg_label = output_labels[-1][0]
    # pos_label = output_labels[-1][1]
    # print(neg_label, pos_label)

    return data_set, data_attr, data_attr_types_ranges, output_labels

def remove_quotes(target_string):
    dest_string = target_string
    if dest_string[0] == "'":
        dest_string = dest_string[1:]
    if dest_string[-1] == "'":
        dest_string = dest_string[:-1]
    return  dest_string

# def part2():
#     Xy, attr, attr_types_ranges, out_labels = data_loader("chess-KingRookVKingPawn.arff")
#     y_list = []
#     for example in Xy:
#         if example[-1] == attr_types_ranges[-1][-1][0]:
#             y_list.append(0)
#         else:
#             y_list.append(1)
#     y = np.array(y_list)
#     nb_accuracy = []
#     tan_accuracy = []
#     skf = StratifiedKFold(n_splits=10, shuffle=True)
#     for train_index, test_index in skf.split(Xy, y):
#         #print(train_index, test_index)
#         X_train, X_test = Xy[train_index], Xy[test_index]
#         Y_train, Y_test = y[train_index], y[test_index]
#         naive_bayes_params = train_naive_bayes(X_train, attr, attr_types_ranges, out_labels)
#         predicted_probs, predicted_labels, n_correct = predict_n_bayes(naive_bayes_params, X_test, attr, attr_types_ranges, out_labels)
#         accuracy = (n_correct * 1.0)/X_test.size
#         nb_accuracy.append(accuracy)
#         tan = train_tan(X_train, attr, attr_types_ranges, out_labels)
#         predicted_probs, predicted_labels, n_correct = tan.predict(X_test, attr, attr_types_ranges, out_labels)
#         accuracy = (n_correct * 1.0)/X_test.size
#         tan_accuracy.append(accuracy)
#     print("naive bayes:", nb_accuracy)
#     print("tan bayes:", tan_accuracy)
#     t, p = stats.ttest_ind(nb_accuracy, tan_accuracy, equal_var=False)
#     print("t, p",t,p)
#     pdb.set_trace()
#     print("done>>>>>>>>>>>>>>>>>>>>>>>")


def main():
    '''
    Parses the arguments and call appropriate functions
    # How to call the program from commandline - bayes <lymph_train.arff> <lymph_test.arff> <n/t>    # n/t - naive bayes or tree augmented naive bayes
    # example - bayes lymph_train.arff lymph_test.arff n
    '''
    # #TODO: COMMENT THE PART 2 CODE
    # part2()
    ##PROCESS THE ARGURMENT
    args = sys.argv
    # print(args)
    num_args = len(args)
    # print(num_args)

    if (num_args < 4):
        print("Wrong Usage - Script takes 4 arguments")
        print("Example Usage- python bayes.py lymph_train.arff lymph_test.arff n")
        exit(0)

    train_filename = args[1]
    test_filename = args[2]
    algo = args[3]
    #print(train_filename, test_filename, alog)

    ## LOAD THE DATA
    Xy_train, attr_train, attr_types_ranges_train, out_labels_train = data_loader(train_filename)
    Xy_test, attr_test, attr_types_ranges_test, out_labels_test = data_loader(test_filename)
    # for ex in Xy_train:
    #     print(ex)

    if algo == 'n':              #naive bayes
        naive_bayes_params = train_naive_bayes(Xy_train, attr_train, attr_types_ranges_train, out_labels_train)
        predicted_probs, predicted_labels, n_correct = predict_n_bayes(naive_bayes_params, Xy_test, attr_test, attr_types_ranges_train, out_labels_test)
        #Print the results
        for attr in attr_test:
            if attr == attr_test[-1]:
                continue
            else:
                print("{} {}".format(attr, attr_test[-1]))
        print("")
        for idx, example in enumerate(Xy_test):
            # print("{:s} {:s} {:.12f}".format(predicted_labels[idx], Xy_test[idx][-1].decode('UTF-8'), predicted_probs[idx]))
            print("{:s} {:s} {:.12f}".format(remove_quotes(predicted_labels[idx]), remove_quotes(Xy_test[idx][-1]), predicted_probs[idx]))
        print("")
        print("{}".format(n_correct))
    elif algo == 't':           # Tree augmented naive bayes
        tan = train_tan(Xy_train, attr_train, attr_types_ranges_train, out_labels_train)
        predicted_probs, predicted_labels, n_correct = tan.predict(Xy_test, attr_test, attr_types_ranges_train, out_labels_test)
        # Print the results
        for idx, parent_pair in enumerate(tan.parents):
            if idx == 0:
                print("{} {}".format(parent_pair[0], attr_test[-1]))
            else:
                print("{} {} {}".format(parent_pair[0], parent_pair[-1], attr_test[-1]))
        print("")
        for idx, example in enumerate(Xy_test):
            # print("{:s} {:s} {:.12f}".format(predicted_labels[idx], Xy_test[idx][-1].decode('UTF-8'), predicted_probs[idx]))
            print("{:s} {:s} {:.12f}".format(remove_quotes(predicted_labels[idx]), remove_quotes(Xy_test[idx][-1]), predicted_probs[idx]))
        print("")
        print("{}".format(n_correct))
    else:
        print('choose n for naive bayes, t for TAN')



if __name__ == "__main__":
    main()
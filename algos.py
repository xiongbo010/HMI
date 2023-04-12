#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Distributed under terms of the MIT license.
# Copyright 2021 Chao Pan.

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, confusion_matrix, f1_score
from numpy.linalg import norm
from sklearn.svm import LinearSVC
from GrahamScan import GrahamScan
from platt import *
# from hsvm import *
from collections import Counter
from sklearn.preprocessing import label_binarize
from scipy.special import softmax

"""
This file contains all functions for hyperbolic perceptrons and SVM. A random data generator is included (Poincare_Uniform_Data) for synthetic experiments. Two methods (ConvexHull and QuickHull) are included to learn the reference point for tangent space.

For perceptron algorithms, we implement
1. Our hyperbolic perceptron: HP
2. Euclidean perceptron: EP
3. Hyperbolic perceptron from Weber et al. 2020: WeberHP

For SVM algorithms, we implement
1. Our hyperbolic SVM: tangent_hsvm
2. SVM from Cho et al. 2019: cho_hsvm
3. Euclidean SVM: euclidean_svm, based on sklearn.svm.LinearSVC
"""

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
          'tab:olive']


def Parrallel_transport(v,p):
    """
    Parrallel transport of v in T_0 to T_p.
    """
    return (1-np.linalg.norm(p)**2)*v


def Mobius_add(x,y,c=1.):
    """
    Mobius addition x \oplus y.
    c is the negative curvature, which we set default as 1.
    """
    DeNom = 1+2*c*np.dot(x,y)+c**2*np.dot(x,x)*np.dot(y,y)
    Nom = (1+2*c*np.dot(x,y)+c*np.dot(y,y))*x + (1-c*np.dot(x,x))*y
    return Nom/DeNom


def Exp_map(v,p,c=1.):
    """
    Exp map. v is the tangent vector in T_p and c is the negative curvature.
    """
    lbda = 2/(1-c*np.dot(p,p))
    temp = np.tanh(np.sqrt(c)*lbda*np.sqrt(np.dot(v,v))/2)*v/np.sqrt(c)/np.sqrt(np.dot(v,v))
    return Mobius_add(p,temp,c)


def Log_map(x,p,c=1.):
    """
    Log map. x is the vector in hyperbolic space, p is the reference point and c is the negative curvature.
    """
    lbda = 2/(1-c*np.dot(p,p))
    temp = Mobius_add(-p,x,c)
    return 2/np.sqrt(c)/lbda*np.arctanh(np.sqrt(c)*np.sqrt(np.dot(temp,temp)))*temp/np.sqrt(np.dot(temp,temp))


def poincare_dist(x, y):
    """
    Poincare distance of two hyperbolic points x,y.
    """
    return np.arccosh(1 + 2*(norm(x-y)**2)/(1-norm(x)**2)/(1-norm(y)**2))


def point_on_geodesic(x, y, t):
    """
    This gives the point on the geodesic x -> y. t is the time, where t=0 gives x and t=1 gives y.
    """
    return Exp_map(t*Log_map(y, x), x)


def Poincare_Uniform_Data(N,d,gamma,R = 0.9, p = None, w = None):
    """
    Generate points uniformly (in Euclidean sense) on the Poincare ball.
    N: Number of points.
    d: Dimension.
    gamma: Margin.
    R: Upper bound of the radius of points.
    p: Reference point. If not given, we randomly generate if within the ball of radius R.
    w: Normal vector of ground truth linear classifier. If not given, we generate it uniformly at random on the ball of radius R.
    """

    # Points within uniform ball
    X = np.random.randn(d,N)
    r = np.random.rand(N)**(1/d)
    c = r/np.sqrt(np.sum(X**2,0))
    X = X*c*R
    
    # Construct true classifier
    if p is None:
        p = np.random.randn(d)
        p = p/np.linalg.norm(p)*np.random.rand(1)*R
    
    if w is None:
        w = np.random.randn(d)
        
    w = Parrallel_transport(w,p)
    w = w/np.linalg.norm(w)
    
    # Define true labels
    y = np.zeros(N)
    for n in range(N):
        if np.dot(w,Log_map(X[:,n],p))> 0:
            y[n] = 1
            
        else:
            y[n] = -1
    
    # Remove points which invalidates margin assumption
    lmda_p = 2/(1-np.linalg.norm(p)**2)
    for n in range(N):
        xn = X[:,n]
        vn = Log_map(xn,p)
        if np.dot(w,vn)*y[n] < np.sinh(gamma)*np.linalg.norm(vn)*(1-np.tanh(lmda_p*np.linalg.norm(vn)/2)**2)/np.tanh(lmda_p*np.linalg.norm(vn)/2)/2:
            y[n] = 0
            X[:,n] = 0
            
    idx = np.argwhere(np.all(X[..., :] == 0, axis=0))
    X = np.delete(X, idx, axis=1)
    y = np.delete(y, idx)
    N = len(y)
    
    return N,X,y,w,p


def HP(X,y,p,gamma=0.01,R=0.9,option='first',a=None):
    """
    Our proposed hyperbolic perceptron.
    X: Data matrix of size (N,d), where N is the number of points and d is the dimension.
    y: Label vector of size (N,). Labels should be {1,-1}.
    p: Reference point.
    gamma: Margin.
    R: Upper bound of the radius of points.
    option: 'first' for Hyperbolic perceptron and 'second' for second order hyperbolic perceptron.
    a: The hyperparameter of second order hyperbolic perceptron.
    """
    assert option in ['first','second']
    
    d,N = np.shape(X)
    w0 = np.zeros(d)
    lmda_p = 2/(1-np.linalg.norm(p)**2)
    Rp = (np.linalg.norm(p)+R)/(1+np.linalg.norm(p)*R);
    Gamma = gamma/2;
    MaxIter = ((2*Rp)/(1-Rp**2)/np.sinh(Gamma))**2;
    itercount = 0
    
    
#     first order (normal) hyperbolic perceptron
    if option is 'first':
#         pbar = tqdm(total = MaxIter+1)
        while True:
            Flag = False
            for n in range(N):
                xn = X[:,n]
                vn = Log_map(xn,p)
#                 ipdb.set_trace()
                if np.dot(w0,vn)*y[n] <= 0:
                    etan = 2*np.tanh(lmda_p*np.linalg.norm(vn)/2)/np.linalg.norm(vn)/(1-np.tanh(lmda_p*np.linalg.norm(vn)/2)**2)
                    w0 = w0 + etan*vn*y[n]
                    itercount += 1
#                     pbar.update(1)
                    Flag = True
            
            if Flag is False:
#                 pbar.close()
                return w0/np.linalg.norm(w0),itercount
            elif itercount > MaxIter:
                print('Exceed MaxIter! Something wrong!')
#                 pbar.close()
                return w0/np.linalg.norm(w0),itercount
            
    if option is 'second':
        if a is None:
            print('Parameter a is undefined for second-order perceptron!')
            return None
        
        # Compute Z
        Z = np.zeros((d,N))
        for n in range(N):
            vn = Log_map(X[:,n],p)
            etan = 2*np.tanh(lmda_p*np.linalg.norm(vn)/2)/np.linalg.norm(vn)/(1-np.tanh(lmda_p*np.linalg.norm(vn)/2)**2)
            Z[:,n] = etan*vn
        
#         First round
        C = np.zeros((d,1))
        C[:,0] = Z[:,n]
        S = C
        xi = y[0]*Z[:,0]
        I = np.identity(d)
        while True:
            Flag = False
            for n in range(N):
                zn = np.zeros((d,1))
                zn[:,0] = Z[:,n]
                S = np.concatenate((C,zn),axis = 1)
                w = np.matmul(np.linalg.pinv(a*I+np.matmul(S,S.T)),xi)
                y_hat = np.sign(np.dot(w,Z[:,n]))
                if np.dot(w,Z[:,n])*y[n]<=0:
                    xi = xi + y[n]*Z[:,n]
                    C = S
                    Flag = True
                    itercount += 1
            
            if Flag is False:
                return xi, C, itercount
            
                    
def HCentroid(X):
    N = np.shape(X)[1]
    Weight = np.zeros(N)
    for n in range(N):
        Weight[n] = (1/(1-np.linalg.norm(X[:,n])**2))
        
    c = np.sum(X*Weight,axis = 1)/(np.sum(Weight)-N/2)
    c = np.tanh(0.5*np.arctanh(np.linalg.norm(c)))*c/np.linalg.norm(c)
    return c


def Weightedmidpt(C1,C2,t):
    """
    Compute the weighted midpoint from C1 to C2 in Poincare ball. t is the time where t=0 we get C1 and t=1 we get C2.
    """
    v = Mobius_add(-C1,C2)
    v = np.tanh(t*np.arctanh(np.linalg.norm(v)))*v/np.linalg.norm(v)
    return Mobius_add(C1,v)


def P2L(X):
    """
    Map points on Poincare ball to Loid' model.
    """
#     input: X is a d*n matrix
    d,n = np.shape(X)
    Z = np.zeros((d+1,n))
    for i in range(n):
        Z[0,i] = (1+np.sum(X[:,i]**2))/(1-np.sum(X[:,i]**2))
        Z[1:,i] = 2*X[:,i]/(1-np.sum(X[:,i]**2))
    return Z


def L2P(Z):
    """
    Map points on Loid' model to Poincare ball.
    """
    d,n = np.shape(Z)
    X = np.zeros((d-1,n))
    for i in range(n):
        X[:,i] = Z[1:,i]/(1+Z[0,i])
    return X


def ConvexHull(X):
    """
    Finding convexhull in Poincare disk using hyperbolic version of Graham scan.
    """
#     input: X is a d*n matrix
#     Assume d = 2 so far

#     Step1: Find the point with lowest y-cord
    Y_ax = X[1,:]
#     Check if multiple minimum
    idx = np.argwhere(Y_ax==np.amin(Y_ax))
    if len(idx)>1:
        X_ax = X[0,idx]
        idx_2 = np.argmin(X_ax)
        idx = idx[idx_2]
    idx = np.squeeze(idx)
    p0 = X[:,idx]
#     Sort points by inner angle with p0
    logX = np.zeros(np.shape(X))
    Iplist = np.zeros(np.shape(X)[1])
    for n in range(np.shape(X)[1]):
        logX[:,n] = Log_map(X[:,n],p0)
        Iplist[n] = np.dot(logX[:,n]/np.linalg.norm(logX[:,n]),[1,0])
#     Make sure p0 is sorted as the last point
    Iplist[idx] = -np.inf
#     Sort Iplist
    Ipidx = np.flip(np.argsort(Iplist))
    
    Points = X[:,Ipidx]
    
    d = np.shape(X)[0]
    N = np.shape(X)[1]
    Stack = np.zeros((d,N+1))
    end_idx = 0
    Stack[:,0] = p0

    for point in Points.T:
        while (end_idx>0) and (ccw(Stack[:,end_idx-1],Stack[:,end_idx],point)<0):
            end_idx -= 1
            
        end_idx += 1
        Stack[:,end_idx] = point
        
    return Stack[:,:(end_idx+1)]


def ccw(p0,p1,p2):
    """
    Outer product in hyperbolic Graham scan.
    """
    v01 = Log_map(p1,p0)
    v12 = Log_map(p2,p0)
    return v01[0]*v12[1]-v01[1]*v12[0]


def plotgeodesic(p0,p1=None,c='blue', v=None,l=None, option='segment'):

    assert option in ['segment','p2p_line','pv_line']
    if option is 'segment':
    #     default use 100 point
        t = np.linspace(0,1,100)
        output = np.zeros((2,100))
        for i in range(100):
            output[:,i] = Weightedmidpt(p0,p1,t[i])
    #     ipdb.set_trace()
        plt.plot(output[0,:],output[1,:],c=c,label=l)
        return None
    else:
#         Assume that 2 end points need to pass the circle R = 0.99
        R = 0.99
        if option is 'p2p_line':
            v = Mobius_add(-p0,p1)
            v = v/np.linalg.norm(v)
        
        v = np.array(v)
        t = np.linspace(0,1,100)
        Line = np.zeros((2,100))
        for n in range(100):
            Line[:,n] = Exp_map(v*t[n],p0)
        Line[:,0] = p0
        AdLine = np.zeros((2,100))
        count = 1.0
        while np.linalg.norm(Line[:,-1])<R:
            for n in range(100):
                AdLine[:,n] = Exp_map(v*(t[n]+count),p0)
            Line = np.append(Line,AdLine,axis= 1)
            count += 1.
        plt.plot(Line[0,:],Line[1,:],c=c,linewidth=3)
        
        Line = np.zeros((2,100))
        for n in range(100):
            Line[:,n] = Exp_map(-v*t[n],p0)
        Line[:,0] = p0
        count = 1.0
        while np.linalg.norm(Line[:,-1])<R:
            for n in range(100):
                AdLine[:,n] = Exp_map(-v*(t[n]+count),p0)
            Line = np.append(Line,AdLine,axis= 1)
            count += 1.
            
        plt.plot(Line[0,:],Line[1,:],c=c,label=l,linewidth=3)
        return None
        

def minDpair(CH1,CH2):
    """
    Finding minimum distance pair for convex hull CH1 and CH2 in Poincare disk.
    """
    N1 = np.shape(CH1)[1]
    N2 = np.shape(CH2)[1]
    cur_minD = np.inf
    output = np.zeros((2,2))
    for n1 in range(N1):
        for n2 in range(N2):
            dist = np.arccosh(1+2*(np.linalg.norm(CH1[:,n1]-CH2[:,n2])**2/((1-np.linalg.norm(CH1[:,n1])**2)*(1-np.linalg.norm(CH2[:,n2])**2))))
            if dist < cur_minD:
                output[:,0] = CH1[:,n1]
                output[:,1] = CH2[:,n2]
                cur_minD = dist
    return output


def QuickHull(X):
    """
    Poincare disk version of QuickHull algorithm.
    Assume the data matrix X is 2*N for in this function.
    """
#     Assume X is 2*n
    A = X[:,np.argmin(X[0,:])]
    B = X[:,np.argmax(X[0,:])]
    p = Weightedmidpt(A,B,0.5)
    v = Log_map(B,p)
    w = [v[1],-v[0]]
    X_log = np.copy(X)
    for n in range(np.shape(X)[1]):
        X_log[:,n] = Log_map(X[:,n],p)
    
    S1 = X[:,np.dot(X_log.T,w)>0]
    S2 = X[:,np.dot(X_log.T,w)<0]
#     ipdb.set_trace()
    Arr1 = FindHull(S1,A,B)
    Arr2 = FindHull(S2,B,A)
    
    Output = np.append(Arr1.reshape((2,-1)),Arr2.reshape((2,-1)),axis=1)
#     ipdb.set_trace()
    Output = np.append(Output,A.reshape((2,-1)),axis=1)
    Output = np.append(Output,B.reshape((2,-1)),axis=1)
    Output = np.unique(Output.T,axis=0).T
    
    return Output


def FindHull(Sk,P,Q):
    if Sk.size is 0:
        return 0
    elif np.shape(Sk)[1] is 1:
        if (np.squeeze(Sk) is P) or (np.squeeze(Sk) is Q):
            return 0
        else:
            return Sk
    
    p = Weightedmidpt(P,Q,0.5)
    v = Log_map(Q,p)
    w = [v[1],-v[0]]
    Dist = np.zeros(np.shape(Sk)[1])
    
    for n in range(np.shape(Sk)[1]):
        temp = Mobius_add(-p,Sk[:,n])
        Dist[n] = np.arcsinh(2*np.dot(temp,w)/(1-np.linalg.norm(temp)**2)*np.linalg.norm(w))
    
    
    C = Sk[:,np.argmax(Dist)]
    
    PC_mid = Weightedmidpt(P,C,0.5)
    CQ_mid = Weightedmidpt(C,Q,0.5)
    
    v_PC = Log_map(C,PC_mid)
    w_PC = [v_PC[1],-v_PC[0]]
    v_CQ = Log_map(Q,CQ_mid)
    w_CQ = [v_CQ[1],-v_CQ[0]]
    
    Sk_log_PC = np.copy(Sk)
    Sk_log_CQ = np.copy(Sk)
    for n in range(np.shape(Sk)[1]):
        Sk_log_PC[:,n] = Log_map(Sk[:,n],PC_mid)
        Sk_log_CQ[:,n] = Log_map(Sk[:,n],CQ_mid)
        
    S1 = Sk[:,np.dot(Sk_log_PC.T,w_PC)>0]
    S2 = Sk[:,np.dot(Sk_log_CQ.T,w_CQ)>0]
    
    Arr1 = FindHull(S1,P,C)
    Arr2 = FindHull(S2,C,Q)
    
#     ipdb.set_trace()
    if np.shape(Arr1) and np.shape(Arr2):
        Output = np.append(Arr1.reshape((2,-1)),Arr2.reshape((2,-1)),axis=1)
        Output = np.append(Output,C.reshape((2,-1)),axis=1)
        
    elif np.shape(Arr1):
        Output = Arr1.reshape((2,-1))
        Output = np.append(Output,C.reshape((2,-1)),axis=1)
    elif np.shape(Arr2):
        Output = Arr2.reshape((2,-1))
        Output = np.append(Output,C.reshape((2,-1)),axis=1)
    else:
        Output = C.reshape((2,-1))
    return Output


def Eval(X,y,p=None,w1=None,xi=None,C=None,a=0):
    """
    Evaluate the performance of resulting classifier.
    X: Data matrix of size (N,d), where N is the number of points and d is the dimension.
    y: Ground truth label vector of size (N,). Labels should be {1,-1}.
    p: Reference point.
    w1: Normal verctor of the resulting classifier

    The rest arguments are from the output of second order hyperbolic perceptron.
    """
    if p is not None:
        d = np.shape(X)[0]
        N = np.shape(X)[1]
        lmda_p = 2/(1-np.linalg.norm(p)**2)
        Z = np.zeros((d,N))
        I = np.identity(d)
        for n in range(N):
            vn = Log_map(X[:,n],p)
            etan = 2*np.tanh(lmda_p*np.linalg.norm(vn)/2)/np.linalg.norm(vn)/(1-np.tanh(lmda_p*np.linalg.norm(vn)/2)**2)
            Z[:,n] = etan*vn

        y_hat1 = np.zeros(N)
        y_hat2 = np.zeros(N)
        decision_val = np.zeros(N)
        for n in range(N):
            if (w1 is not None):
                y_hat1[n] = np.sign(np.dot(w1,Log_map(X[:,n],p)))
                decision_val[n] = np.dot(w1,Log_map(X[:,n],p))
            if (C is not None):
                zn = np.zeros((d,1))
                zn[:,0] = Z[:,n]
                S = np.concatenate((C,zn),axis = 1)
                w = np.matmul(np.linalg.pinv(a*I+np.matmul(S,S.T)),xi)
                y_hat2[n] = np.sign(np.dot(w,Z[:,n]))


        if (w1 is not None):
            if (C is not None):
                return np.sum(y*y_hat1>0)/N*100,np.sum(y*y_hat2>0)/N*100
            else:
                return np.sum(y*y_hat1>0)/N*100,decision_val
        else:
            if (C is not None):
                return np.sum(y*y_hat2>0)/N*100
    else:
        d = np.shape(X)[0]
        N = np.shape(X)[1]

        y_hat = np.zeros(N)
        decision_val = np.zeros(N)
        for n in range(N):
            y_hat[n] = np.sign(np.dot(w1,X[:,n]))
            decision_val[n] = np.dot(w1,X[:,n])

        return np.sum(y*y_hat>0)/N*100,decision_val


def WeberHP(X,y,Max_pass = 5):
    """
    Hyperbolic perceptron from Weber et al. 2020.
    X: Data matrix
    y: Label vector
    Max_pass: Maximum number of pass through all data. This prevents the algorithm never converge.

    Output:
    w: resulting classifier
    Mistake: number of updates (mistakes).
    Flag: Converge or not.
    """

#     Conver points in P to L
    X = P2L(X)
#     Initialize w, d here is the dimension of the ambient space!
    d,n = np.shape(X)
    w = np.zeros(d)
    w[1] = 1
#     h = diag(H) is for minkowski inner product
    h = np.ones(d)
    h[0] = -1
    pRound = 0
    Flag = False
    Mistake = 0
    while not Flag:
        Flag = True
        for i in range(n):
            if y[i]*np.sign(-np.dot(X[:,i],h*w)) <= 0:
                v = w + y[i]*X[:,i]
                w = v/np.minimum(1,np.sqrt(np.dot(v,h*v)))
                Mistake += 1
                Flag = False
        if pRound > Max_pass:
            return w,Mistake,Flag
        
        pRound += 1
        
    return w,Mistake,Flag


def EP(X,y,Max_pass = 5):
    """
    Euclidean Perceptron
    X: Data matrix
    y: Label vector
    Max_pass: Maximum number of pass through all data. This prevents the algorithm never converge.

    Output:
    w: resulting classifier
    Mistake: number of updates (mistakes).
    Flag: Converge or not.
    """
    d,n = np.shape(X)
    w = np.zeros(d)
    pRound = 0
    Flag = False
    Mistake = 0
    while not Flag:
        Flag = True
        for i in range(n):
            if y[i]*np.sign(-np.dot(X[:,i],w)) <= 0:
                w = w + y[i]*X[:,i]
                Mistake += 1
                Flag = False
        if pRound > Max_pass:
            return w,Mistake,Flag
        
        pRound += 1
        
    return w,Mistake,Flag


def zero_based_labels(y):
    labels = list(np.unique(y))
    new_y = [labels.index(y_val) for y_val in y]
    return np.array(new_y)


def plot_geodesic_new(p0, v, ax, c):
    R = 0.999
    t = np.linspace(0, 1, 100)
    # pos
    Line = np.zeros((2, 100))
    for n in range(1, 100):
        Line[:, n] = Exp_map(v * t[n], p0)
    Line[:, 0] = p0
    AdLine = np.zeros((2, 100))
    count = 1.0
    while np.linalg.norm(Line[:, -1]) < R:
        for n in range(100):
            AdLine[:, n] = Exp_map(v * (t[n] + count), p0)
        Line = np.append(Line, AdLine, axis=1)
        count += 1.
    ax.plot(Line[0, :], Line[1, :], c=c)
    # neg
    Line = np.zeros((2, 100))
    for n in range(1, 100):
        Line[:, n] = Exp_map(-v * t[n], p0)
    Line[:, 0] = p0
    count = 1.0
    while np.linalg.norm(Line[:, -1]) < R:
        for n in range(100):
            AdLine[:, n] = Exp_map(-v * (t[n] + count), p0)
        Line = np.append(Line, AdLine, axis=1)
        count += 1.
    ax.plot(Line[0, :], Line[1, :], c=c)


def tangent_hsvm(X_train, train_labels, X_test, test_labels, C=1000, X=None, labels = None, p=None, multiclass=False, saveplot=False):
    # the labels need to be 0-based indexed
    if saveplot:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        
    start = time.time()
    n_classes = train_labels.max() + 1
    n_train_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    if multiclass:
        # there is more than 2 classes, using ovr strategy
        # find optimal p for each ovr classifier
        test_probability = np.zeros((n_test_samples, n_classes), dtype=float)
        for class_label in range(n_classes):
            pos_coords = []
            neg_coords = []
            binarized_labels = []
            for i in range(n_train_samples):
                if train_labels[i] == class_label:
                    pos_coords.append((X_train[i][0], X_train[i][1]))
                    binarized_labels.append(1)
                else:
                    neg_coords.append((X_train[i][0], X_train[i][1]))
                    binarized_labels.append(-1)
            if len(pos_coords) <= 1:
                # skip very small classes
                continue
            binarized_labels = np.array(binarized_labels)
            # convex hull of positive cluster
            pos_hull = GrahamScan(pos_coords)
            neg_hull = GrahamScan(neg_coords)
            # get the reference point p by finding the min dis pair
            p = np.zeros(2)
            min_dis = float('inf')
            for i in range(pos_hull.shape[0]):
                for j in range(neg_hull.shape[0]):
                    if poincare_dist(pos_hull[i], neg_hull[j]) < min_dis:
                        min_dis = poincare_dist(pos_hull[i], neg_hull[j])
                        p = point_on_geodesic(pos_hull[i], neg_hull[j], 0.5)
            # map training data using log map
            X_train_log_map = np.zeros_like(X_train, dtype=float)
            for i in range(n_train_samples):
                X_train_log_map[i] = Log_map(X_train[i], p)
            linear_svm = LinearSVC(penalty='l2', loss='squared_hinge', C=C,max_iter=100000)
            linear_svm.fit(X_train_log_map, binarized_labels)
            w = linear_svm.coef_[0]
            decision_vals = np.array([np.dot(w, x) for x in X_train_log_map])
            ab = SigmoidTrain(deci=decision_vals, label=binarized_labels, prior1=None, prior0=None)
            # map testing data using log map
            for i in range(n_test_samples):
                x_test_log_map = Log_map(X_test[i], p)
                test_decision_val = np.dot(w, x_test_log_map)
                test_probability[i, class_label] = SigmoidPredict(deci=test_decision_val, AB=ab)

            if saveplot:
                # plot the geodesics
                # find two points near boundary
                v = np.array([-w[1], w[0]])
                v = v / norm(v)
#                 plot_geodesic_new(p, v, ax, colors[class_label])

        y_pred = np.argmax(test_probability, axis=1)
        matrix = confusion_matrix(test_labels, y_pred)
        acc_scores = matrix.diagonal() / matrix.sum(axis=1)
    else:
        # if there if only two classes, no need for Platt probability
        # if p is given, use the given p, else first estimate p
        if p is None:
            pos_coords = []
            neg_coords = []
            for i in range(n_train_samples):
                if train_labels[i] == 1:
                    pos_coords.append((X_train[i][0], X_train[i][1]))
                else:
                    neg_coords.append((X_train[i][0], X_train[i][1]))
            # convex hull of positive cluster
            pos_hull = GrahamScan(pos_coords)
            neg_hull = GrahamScan(neg_coords)
            # get the reference point p by finding the min dis pair
            p = np.zeros(2)
            min_dis = float('inf')
            for i in range(pos_hull.shape[0]):
                for j in range(neg_hull.shape[0]):
                    if poincare_dist(pos_hull[i], neg_hull[j]) < min_dis:
                        min_dis = poincare_dist(pos_hull[i], neg_hull[j])
                        p = point_on_geodesic(pos_hull[i], neg_hull[j], 0.5)
        # we have p now
        X_train_log_map = np.zeros_like(X_train, dtype=float)
        for i in range(n_train_samples):
            X_train_log_map[i] = Log_map(X_train[i], p)
        linear_svm = LinearSVC(penalty='l2', loss='squared_hinge', C=C, fit_intercept=False)
        linear_svm.fit(X_train_log_map, train_labels)
        X_test_log_map = np.zeros_like(X_test, dtype=float)
        for i in range(n_test_samples):
            X_test_log_map[i] = Log_map(X_test[i], p)
        y_pred = linear_svm.predict(X_test_log_map)
        matrix = confusion_matrix(test_labels, y_pred)
        acc_scores = matrix.diagonal() / matrix.sum(axis=1)
        w = linear_svm.coef_[0]
        if saveplot:
            v = np.array([-w[1], w[0]])
            v = v / norm(v)
#             plot_geodesic_new(p, v, ax, colors[0])

    if saveplot:
        # draw all data points
        circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=2, alpha=0.7)
        ax.add_patch(circ)
        for class_label in range(n_classes):
            ax.scatter(X[(y_pred == class_label), 0], X[(y_pred == class_label), 1], color=colors[class_label],
                       label='acc: ' + str(np.round(acc_scores[class_label], 3)),
                       alpha=0.5, edgecolors='black', linewidths=1, s=40)
        # plot legend and display
        ax.legend(loc='upper right', fontsize=10, shadow=True, edgecolor='black')
        plt.xlim([-1.1, 1.2])
        plt.ylim([-1.1, 1.2])
        plt.title('Poincare SVM. Overall Classification Accuracy: {}'.format(np.round(accuracy_score(test_labels, y_pred),decimals = 3)), size=10)
        # plt.show()
#         plt.savefig('tangent_hsvm_decision_boundaries.png')

    return f1_score(test_labels, y_pred, average='macro'), time.time() - start, w


def cho_hsvm(X_train, train_labels, X_test, test_labels, C=1000, X=None, labels=None, multiclass=False, saveplot=False, max_epoches=500):
    # fit multiclass hsvm and get prediction accuracy
    start = time.time()
    n_train_samples = X_train.shape[0]
    hsvm_clf = LinearHSVM(early_stopping=3, C=C, num_epochs=max_epoches, lr=0.001, verbose=False,
                          multiclass=multiclass, batch_size=int(n_train_samples/50))
    hsvm_clf.fit(poincare_pts_to_hyperboloid(X_train, metric='minkowski'), train_labels)
    y_pred = hsvm_clf.predict(poincare_pts_to_hyperboloid(X_test, metric='minkowski'))
    matrix = confusion_matrix(test_labels, y_pred)
    acc_scores = matrix.diagonal() / matrix.sum(axis=1)
    if saveplot:
        n_classes = train_labels.max()+1
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=2, alpha=0.7)
        ax.add_patch(circ)
        for class_label in range(n_classes):
            ax.scatter(X[(y_pred == class_label), 0], X[(y_pred == class_label), 1], color=colors[class_label],
                       label='acc: ' + str(np.round(acc_scores[class_label], 3)),
                       alpha=0.5, edgecolors='black', linewidths=1, s=40)
        # plot legend and display
        ax.legend(loc='upper right', fontsize=10, shadow=True, edgecolor='black')
        plt.xlim([-1.1, 1.2])
        plt.ylim([-1.1, 1.2])
        plt.title('Cho SVM. Overall Classification Accuracy: {}'.format(np.round(accuracy_score(test_labels, y_pred),decimals = 3)), size=10)
    return f1_score(test_labels, y_pred, average='macro'), time.time() - start


def euclidean_svm(X_train, train_labels, X_test, test_labels, C=1000, X=None, labels=None, saveplot=False):
    start = time.time()
    linear_svm = LinearSVC(penalty='l2', loss='squared_hinge', C=C,max_iter=100000)
    linear_svm.fit(X_train, train_labels)
    w = linear_svm.coef_[0]
    y_pred = linear_svm.predict(X_test)
    matrix = confusion_matrix(test_labels, y_pred)
    acc_scores = matrix.diagonal() / matrix.sum(axis=1)
    # print('Overall Classification Accuracy: {}'.format(accuracy_score(test_labels, y_pred)))
    # print('Time used:', time.time() - start)
    # only plot for Gaussian Mixture
    # return accuracy_score(test_labels, y_pred), time.time() - start
    if saveplot:
        # draw all data points
        n_classes = train_labels.max()+1
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=2, alpha=0.7)
        ax.add_patch(circ)
        for class_label in range(n_classes):
            ax.scatter(X[(y_pred == class_label), 0], X[(y_pred == class_label), 1], color=colors[class_label],
                       label='acc: ' + str(np.round(acc_scores[class_label], 3)),
                       alpha=0.5, edgecolors='black', linewidths=1, s=40)
        # plot legend and display
        ax.legend(loc='upper right', fontsize=10, shadow=True, edgecolor='black')
        plt.xlim([-1.1, 1.2])
        plt.ylim([-1.1, 1.2])
        plt.title('Euclidean SVM. Overall Classification Accuracy: {}'.format(np.round(accuracy_score(test_labels, y_pred),decimals = 3)), size=10)
        # plt.show()
#         plt.savefig('tangent_hsvm_decision_boundaries.png')
    return f1_score(test_labels, y_pred, average='macro'), time.time() - start, w


%% This file will demonstrate the usage of neural network via solving the following problem.
% 
% Neural network training: Develop a neural network (NN) for a two class classification
% problem. You can consider training vectors with two features each, provided in the accompanying mat
% file twoclass.mat. You need to implement a two-layer neural network (i.e., consisting of an input
% layer, a hidden layer, and an output layer) and the training algorithm. The input layer has three nodes
% corresponding to two feature variables and a bias; the hidden layer has Nh hidden nodes plus one bias;
% and the output layer has one node. You can assume the non-linear function at each node in the hidden
% layer and the output layer to be the sigmoid function
% (x) = h(x) = a exp(b x) ? exp(?b x)
% exp(b x) + exp(?b x) (1)
% The derivative of this function is specified by
% ?(x) = a b sech2(b x). (2)
% (a) Develop the forward propagation to compute the output, given the input features. Specifically,
% you need to consider weight matricesW1 andW2 of dimensions Nh×3 and 1×Nh+1, respectively.
% Note that each of the hidden nodes have three inputs, where two of them are the features and the
% third one is the bias. Similarly, the output node has Nh +1 inputs, including the bias. To realize
% a fast algorithm, design the propagation scheme such that the input X is a matrix of size 2×Nt,
% where Nt is the number of training data and the output Y is a vector of length Nt. Use matrix
% multiplications instead of for loop over the training vectors for efficiency. Assuming the weight
% matrices to be initialized with Gaussian random entries with standard deviation 0.01, compute
% the output vector.
%
% (b) Develop the backward propagation to compute the gradient of the weights. Specifically, compare
% the output vector to the desired output vector to derive the error. Compute the gradient and
% back propagate the gradients to compute the gradients of the weight matrices.
% (c) You can verify the gradient derived using backward propagation using finite difference methods.
% Specifically, change one entry of the matrix W1 by a small value and compute the new output.
% Compute the difference in the costs ky ? tk2 corresponding to the old W1 and the perturbed
% one. Here t is the desired output vector, which is +1 for class 1 and -1 for class 2. Divide the
% differences by the perturbations to get an approximation for the gradient. Repeat this procedure
% to compute the gradient of all entries of W1 and compare to the one computed using backward
% propagation. Observe that the finite difference approach is much slower than computing the
% gradient using back propagation.
% (d) Develop a steepest descent algorithm to optimize the weights to train the classifier. You can
% choose the step size as 0.001. Plot y and t at each iteration, while observing the criterion ky?tk2
% to determine if the cost is decreasing at each iteration.
% 
% (e) At every 10 iterations, observe the evolution of the decision boundaries corresponding to the
% neurons in the hidden layer. Specifically, create a scatter plot of the two classes. You may use
% the following script to create the scatter plot:
% plot(s2(:,1),s2(:,2),’r*’); hold on; plot(s1(:,1),s1(:,2),’*’);.
% Plot the decision boundaries corresponding to the hidden layers over the scatter plot. Note that
% each of the hidden nodes have three inputs (2 features and a bias), which are weighted by the
% corresponding entries of W1. The decision boundary corresponding that hidden neuron can be
% described by a straight line in the feature space.
% (f) At every 10 iterations, compute the discriminant function y(x) on a regular grid and display it as
% an image. Observe the relation between the discriminant function as a function of the decision
% boundaries plotted in the earlier figure.
% 
% data set, twoclass.mat
% 
% The neural network has configuration [2,Nh,1], 
% the output is a scalar: assign label 1 if > 0.5; assign label 0 if < 0.5
% 
% By JYI, 11/07/2018

clear all
close all
clc
%% data exploration
data = load('twoclass.mat');
c1 = data.s1; c2 = data.s2;
N1 = 100; N2 = 100; Ns = N1+N2;
lab1 = ones(N1,1); lab2 = - ones(N2,1); % lab 1 for class 1, label 0 for class 2

feat = [c1;c2]; lab = [lab1;lab2];

figure; hold on;
plot(c1(:,1),c1(:,2),'*'); plot(c2(:,1),c2(:,2),'o');
xlabel('Feature 1'); ylabel('Feature 2'); title('Dataset Samples');
legend('Class 1','Class 2')

%% parameters set up
rng(0)
Nh = 4; 
Nf = 2; 
nn_config.Nf = Nf;
nn_config.Nh = Nh;
nn_config.No = 1;
mu = 0.001;
Niter = 1000+1;
Nfig = fix(Niter/150);

%% initialization
X = [feat, ones(N1+N2,1)]; % (N1+N2,Nf+1)
W1 = 0.01*randn(Nh,Nf+1); % (Nh,Nf+1)
W2 = 0.01*randn(1,Nh+1); % (1,Nh+1)
iter = 1;
loss_arr = zeros(Niter-1,1);

figure; 
while iter < Niter
    
    %% forward propagation
    [aout_struct, wsum_struct, loss, aout2] = ForwardProp(X,lab,W1,W2,Ns);
    loss_arr(iter) = loss;
    
    %% back propagation
    [delta_struct,~] = BackProp(lab,W1,W2,aout_struct,wsum_struct,nn_config);

    %% system configuration updates
    [grad_W1,grad_W2] = GradCalc(aout_struct,delta_struct,nn_config,Ns);
    W1 = W1 - mu*grad_W1;
    W2 = W2 - mu*grad_W2;
    
    plot_flag = rem(iter,150);
    if plot_flag==0
        plot_ind = iter/150; 
        subplot(Nfig,1,plot_ind); hold on; 
        plot(lab,'-o'); plot(aout2,'-*'); legend('True label','Esti output');
        xlabel('Sample index'); ylabel('Label or Value');
    end
    
    plot_flag2 = rem(iter,10);
    
    
    iter = iter+1;
end

%% Results report
figure; 
plot(loss_arr,'-*');
title('Loss VS Iterations'); xlabel('Num of iterations'); ylabel('Loss');








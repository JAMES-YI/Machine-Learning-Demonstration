%% This file is a demonstration for solving the following problem.
% 
% Use the training data provided
% in the dataset svmdataset-hw, which can be downloaded from ICON to train a support vector machine
% classifier.
% (a) Implement the SVM using the MATLAB implementation of quadprog. See documentation of
% quadprog for details.
% i. You need to extend and normalize the data, setup the H matrix, A matrix, and b vector,
% depending on the penalty and constraints.
% ii. Plot the points, and the separating line. You may use the function ezplot to plot the line; see
% documentation for details. In a separate plot, plot the Lagrange multipliers that are provided
% by quadprog.
% iii. Mark the margin vectors by identifying the Lagrange multipliers that are non-zero (active
% constraints).
% (b) Use the MATLAB routine svmtrain to train a support vector machine for the same data above.
% Note that svmtrain (or fitcsvm) internally relies on quadratic programming for training. Read
% the documentation of svmtrain before you get started. You need to try different settings to
% understand the impact of these settings on the classification. Set showplot as true to visualize
% the results.
% i. Set ’boxconstraint’ to 0.1 and run the algorithm. Repeat the same with boxconstraint set
% to 1000. What differences do you observe & why ? When is the separating line provided by
% svmtrain similar to the one you derived using quadprog.
% ii. Change kernelfunction to ’quadratic’ and run the algorithm. What differences do you
% observe from the previous setting & why ?
% iii. Change ’kernelfunction’ to ’rbf’ and run the algorithm. What differences do you observe
% from the previous setting & why ?
% (c) In problem 2.(b), compute the equation for the separating hyper plane in terms of the support
% vectors. Refer to the course notes for the relation between the support vectors, Lagrange multipli-
% ers and the separating hyperplane. The support vectors and the Lagrange multipliers are stored
% in the structure returned by svmtrain (SupportVectors and Alpha, respectively).
% (d) Compute and display the discriminant function for each of the cases 2.b.i, 2.b.ii, and 2.b.iii. Refer
% to the course notes for the relation between support vectors, Lagrange multipliers, and the kernel
% functions (the handle to the kernel function is returned by svmtrain, which could be used to plot
% the discriminant). You can plot it using contour plot.
% 
% by JYI, 10/26/2018

%% data exploration
data = load('svmdataset-hw.mat');

% unnormalized
class1 = data.s1; % (100,2), 100 samples and 2 features
class2 = data.s2; % (100,2)
[c1_num,fea_num] = size(class1);
c2_num = size(class2,1);
x = [class1;class2]; % (200,2), feature data
y = [ones(c1_num,1); -ones(c2_num,1)]; % (200,1), label

% normalization
x_mean = mean(x);
x_std = std(x);
x(:,1) = (x(:,1) - x_mean(1)) / x_std(1);
x(:,2) = (x(:,2) - x_mean(2)) / x_std(2);
class1_n = x(1:100,:);
class2_n = x(101:200,:);
x_n = [class1_n;class2_n];

%% construct classifier 
% 0.1 boxconstraint
classifier1 = fitcsvm(x_n,y,'BoxConstraint',0.1);
w = classifier1.Beta;
b = classifier1.Bias;
x2 = @(x1) -(w(1)*x1+b)/w(2);
classifier_f1 = @(z) w(1)*z(1) + w(2)*z(2) + b;
figure; hold on;
fplot(x2);
plot(class1_n(:,1),class1_n(:,2),'*');
plot(class2_n(:,1),class2_n(:,2),'o');
legend('Separating plane','Class 1','Class 2');
title('Separating plane and data samples: 0.1');
xlabel('feature 1');
ylabel('feature 2');

xx1 = [-2:0.1:2];
xx2 = [-2:0.1:2];
[XX1,XX2] = meshgrid(xx1,xx2);
ZZ = w(1)*XX1 + w(2)*XX2 + b;
figure;
contour(XX1,XX2,ZZ);
title('Contour of discriminant function');
xlabel('feature 1');
ylabel('feature 2');


% 1000 boxconstrain
classifier2 = fitcsvm(x_n,y,'BoxConstraint',1000);
w = classifier2.Beta;
b = classifier2.Bias;
x2 = @(x1) -(w(1)*x1+b)/w(2);
classifier_f2 = @(z) w(1)*z(1) + w(2)*z(2) + b;
figure; hold on;
fplot(x2);
plot(class1_n(:,1),class1_n(:,2),'*');
plot(class2_n(:,1),class2_n(:,2),'o');
legend('Separating plane','Class 1','Class 2');
title('Separating plane and data samples: 1000');
xlabel('feature 1');
ylabel('feature 2');

xx1 = [-2:0.1:2];
xx2 = [-2:0.1:2];
[XX1,XX2] = meshgrid(xx1,xx2);
ZZ = w(1)*XX1 + w(2)*XX2 + b;
figure;
contour(XX1,XX2,ZZ);
title('Contour of discriminant function');
xlabel('feature 1');
ylabel('feature 2');

yx_n = y .* x_n;
% 0.1 boxconstraint, quadratic kernel
classifier_q1 = fitcsvm(x_n,y,...
                      'BoxConstraint',0.1,...
                      'KernelFunction','polynomial',...
                      'PolynomialOrder',2);
w = classifier_q1.Beta;
b = classifier_q1.Bias;
x2 = @(x1) -(w(1)*x1+b)/w(2);
figure; hold on;
fplot(x2);
plot(class1_n(:,1),class1_n(:,2),'*');
plot(class2_n(:,1),class2_n(:,2),'o');
legend('Separating plane','Class 1','Class 2');
title('Separating plane and data samples: 0.1, quadratic kernel');
xlabel('feature 1');
ylabel('feature 2');

% 1000 boxconstrain, quadratic kernel
classifier_q2 = fitcsvm(x_n,y,...
                      'BoxConstraint',1000,...
                      'KernelFunction','polynomial',...
                      'PolynomialOrder',2);
w = classifier_q2.Beta;
b = classifier_q2.Bias;
x2 = @(x1) -(w(1)*x1+b)/w(2);
figure; hold on;
fplot(x2);
plot(class1_n(:,1),class1_n(:,2),'*');
plot(class2_n(:,1),class2_n(:,2),'o');
legend('Separating plane','Class 1','Class 2');
title('Separating plane and data samples: 1000, quadratic kernel');
xlabel('feature 1');
ylabel('feature 2');

% 0.1 boxconstraint, gaussian kernel
classifier_g1 = fitcsvm(x_n,y,...
                      'BoxConstraint',0.1,...
                      'KernelFunction','gaussian');
w = classifier_g1.Beta;
b = classifier_g1.Bias;
x2 = @(x1) -(w(1)*x1+b)/w(2);
figure; hold on;
fplot(x2);
plot(class1_n(:,1),class1_n(:,2),'*');
plot(class2_n(:,1),class2_n(:,2),'o');
legend('Separating plane','Class 1','Class 2');
title('Separating plane and data samples: 0.1, gaussian kernel');
xlabel('feature 1');
ylabel('feature 2');

% 1000 boxconstrain, gaussian kernel
classifier_g2 = fitcsvm(x_n,y,...
                      'BoxConstraint',1000,...
                      'KernelFunction','gaussian');
w = classifier_g2.Beta;
b = classifier_g2.Bias;
x2 = @(x1) -(w(1)*x1+b)/w(2);
figure; hold on;
fplot(x2);
plot(class1_n(:,1),class1_n(:,2),'*');
plot(class2_n(:,1),class2_n(:,2),'o');
legend('Separating plane','Class 1','Class 2');
title('Separating plane and data samples: 1000, gaussian kernel');
xlabel('feature 1');
ylabel('feature 2');

%% use supporting vector to compute w and b
yx_n = y .* x_n;
% 0.1 boxconstraint, quadratic kernel
classifier_q1 = fitcsvm(x_n,y,...
                      'BoxConstraint',0.1,...
                      'KernelFunction','polynomial',...
                      'PolynomialOrder',2);
nonzero_ind = find(classifier_q1.IsSupportVector);
alp = classifier_q1.Alpha;
alp_yx_n = alp .* yx_n(nonzero_ind,:);
w = sum(alp_yx_n,1);
b = classifier_q1.Bias;
x2 = @(x1) -(w(1)*x1+b)/w(2);
figure; hold on;
fplot(x2);
plot(class1_n(:,1),class1_n(:,2),'*');
plot(class2_n(:,1),class2_n(:,2),'o');
legend('Separating plane','Class 1','Class 2');
title('Separating plane and data samples: 0.1, quadratic kernel');
xlabel('feature 1');
ylabel('feature 2');

% 1000 boxconstrain, quadratic kernel
classifier_q2 = fitcsvm(x_n,y,...
                      'BoxConstraint',1000,...
                      'KernelFunction','polynomial',...
                      'PolynomialOrder',2);
nonzero_ind = find(classifier_q2.IsSupportVector);
alp = classifier_q2.Alpha;
alp_yx_n = alp .* yx_n(nonzero_ind,:);
w = sum(alp_yx_n,1);
b = classifier_q2.Bias;
x2 = @(x1) -(w(1)*x1+b)/w(2);
figure; hold on;
fplot(x2);
plot(class1_n(:,1),class1_n(:,2),'*');
plot(class2_n(:,1),class2_n(:,2),'o');
legend('Separating plane','Class 1','Class 2');
title('Separating plane and data samples: 1000, quadratic kernel');
xlabel('feature 1');
ylabel('feature 2');



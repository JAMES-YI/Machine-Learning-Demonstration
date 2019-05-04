%% This file is a demonstration of solving the following problem.
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
% by JYI, 10/24/2018

%% data exploration
data = load('svmdataset-hw.mat');

% unnormalized
class1 = data.s1; % (100,2), 100 samples and 2 features
class2 = data.s2; % (100,2)
[c1_num,fea_num] = size(class1);
c2_num = size(class2,1);
x = [class1;class2]; % (200,2), feature data
y = [ones(c1_num,1); -ones(c2_num,1)]; % (200,1), label

figure; hold on;
xlabel('feature 1');ylabel('feature2');
plot(class1(:,1),class1(:,2),'*');
plot(class2(:,1),class2(:,2),'o');
legend('C1','C2','Location','southeast');

% normalized
% norm_c1 = sqrt(class1(:,1).^2 + class1(:,2).^2);
% class1_n = class1 ./ norm_c1;
% norm_c2 = sqrt(class2(:,1).^2 + class2(:,2).^2);
% class2_n = class2 ./ norm_c2;

x_mean = mean(x);
x_std = std(x);
x(:,1) = (x(:,1) - x_mean(1)) / x_std(1);
x(:,2) = (x(:,2) - x_mean(2)) / x_std(2);
class1_n = x(1:100,:);
class2_n = x(101:200,:);


figure; hold on;
xlabel('feature 1');ylabel('feature 2');
plot(class1_n(:,1),class1_n(:,2),'*');
plot(class2_n(:,1),class2_n(:,2),'o');

%% solve dual optimization problem of SVM via quadprog
% modeling
yx = y .* x; % (200,2)
H = (1/2) * (yx * yx');
H_diag = diag(H);
H = H + diag(H_diag);
clear H_diag
f = - ones(200,1);
A = - eye(200);
b = zeros(200,1);
Aeq = y';
beq = 0;

% solving optimization
[opt_var,obj_val,exitflag,output,lambda] = quadprog(H,f,A,b,Aeq,beq);

zero_ind = find(opt_var<10^(-10));
opt_var(zero_ind) = 0;

figure;
plot(opt_var,'*');
xlabel('element index');
ylabel('element value');
title('Distribution of optimal solution of dual problem of SVM');

%% classfier construction
% classifier
alp_x_y = opt_var .* yx;
w = sum(alp_x_y,1)'; % (2,1)
xt_w = x*w; % (200,1)
c1_min = min(xt_w(1:100));
c2_max = max(xt_w(101:200));
b = -(c1_min+c2_max)/2;
classifier = @(z) w*z+b;

% visualization: w1*x1 + w2*x2 + b=0 => x2 = -(w1*x1+b)/w2
x2 = @(x1) -(w(1)*x1+b)/w(2);

figure;hold on;
fplot(x2);
plot(class1_n(:,1),class1_n(:,2),'*');
plot(class2_n(:,1),class2_n(:,2),'o');
title('Data samples and separating plane');
xlabel('feature 1');ylabel('feature 2');
legend('Separating plan','Class 1','Class 2');

%% supporting vector
nonzero_ind = find(opt_var>10^(-10));
x_n = [class1_n;class2_n];
sup_vec = x_n(nonzero_ind,:);

figure;hold on;
fplot(x2);
plot(class1_n(:,1),class1_n(:,2),'*');
plot(class2_n(:,1),class2_n(:,2),'o');
plot(sup_vec(:,1),sup_vec(:,2),'<');
title('Data samples and separating plane');
xlabel('feature 1');ylabel('feature 2');
legend('Separating plan','Class 1','Class 2','Supporting vector');

%% This file is a demo for Problem 7.21 in [1], 
% i.e., design a Bayes classifier and a logistic classifier for two
% dimensional binary classification task.
% 
% [1] S. Theodoridis. Machine learning, a Bayesian and optimization
% prospective. Academic Press, 2015.
% 
% by JYI, 09/29/2018
clear all
close all
clc
%% data generation
rng(0)
N = 1500; 
Mu_1 = [0,2]; Sigma_1 = [4,1.8;1.8,1];
train_1 = mvnrnd(Mu_1,Sigma_1,N);
test_1 = mvnrnd(Mu_1,Sigma_1,N);

Mu_2 = [0,0]; Sigma_2 = [4,1.8;1.8,1];
train_2 = mvnrnd(Mu_2,Sigma_2,N);
test_2 = mvnrnd(Mu_2,Sigma_2,N);

figure; hold on;
plot(train_1(:,1),train_1(:,2),'*');
plot(train_2(:,1),train_2(:,2),'o');
plot(test_1(:,1),test_1(:,2),'*');
plot(test_2(:,1),test_2(:,2),'o');
xlabel('Feature 1'); ylabel('Feature 2');
legend('Train data (C1)','Train data (C2)','Test data (C1)','Test data (C2)');

%% compute mean and covariance from training data
est_mu_1 = mean(train_1);
est_mu_2 = mean(train_2);

cov_sum_1 = 0;
cov_sum_2 = 0;
for i = 1:N
    samp_temp_1 = train_1(i,:);
    cov_sum_1 = cov_sum_1 + samp_temp_1' * samp_temp_1;
    
    samp_temp_2 = train_2(i,:);
    cov_sum_2 = cov_sum_1 + samp_temp_2' * samp_temp_2;
end
est_sigma_1 = (1/N) * cov_sum_1;
est_sigma_2 = (1/N) * cov_sum_2;

clear samp_temp_1 samp_temp_2 cov_sum_1 cov_sum_2

%% construct classifier from training data
log_term = (1/2) * log( det(est_sigma_2) / det(est_sigma_1) ); 
Inv_est_sigma_1 = inv(est_sigma_1);
Inv_est_sigma_2 = inv(est_sigma_2);

classifier = @(x) log_term ...
                 + 1/2 * (x-est_mu_2) * Inv_est_sigma_2 * (x-est_mu_2)'...
                 - 1/2 * (x-est_mu_1) * Inv_est_sigma_1 * (x-est_mu_1)';
% if classifier(x) > 0, class 1; if classifier(x) < 0, class 2;

%% visualization of classification on test data
test_1_x1 = test_1(:,1);
test_1_x2 = test_1(:,2);

test_2_x1 = test_2(:,1);
test_2_x2 = test_2(:,2);

x1_min = min([min(test_1_x1),min(test_2_x1)]);
x1_max = max([max(test_1_x1),max(test_2_x1)]);

x2_min = min([min(test_1_x2),min(test_2_x2)]);
x2_max = max([max(test_1_x2),max(test_2_x2)]);

N_check = 30;
x1 = linspace(x1_min,x1_max,N_check);
x2 = linspace(x2_min,x2_max,N_check);
[X1,X2] = meshgrid(x1,x2);
Y = zeros(N_check,N_check);

for i = 1:N_check
    for j = 1:N_check
        samp_temp = [X1(i,j),X2(i,j)];
        Y(i,j) = classifier(samp_temp);
    end
end

figure; 
Y_dec = zeros(N_check,N_check);
surf(X1,X2,Y); hold on;
surf(X1,X2,Y_dec); % decision plane

% classifier acting on each sample from the test data
y_1 = [];
y_2 = [];
for i=1:N
    y_temp_1 = classifier( test_1(i,:) );
    y_1 = [y_1;y_temp_1];
    
    y_temp_2 = classifier( test_2(i,:) );
    y_2 = [y_2;y_temp_2];
end

scatter3(test_1_x1,test_1_x2,y_1);
scatter3(test_2_x1,test_2_x2,y_2);

xlabel('feature 1');
ylabel('feature 2');
zlabel('p(w1|x) - p(w2|x)');
legend('p(w1|x) - p(w2|x)','Decision Hyperplane','Class 1','Class 2');
colorbar
%% compute optimal theta in logistic classification

train_x = [train_1',train_2']; % (2,2N)
train_y = [ones(N,1);zeros(N,1)]; % (2N,1)

rng(0);
theta = randn(2,1) % intialization
step_size = 0.01;
MaxIte = 500;

for i=1:MaxIte
    weight_sum = theta'*train_x; % (1,2N)
    sigmoid = (1 ./ (1 + exp(-weight_sum)) )';
    search_dir = train_x * (sigmoid - train_y);
    theta = theta - step_size * search_dir;
end

prob_1 = @(x) 1 / (1+exp(-theta'*x)); % classifier: if > 0.5, class 1; if < 0.5, class 2;

test_x = [test_1',test_2'];
test_y = [];
NSample = 2*N;
for i=1:NSample
    test_y_temp = prob_1(test_x(:,i));
    test_y = [test_y;test_y_temp];
end

N_check = 30;
y_logistic_dec = 0.5*ones(N_check,N_check);
figure; 
surf(X1,X2,y_logistic_dec); hold on;
scatter3(test_1(:,1),test_1(:,2),test_y(1:N)); % class 1
scatter3(test_2(:,1),test_2(:,2),test_y(N+1:2*N)); % class 2

xlabel('feature 1');ylabel('feature 2');zlabel('Prob. in Class 1');
legend('Decision plane','Class 1','Class 2');

%% error rate computation in logistic classifier
test_y_1 = test_y(1:N); % if < 0.5, error classification;
test_y_2 = test_y(N+1:2*N); % if > 0.5, error classification; 

error_num = sum(test_y_1<0.5) + sum(test_y_2>0.5)
error_rate = error_num/NSample









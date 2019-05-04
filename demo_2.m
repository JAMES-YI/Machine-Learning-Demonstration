%% This file is to demonstrate how to solve
% 
% Consider a two-class fish classification problem, 
% where the goal is to determine if a given fish is salmon or sea-bass. 
% You are provided with two training datasets, 
% corresponding to the lengths of the fishes. 
% The first one is with 100 training samples each, while the second one is with 10000 samples.			
% (a) Estimate the likelihood distributions p(x|?_1) and p(x|?_2), assuming them to be Gaussian distributions
%     p(x??_1 )= N(?_1,?_1^2  )
%     p(x??_2 )= N(?_2,?_2^2  )
% 
% Specifically, determine the maximum likelihood estimates of the parameters of the likelihood distributions ?_1,?_2,?_1 and ?_2for each dataset.
% (b) With these estimates, plot the Bayes classifier.
% (c) Assuming the two classes are equiprobable (i.e., p(?_1 )= p(?_2) = ½), 
%     determine the minimum error classification rule in each case (100 samples and 10000 samples). 
%     Specifically, determine the decision regions R_1 and R_2.
% (d) Estimate the probability distributions using the histogram approach (use hist in MATLAB) for each of the datasets. 
%     Compute and plot the Bayes classifier and the minimum error classification rule in each case.
% (e) What are the pros and cons of the above probability estimation methods (histogram and Gaussian). 
%     When can you safely use each approach? Justify your anser.
% (f) data sets, fishdataset100.mat, fishdataset10000.mat
% by JYI, 09/27/2018
% bass: w1
% slamon: w2

%% data import and exploration
clear all
close all
clc

data100 = load('fishdataset100.mat');
len_b100 = data100.bass;
len_s100 = data100.salmon;

data10k = load('fishdataset10000.mat');
len_b10k = data10k.bass;
len_s10k = data10k.salmon;

figure
subplot(2,2,1)
plot(len_b100); title('bass in data100');
subplot(2,2,2)
plot(len_s100); title('salmon in data100');
subplot(2,2,3)
plot(len_b10k); title('bass in data10k');
subplot(2,2,4)
plot(len_s10k); title('salmon in data10k');

%% mean, variance computation under Gaussian assumption
% 100
mu100_1 = mean(len_b100);
diff_square = (len_b100 - mu100_1).^2;
sigma100_1 = sqrt( mean(diff_square) );
fprintf('In dataset_100, mu_1 in bass is: %d \n', mu100_1);
fprintf('In dataset_100, sigma_1 in bass is: %d \n', sigma100_1);

mu100_2 = mean(len_s100);
diff_square = (len_s100 - mu100_2).^2;
sigma100_2 = sqrt( mean(diff_square) );
fprintf('In dataset_100, mu_2 in salmon is: %d \n', mu100_2);
fprintf('In dataset_100, sigma_2 in salmon is: %d \n', sigma100_2);

% 10k
mu10k_1 = mean(len_b10k);
diff_square = (len_b10k - mu10k_1).^2;
sigma10k_1 = sqrt( mean(diff_square) );
fprintf('In dataset_10k, mu_1 in bass is: %d \n', mu10k_1);
fprintf('In dataset_10k, sigma_1 in bass is: %d \n', sigma10k_1);

mu10k_2 = mean(len_s10k);
diff_square = (len_s10k - mu10k_2).^2;
sigma10k_2 = sqrt( mean(diff_square) );
fprintf('In dataset_10k, mu_2 in salmon is: %d \n', mu10k_2);
fprintf('In dataset_10k, sigma_2 in salmon is: %d \n', sigma10k_2);

%% Bayes classifier under equiproble assumption, synthesis x (length)
% add more features and better visualization
x = 500:5:3000;

% 100, g(x) is classifier: > 0, class w1, bass; <0, class w2;
log_term = log(sigma100_2/sigma100_1);
normal_term_2 = 1/(2*sigma100_2^2) * (x-mu100_2).^2;
normal_term_1 = 1/(2*sigma100_1^2) * (x-mu100_1).^2;
g100_x = log_term + normal_term_2 - normal_term_1; % theoretical

% 100, g(x) is classifier: > 0, class w1; <0, class w2;
log_term = log(sigma100_2/sigma100_1);
normal_term_2 = 1/(2*sigma100_2^2) * (len_b100-mu100_2).^2;
normal_term_1 = 1/(2*sigma100_1^2) * (len_b100-mu100_1).^2;
g_b100 = log_term + normal_term_2 - normal_term_1; % true bass, w1

log_term = log(sigma100_2/sigma100_1);
normal_term_2 = 1/(2*sigma100_2^2) * (len_s100-mu100_2).^2;
normal_term_1 = 1/(2*sigma100_1^2) * (len_s100-mu100_1).^2;
g_s100 = log_term + normal_term_2 - normal_term_1; % true salmon, w2

figure;hold on;
plot(x, g100_x); plot(x, zeros(size(x))); % theoretical
plot(len_b100,g_b100,'b*'); % true data for bass
hold off
legend('Theoretical','Bass')
title('data100'); xlabel('length');

figure;hold on;
plot(x, g100_x); plot(x, zeros(size(x))); % theoretical
plot(len_s100,g_s100,'bo'); % true data for salmon
hold off
legend('Theoretical','Salmon')
title('data100'); xlabel('length');

% 10k, g(x) is classifier: >0, class w1; <0, class w2;
log_term = log(sigma10k_2/sigma10k_1);
normal_term_2 = 1/(2*sigma10k_2^2) * (x-mu10k_2).^2;
normal_term_1 = 1/(2*sigma10k_1^2) * (x-mu10k_1).^2;
g10k_x = log_term + normal_term_2 - normal_term_1; % theoretical

log_term = log(sigma10k_2/sigma10k_1);
normal_term_2 = 1/(2*sigma10k_2^2) * (len_b10k-mu10k_2).^2;
normal_term_1 = 1/(2*sigma10k_1^2) * (len_b10k-mu10k_1).^2;
g_b10k = log_term + normal_term_2 - normal_term_1;

log_term = log(sigma10k_2/sigma10k_1);
normal_term_2 = 1/(2*sigma10k_2^2) * (len_s10k-mu10k_2).^2;
normal_term_1 = 1/(2*sigma10k_1^2) * (len_s10k-mu10k_1).^2;
g_s10k = log_term + normal_term_2 - normal_term_1;

figure; hold on;
plot(x, g10k_x); plot(x, zeros(size(x))); % theoretical
plot(len_b10k,g_b10k,'b*') % true bass sample
legend('Theoretical','bass')
hold off
title('data10k'); xlabel('length');

figure; hold on;
plot(x,g10k_x); plot(x,zeros(size(x))); % theoretical
plot(len_s10k,g_s10k,'bo') % true salmon sample
legend('Theoretical','salmon')
hold off
title('data10k'); xlabel('length');

%% plot distributions for each class
% 100 samples
pdf_b100 = normpdf(len_b100,mu100_1,sigma100_1);
pdf_s100 = normpdf(len_s100,mu100_2,sigma100_2);
figure;hold on;
plot(len_b100,pdf_b100,'b*');
plot(len_s100,pdf_s100,'ro');
legend('bass','salmon'); title('data 100: distributions'); xlabel('length');
hold off;

% 10k samples
pdf_b10k = normpdf(len_b10k,mu10k_1,sigma10k_1);
pdf_s10k = normpdf(len_s10k,mu10k_2,sigma10k_2);
figure;hold on;
plot(len_b10k,pdf_b10k,'b*');
plot(len_s10k,pdf_s10k,'ro');
legend('bass','salmon'); title('data 10k: distributions'); xlabel('length');
hold off;

%% distribution estimate via histgram
% histogram for dataset with 100 samples, 100 bins
% for n=5:5:100
%     figure; h_b100 = histogram(len_b100,n);
%     p_b100 = h_b100.Values/numel(len_b100);
%     edge_b100 = h_b100.BinEdges;
%     figure; plot(edge_b100(2:end),p_b100,'bo')
% end
    
figure; h_b100 = histogram(len_b100,5);
p_b100 = h_b100.Values/numel(len_b100);
edge_b100 = h_b100.BinEdges;

figure; h_s100 = histogram(len_s100,5);
p_s100 = h_s100.Values/numel(len_s100);
edge_s100 = h_s100.BinEdges;

figure; hold on;
plot(edge_b100(2:end),p_b100,'bo');
plot(edge_s100(2:end),p_s100,'b*');
title('data 100: distribution from histogram');
legend('bass','salmon');
xlabel('length');

% histogram for dataset with 10k samples, 100 bins
figure; h_b10k = histogram(len_b10k,20); % Gaussian
p_b10k = h_b10k.Values/numel(len_b10k);
edge_b10k = h_b10k.BinEdges;

figure; h_s10k = histogram(len_s10k,20); % Gaussian
p_s10k = h_s10k.Values/numel(len_s10k);
edge_s10k = h_s10k.BinEdges;

figure; hold on;
plot(edge_b10k(2:end),p_b10k,'bo')
plot(edge_s10k(2:end),p_s10k,'b*');
title('data 10k: distributions from historgam');
legend('bass','salmon');
xlabel('length');

%% Bayes classifier based on histogram distribution estimate
% dataset with 100 samples

% dataset with 10k samples

%% minimal error classifier based on histogram distribution estimate
% dataset with 100 samples

% dataset with 10k samples





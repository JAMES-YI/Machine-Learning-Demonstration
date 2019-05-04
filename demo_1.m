%% This file is to 
% (1) demonstrate how regularization parameters will affect machine learning models 
% (both the case where a single data set is considered, and the case where multiply data set are considered); 
% demonstrate how the regularization parameters will affect the bias of estimation of machine learning models; 
% demonstrate the bias and variance tradeoff; demonstrate the MSE behavior with respect to regularization parameter; 
% demonstrate how the noise and the size of training data set will affect the performance of machine learning models
% By JYI on 09/17/2018, jirong-yi@uiowa.edu

rng(1);

%% data generation
N_samp = 100; % number of samples
N_feat = 1; % number of features
w_true = 0.04; 
y = w_true + normrnd(0,sqrt(0.1),[N_samp,N_feat]);

%% estimate of w_true under different lambdas
N_lamb = 50000;
lambda = linspace(0,500,N_lamb);
w_estimate = zeros(N_lamb,1);
for ite_lambda = 1:N_lamb
    temp_lambda = lambda(ite_lambda);
    w_estimate(ite_lambda) = sum(y)/(N_samp + temp_lambda);
end
w_error = (w_true - w_estimate).^2;

figure;
subplot(2,1,1)
plot(lambda,w_estimate);
xlabel('estimate index')
ylabel('estimate value')
title('Estimated value for w')

subplot(2,1,2)
plot(lambda,w_error);
xlabel('Error index')
ylabel('Error value')
title('Estimate error for w');

%% estimates of w_true under different dataset (as a function of lambda)
% MSE plot
rng(1)
N_lamb = 1000;
lambda = linspace(0,500,N_lamb);
N_dataset = 100;
N_samp = 1000;
N_feat = 1;
w_true = 0.04;
MSE_lamb = zeros(N_lamb,1);
for ite_lamb = 1:N_lamb
    lambda_temp = lambda(ite_lamb);
    error_scumsum = 0; % store the cummulative sum of the estimate over different datasets
    for ite_dataset = 1:N_dataset
        y = w_true + normrnd(0,sqrt(0.1),[N_samp,N_feat]);
        w_estimate = sum(y) / (N_samp + lambda_temp);
        error_scumsum = error_scumsum + (w_estimate - w_true)^2; 
    end
    error_mean = error_scumsum/N_dataset;
    MSE_lamb(ite_lamb) = error_mean;
end

figure 
plot(lambda,MSE_lamb);
xlabel('estimate index (or lambda value)')
ylabel('MSE(lambda)');
title('MSE vs lambda');

%% estimate of w_true under different dataset (as a function of lambda)
% squared bias plot
rng(1)
N_lamb = 1000;
lambda = linspace(0,500,N_lamb);
N_dataset = 100;
N_samp = 1000;
N_feat = 1;
w_true = 0.04;
MSE_lamb = zeros(N_lamb,1);
SBias_lamb = zeros(N_lamb,1);
for ite_lamb = 1:N_lamb
    lambda_temp = lambda(ite_lamb);
    error_scumsum = 0; % store the cummulative sum of the estimate over different datasets
    estimate_cumsum = 0;
    for ite_dataset = 1:N_dataset
        y = w_true + normrnd(0,sqrt(0.1),[N_samp,N_feat]);
        w_estimate = sum(y) / (N_samp + lambda_temp);
        error_scumsum = error_scumsum + (w_estimate - w_true)^2; 
        estimate_cumsum = estimate_cumsum + w_estimate;
    end
    error_mean = error_scumsum/N_dataset;
    estimate_mean = estimate_cumsum/N_dataset;
    MSE_lamb(ite_lamb) = error_mean;
    SBias_lamb(ite_lamb) = (w_true - estimate_mean)^2;
end

figure 
plot(lambda,MSE_lamb);
hold on
plot(lambda,SBias_lamb);
legend('MSE','Bias^2');

xlabel('estimate index (or lambda value)')
ylabel('MSE, Bias^2');
title('MSE, Bias^2 vs lambda');

%% bias and variance tradeoff with respect to regularization parameter
% variance plot
% rng(1)
N_lamb = 1000;
lambda = linspace(0,500,N_lamb);
N_dataset = 100;
N_samp = 1000;
N_feat = 1;
w_true = 0.04;
MSE_lamb = zeros(N_lamb,1);
SBias_lamb = zeros(N_lamb,1);
Var_lamb = zeros(N_lamb,1);
for ite_lamb = 1:N_lamb
    lambda_temp = lambda(ite_lamb);
    error_scumsum = 0; % store the cummulative sum of the estimate over different datasets
    estimate_cumsum = 0;
    w_estimate_holder = zeros(N_dataset,1);
    rng(1)
    for ite_dataset = 1:N_dataset
        y = w_true + normrnd(0,sqrt(0.1),[N_samp,N_feat]);
        w_estimate = sum(y) / (N_samp + lambda_temp);
        w_estimate_holder(ite_dataset) = w_estimate;
        error_scumsum = error_scumsum + (w_estimate - w_true)^2; 
        estimate_cumsum = estimate_cumsum + w_estimate;
    end
    error_mean = error_scumsum/N_dataset;
    estimate_mean = estimate_cumsum/N_dataset;
    MSE_lamb(ite_lamb) = error_mean;
    SBias_lamb(ite_lamb) = (w_true - estimate_mean)^2;
    Var_lamb(ite_lamb) = sum((w_estimate_holder - estimate_mean).^2)/N_dataset;
end

figure 
plot(lambda,MSE_lamb);
hold on
plot(lambda,SBias_lamb);
plot(lambda,Var_lamb);
legend('MSE','Bias^2','Var');

xlabel('estimate index (or lambda value)')
ylabel('MSE, Bias^2, Var');
title('MSE, Bias^2, Var vs lambda');

%% optimal lambda for MSE
SSigma = 0.1;
w_true = 0.04;
N_samp = 1000;

N_lamb = 1000;
lambda = linspace(0,500,N_lamb);
N_dataset = 100;
N_feat = 1;
MSE_lamb = zeros(N_lamb,1);
for ite_lamb = 1:N_lamb
    lambda_temp = lambda(ite_lamb);
    error_scumsum = 0; % store the cummulative sum of the estimate over different datasets
    rng(1)
    for ite_dataset = 1:N_dataset
        y = w_true + normrnd(0,sqrt(SSigma),[N_samp,N_feat]);
        w_estimate = sum(y) / (N_samp + lambda_temp);
        error_scumsum = error_scumsum + (w_estimate - w_true)^2; 
    end
    error_mean = error_scumsum/N_dataset;
    MSE_lamb(ite_lamb) = error_mean;
end

lambda_opt = 0.1/w_true^2;
lambda_temp = lambda_opt;
error_scumsum = 0; % store the cummulative sum of the estimate over different datasets
rng(1)
for ite_dataset = 1:N_dataset
    y = w_true + normrnd(0,sqrt(0.1),[N_samp,N_feat]);
    w_estimate = sum(y) / (N_samp + lambda_temp);
    error_scumsum = error_scumsum + (w_estimate - w_true)^2; 
end
error_mean = error_scumsum/N_dataset;
MSE_opt = error_mean;

figure 
plot(lambda,MSE_lamb);
hold on 
plot(lambda_opt,MSE_opt,'*');
text(65,1.2e-4,'optimal lambda')
xlabel('estimate index (or lambda value)')
ylabel('MSE(lambda)');
title('MSE vs lambda');

%% optimal lambda for MSE under different settings

N_lamb = 1000;
lambda = linspace(0,500,N_lamb);
N_dataset = 100;
N_feat = 1;

for SSigma = [1,0.5]
    figure
    fig_ind = 1;
    for w_true = [0.4,0.2]
        for N_samp = [400,600]
            
            MSE_lamb = zeros(N_lamb,1);
            subplot(2,2,fig_ind)
            for ite_lamb = 1:N_lamb
                lambda_temp = lambda(ite_lamb);
                error_scumsum = 0; % store the cummulative sum of the estimate over different datasets
                rng(1)
                for ite_dataset = 1:N_dataset
                    y = w_true + normrnd(0,sqrt(SSigma),[N_samp,N_feat]);
                    w_estimate = sum(y) / (N_samp + lambda_temp);
                    error_scumsum = error_scumsum + (w_estimate - w_true)^2; 
                end
                error_mean = error_scumsum/N_dataset;
                MSE_lamb(ite_lamb) = error_mean;
            end
            
            lambda_opt = SSigma/w_true^2;
%             error_scumsum = 0; % store the cummulative sum of the estimate over different datasets
%             rng(1)
%             for ite_dataset = 1:N_dataset
%                 y = w_true + normrnd(0,sqrt(0.1),[N_samp,N_feat]);
%                 w_estimate = sum(y) / (N_samp + lambda_opt);
%                 error_scumsum = error_scumsum + (w_estimate - w_true)^2; 
%             end
%             error_mean = error_scumsum/N_dataset;
%             MSE_opt = error_mean;
           
            title_prompt = sprintf('MSE vs lambda: sigma^2 (%d), w (%d), N (%d):',SSigma,w_true,N_samp);
            plot(lambda,MSE_lamb);
            hold on 
            plot(lambda_opt,0,'*');
            xlabel('estimate index (or \lambda value)')
            ylabel('MSE(\lambda)');
            title(title_prompt);
            hold off
            fig_ind = fig_ind + 1;
            
        end
    end
end






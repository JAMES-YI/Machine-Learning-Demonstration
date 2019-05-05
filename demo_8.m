
%% This file will demonstrate the usage of neural network
% for designing a classifier
%
% The neural network has configuration [2,Nh,1], 
% the output is a scalar: assign label 1 if > 0.5; assign label 0 if < 0.5

% JYI, 11/07/2018

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
Nh = 2; 
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
    if plot_flag2==0
        %% sample distribution and decision boundary in hidden layer
        tit = sprintf('Ite %d: Sample distribution in hidden layer', iter)
        figure;hold on;
        title(tit);
        xlabel('Hidden feature 1');
        xlabel('Hidden feature 2'); 
        aout1 = aout_struct.aout1;
        plot(aout1(1:N1,1),aout1(1:N1,2),'-*');
        plot(aout1(N1+1:N1+N2,1), aout1(N1+1:N1+N2,2),'-o');
        h_feat2 = @(h_feat1) -(W2(1)*h_feat1 + W2(3)) / W2(2);
        fplot(h_feat2);
        legend('Class 1','Class 2','Decision boundary');
        
        %% gird sample classes
        x1range = [-10:0.1:10]; 
        x2range = [-10:0.1:10];
        [X1range,X2range] = meshgrid(x1range,x2range);
        X1range_1D = X1range(:); X2range_1D = X2range(:);
        ens_N = length(X1range_1D);
        ens_feat = [X1range_1D,X2range_1D,ones(ens_N,1)];
        
        ens_wsum1 = ens_feat*W1'; 
        ens_aout1 = tanh(ens_wsum1); 

        ens_aout1 = [ens_aout1,ones(ens_N,1)];
        ens_wsum2 = ens_aout1*W2'; 
        ens_aout2 = tanh(ens_wsum2); 
        ind_c1 = find(ens_aout2>0);
        ind_c2 = find(ens_aout2<0);
        
        figure; hold on;
        title('Discriminate function over grid');
        xlabel('Feature 1');
        ylabel('Feature 2');
        plot(X1range_1D(ind_c1), X2range_1D(ind_c1),'*');
        plot(X1range_1D(ind_c2), X2range_1D(ind_c2),'o');
        legend('Class 1','Class 2');
    
        
    end
    
    iter = iter+1;
end

%% Results report
figure; 
plot(loss_arr,'-*');
title('Loss VS Iterations'); xlabel('Num of iterations'); ylabel('Loss');








%% This file is to compute the gradient of loss with respect to the weight matirx via
% a finite difference method
% 
% JYI, 11/11/2018

clear all
close all
clc
%% data exploration
data = load('twoclass.mat');
c1 = data.s1; c2 = data.s2;
N1 = 100; N2 = 100; Ns = N1+N2;
lab1 = ones(N1,1); lab2 = zeros(N2,1); % lab 1 for class 1, label 0 for class 2

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

%% initialization
X = [feat, ones(N1+N2,1)]; % (N1+N2,Nf+1)
W1 = 0.01*randn(Nh,Nf+1); % (Nh,Nf+1)
W2 = 0.01*randn(1,Nh+1); % (1,Nh+1)
iter = 1;
loss_arr = zeros(Niter-1,1);

grad_W1 = zeros(Nh,Nf+1);
for i=1:Nh
    for j=1:Nf+1
        [~, ~, loss, ~] = ForwardProp(X,lab,W1,W2,Ns);
        
        W1_ij = W1; 
        dW1_ij = 0.001*randn(1);
        W1_ij(i,j) = W1_ij(i,j) + dW1_ij;
        [~,~,loss_ij,~] = ForwardProp(X,lab,W1_ij,W2,Ns);
        
        grad_W1(i,j) = (loss_ij - loss) / (dW1_ij);
    
    end
end


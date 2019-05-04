%% This file is to illustrate the idea for solving the following problem
% 
% Write a MATLAB function to implement the perceptron classifier
% to classify two digits from the MNIST dataset. You may follow the following links
% • You can download the MNIST dataset from http://yann.lecun.com/exdb/mnist/.
% • You can also use the MNIST helper files from http://ufldl.stanford.edu/wiki/resources/
% mnistHelper.zip to load the training images and the labels to MATLAB.
% • See http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset for directions on
% using the helper files.
% You will focus on classifying the images corresponding to zeros and ones.
% (a) Identify and visualize the images corresponding to the digits zero and one using the labels. You
% can use find(labels==0) to find the indices of the zeros. You need to reshape the vectors to
% 28 × 28 to visualize them using imshow.
% (b) Augment the data to obtain zn as shown in page 38 of course notes (Lecture 04). See Section
% 18.2 in the text book for details.
% (c) Implement the iterative algorithm and choose the appropriate step size to ensure convergence. You
% may come up with your own termination algorithm. Keep track of the number of mis-classified
% points at each iteration; this should decrease with iteration.
% (d) Visualize the weight vector a at each iteration. Specifically, note that a = [w,w0]. Extract w
% and visualize it as a 28x28 image.
% (e) Modify the code to classify other pair of digits. Note how does the template w change as a
% function of the pair of digits it is trying to classify ?
% 
% The idea is based on Section 18.2 in [1]
% S. Theodoridis. Machine learning: a Bayes and optimization prospective.
% Academic Press, 2015.

% By JYI on 10/26/2018

%% data exploration
images = loadMNISTImages('train-images.idx3-ubyte'); % (784,60000)
labels = loadMNISTLabels('train-labels.idx1-ubyte'); % (60000,1)

ind_zero = find(labels==0);
lab_0 = labels(ind_zero); % (5923,1)
num_0 = length(lab_0);
img_0 = images(:,ind_zero); % (784,5923)

ind_one = find(labels==1);
lab_1 = labels(ind_one); % (6742,1)
num_1 = length(lab_1);
img_1 = images(:,ind_one); % (784,6742)

% visualization of sample images
figure; hold on;
subplot(1,2,1);
img_illus = reshape(img_0(:,1),[28,28]);
imshow(img_illus);
subplot(1,2,2);
img_illus = reshape(img_1(:,1),[28,28]);
imshow(img_illus);

%% data augmenting
% assign 1 if digit 1, assign -1 if digit 0
% y=x^T*theta: digit 1 (assign 1) if negative; 
%              digit 0 (assign -1) if positive;
lab_0 = -ones(size(lab_0));
aug = ones(1,num_0);
aug_img_0 = [img_0;aug];
z_zero = lab_0' .* aug_img_0;

aug = ones(1,num_1);
aug_img_1 = [img_1;aug];
z_one = lab_1' .* aug_img_1;

z = [z_zero,z_one]; % (785,12665)
lab = [lab_0;lab_1]; % (12665,1)
[num_var,num_samp] = size(z);

%% optimize perceptron: preparation
mu = 0.05; % defaul 0.05 
MaxIter = 1000; % defaul 1000
tol = 10^(-2);
rng(0);
theta = randn(num_var,1);

f_0 = z_zero'*theta; % correct if positive; wrong if negative
ind_wrong0 = find(f_0<0);
f_1 = z_one'*theta; % correct classification if negative; wrong classification if posi
ind_wrong1 = find(f_1>0);

z_wrong = [z_zero(:,ind_wrong0),z_one(:,ind_wrong1)];
zsum_wrong = sum(z_wrong,2);
theta_new = theta + mu*zsum_wrong;

temp_wrong = size(z_wrong,2);
num_wrong = [];
theta_store = [];
theta_store = [theta_store,theta_new];

%% optimize perceptron: iteration
for i=1:MaxIter
    if norm(theta_new - theta,2) < tol
        fprintf('Converged!\n');
        break;
    end
    
    if i==MaxIter
        fprintf('Maximal iterations achieved!\n');
    end
    
    theta = theta_new;
    
    f_0 = z_zero'*theta; % correct if positive; wrong if negative
    ind_wrong0 = find(f_0<0);
    f_1 = z_one'*theta; % correct classification if negative; wrong classification if posi
    ind_wrong1 = find(f_1>0);

    z_wrong = [z_zero(:,ind_wrong0),z_one(:,ind_wrong1)];
    zsum_wrong = sum(z_wrong,2);
    theta_new = theta + mu*zsum_wrong;
    temp_wrong = size(z_wrong,2);
    
    num_wrong = [num_wrong,temp_wrong];
    theta_store = [theta_store,theta_new];
    
end

%% parameter visualization
figure; hold on;
subplot(2,2,1)
title('iteration 1');
w_img = reshape(theta_store(1:end-1,1),[28,28]);
imshow(w_img);

subplot(2,2,2)
title('iteration 2');
w_img = reshape(theta_store(1:end-1,2),[28,28]);
imshow(w_img);

subplot(2,2,3)
title('iteration 3');
w_img = reshape(theta_store(1:end-1,3),[28,28]);
imshow(w_img);

subplot(2,2,4)
title('iteration 4');
w_img = reshape(theta_store(1:end-1,4),[28,28]);
imshow(w_img);


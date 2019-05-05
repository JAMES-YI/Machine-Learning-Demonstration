function [aout_struct, wsum_struct, loss, aout2] = ForwardProp(X,lab,W1,W2,Ns)
%% This file will perform the forward propagation in a two layer neural network
% JYI, 11/11/2018
    aout0 = X; % (N1+N2,Nf+1)

    wsum1 = X*W1'; % (N1+N2,Nh)
    aout1 = tanh(wsum1); % (N1+N2,Nh)

    aout1 = [aout1,ones(Ns,1)]; % (N1+N2,Nh+1)
    wsum2 = aout1*W2'; % (N1+N2,1)
    aout2 = tanh(wsum2); 

    aout_struct.aout0 = aout0;
    aout_struct.aout1 = aout1;
    aout_struct.aout2 = aout2;

    wsum_struct.wsum1 = wsum1;
    wsum_struct.wsum2 = wsum2;

    loss = (1/2) * (norm(aout2 - lab,2))^2;
end
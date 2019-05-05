function [delta_struct,theta_struct] = BackProp(lab,W1,W2,aout_struct,wsum_struct,nn_config)
%% This file will perform the backward propagation in a two layer neural network
% JYI, 11/11/2018
    aout2 = aout_struct.aout2;
    wsum2 = wsum_struct.wsum2;
    wsum1 = wsum_struct.wsum1;
    Nh = nn_config.Nh;

    theta2 = aout2 - lab; % (N1+N2,1)
    delta2 = theta2 .* (sech(wsum2)).^2; % (N1+N2,1)
    theta1 = delta2 * W2; % (N1+N2,Nh+1)
    delta1 = theta1(:,1:Nh) .* (sech(wsum1)).^2; % (N1+N2,Nh); 
    theta0 = delta1 * W1; % (N1+N2,Nf+1)

    delta_struct.delta1 = delta1;
    delta_struct.delta2 = delta2;

    theta_struct.theta0 = theta0;
    theta_struct.theta1 = theta1;
    theta_struct.theta2 = theta2;
end
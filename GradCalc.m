function [grad_W1,grad_W2] = GradCalc(aout_struct,delta_struct,nn_config,Ns)
%% This file will calculate the gradient of loss function with respect to weight
% JYI, 11/11/2018

    aout0 = aout_struct.aout0;
    aout1 = aout_struct.aout1;
    delta1 = delta_struct.delta1;
    delta2 = delta_struct.delta2;
    Nf = nn_config.Nf;
    Nh = nn_config.Nh;

    grad_W1 = zeros(Nh,Nf+1); grad_W2 = zeros(1,Nh+1);
    for i = 1:Ns

        delta1_i = delta1(i,:); % (1,Nh)
        aout0_i = aout0(i,:); % (1,Nf+1)  
        grad_W1 = grad_W1 + kron(delta1_i',aout0_i); 

        delta2_i = delta2(i); % (1,1)
        aout1_i = aout1(i,:); % (1,Nh+1)
        grad_W2 = grad_W2 + kron(delta2_i,aout1_i);

    end
end
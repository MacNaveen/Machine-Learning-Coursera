function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Implement Part I -- returning the cost variable J

% J(theta) = 1/m * sigma(i=1 to m) of sigma(k=1 to K) of [
%                     -y(i)k * log((h(theta)(x(i))k) -
%                    (1y(i)k) * log(1 - (h(theta)(x(i))k))]
% where h(theta(x(i))k) is the result of the NN forward prop
% and K is the total number of possible layers.

% First initialize X input to include a row of ones

X = [ones(m,1) X];

% Now, implement forward propagation, similar to ex3

z1 = sigmoid(Theta1 * X');
a2 = [ones(1, size(z1, 2)); z1];
a3 = sigmoid(Theta2 * a2);
h = a3;

Y_Vector = zeros(num_labels, m); %(10*5000)

for i=1:num_labels,
    Y_Vector(i,:) = (y==i);%(y is 5000*1)
endfor

J = (sum( sum( -1*Y_Vector.*log(h) - (1 - Y_Vector).*log(1-h) ) ))/m;


Theta1Reg = Theta1(:,2:size(Theta1,2));
Theta2Reg = Theta2(:,2:size(Theta2,2));

Reg = (lambda/(2*m)) * (sum(sum( Theta1Reg.^2 )) + sum( sum( Theta2Reg.^2 ) ));

J = J + Reg;

%------------------------------------------------------------


%------------------------------------------------------------------------------------------

for k = 1:m,

    a1 = X(k,:);
    z2 = Theta1 * a1';

    a2 = sigmoid(z2);
    a2 = [1 ; a2];

    a3 = sigmoid(Theta2 * a2);


    d3 = a3 - Y_Vector(:,k);
    
    z2 = [1 ; z2];
    d2 = (Theta2' * d3) .* sigmoidGradient(z2);
    d2 = d2(2:end);

    Theta2_grad = (Theta2_grad + d3 * a2');
    Theta1_grad = (Theta1_grad + d2 * a1);

endfor;


Theta1_grad(:,1) = Theta1_grad(:,1)./m;
Theta2_grad(:,1) = Theta2_grad(:,1)./m;


Theta1_grad(:,2:end) = Theta1_grad(:,2:end)./m + ( (lambda/m) * Theta1(:,2:end) );
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)./m + ( (lambda/m) * Theta2(:,2:end) );

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
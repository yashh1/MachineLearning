function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


predict = sigmoid(X * theta);
errors = -1 * ( (y .* log(predict)) + ( (1 - y) .* log(1 - predict)) );
T = (1/m) * sum(errors);
P = lambda/(2*m) * (sum( theta.^2 )-(theta(1).^2));
J = T + P;



    error = predict - y;
    val = zeros(size(theta));
    for j=1:size(theta,1),
      val(j) = (error')* X(:,j);
    end;
    grad = val.*(1/m) + (lambda/m)*theta ;
    grad(1) = val(1).*(1/m);




% =============================================================

end

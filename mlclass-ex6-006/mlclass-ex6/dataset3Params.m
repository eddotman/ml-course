function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
best_sigma_err = 10000;
best_C_err = 10000;

for i = 1:length(Cs)
   for j = 1:length(sigmas)
        xt1 = X(:, 1);
        xt2 = X(:, 2);
        model = svmTrain(X, y, Cs(i), @(xt1, xt2) gaussianKernel(xt1, xt2, sigmas(j)));
        predictions = svmPredict(model, Xval);

        error = mean(double(predictions ~= yval));
        
        if error < best_sigma_err
            best_sigma_err = error;
            sigma = sigmas(j);
        end
        if error < best_C_err
            best_C_err = error;
            C = Cs(i);
        end
   end
end


% =========================================================================

end

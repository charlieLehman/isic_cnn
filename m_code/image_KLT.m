% usage: [output_image] = image_KLT(input_image_dir, num_eig_vectors)
% 
% This function takes the KLT and keeps only the first k eigen-vectors 

% input_image_dir expects a directory
% num_eig_vectors expects an integer
% output_image is the same size as original
% Example:
% 
%  image_KLT_150 = image_KLT('/Images/statue.png', 150);
%  imshow(image_KLT_150);

function [output_image] = image_KLT(input_image_dir,k) 

I = double(imread(input_image_dir));
for n = 1:3 %have to do it for each RGB channel
    X = I(:,:,n);
    [M,N] = size(X); 
    u = mean(X,2); %takes the mean for each row
    U = repmat(u,1,N); %repeats the mean column (u) N times
    Y = X-U; %calculate the difference matrix
    R = Y*Y'/N; % constructs the autocorrelation matrix from Y
    [V D] = svd(R); %eigenvalue decomposition to get the vector matrix, V.
    A = zeros(M,N);
    A(:,1:k) = V(:,1:k); % Makes the transformation matrix 
    output_image(:,:,n) = A'*Y; %takes the KL-transform of the image
end

end 
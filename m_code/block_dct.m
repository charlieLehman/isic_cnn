% usage: [output_image] = block_dct (input_image, block_size)
% 
% This takes the blocked Discrete Cosine Transform of the
% input image. 

% input_image expects a directory
% block_size expects an integer
%
% Example:
%
%  image_DCT_8 = block_dct ("/images/lena.jpg", 8);
%  imshow (image_DCT_8);

function output_image = block_dct(input_image, block_size)
  for n = [1,2,3]
    output_image(:,:,n) = uint8(255*blockproc(double(input_image(:,:,n))/255,[block_size block_size],'dct2'));
  endfor
end

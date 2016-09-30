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

function [output_image] = block_dct(input_image, block_size)
  I = double(imread(input_image));
  T = dctmtx(block_size);
  dct = @(block_struct) T * block_struct.data * T';
  output_image = blockproc(I,[block_size block_size],dct);
  output_image = uint8(output_image)
end

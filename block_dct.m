function [output_image] = block_dct(input_image, block_size)


I = double(input_image);

T = dctmtx(block_size);
dct = @(block_struct) T * block_struct.data * T';
output_image = blockproc(I,[block_size block_size],dct);


end

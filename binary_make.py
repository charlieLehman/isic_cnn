from multiprocessing import Pool
from tqdm import tqdm
import utils as u

pool = Pool()
#tags = ['_32','_HSV','_FFT','_FFT_of_HSV','_DCT','_DCT_of_HSV']
tags = ['_HSV']
rs = pool.imap(u.cifar.make_binary,tags)
for n in tqdm(tags):
    rs.next()

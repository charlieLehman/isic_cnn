from multiprocessing import Pool
from tqdm import tqdm
import utils as u

#u.cifar.make_binary('_256')
#u.cifar.make_binary('_HSV')
u.cifar.make_binary('_FFT')
#u.cifar.make_binary('_FFT_of_HSV')
#u.cifar.make_binary('_DCT')
#u.cifar.make_binary('_DCT_of_HSV')

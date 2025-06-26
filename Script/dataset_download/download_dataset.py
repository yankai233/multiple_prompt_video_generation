import openxlab
openxlab.login(ak='ro6zvqjgaxy2j1bryb39', sk='om5x93exb4nvzz6kprwnxpz8o1gbj2klgjmwandq') #进行登录，输入对应的AK/SK
from openxlab.dataset import info
info(dataset_repo='OpenDataLab/MovieNet') #数据集信息及文件列表查看

from openxlab.dataset import get
dir_path = r'E:\Data\datasets\Video_Datasets\MovieNet'
get(dataset_repo='OpenDataLab/MovieNet', target_path=dir_path)  # 数据集下载

from openxlab.dataset import download
download(dataset_repo='OpenDataLab/MovieNet',source_path='/README.md', target_path=dir_path) #数据集文件下载



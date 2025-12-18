# data preprocess

## ml-1m
1. 原始数据格式
    * 经过preprocessor处理后，得到`sasrec_format.csv`
![alt text](images/ml-1m-sasrec_format_describ.png)
2. DatasetV2
    * 读取ratings_file
    * reverse+padding
        * padding是必须的，因为需要一个batch的数据进行stack
3. `interleave`格式处理
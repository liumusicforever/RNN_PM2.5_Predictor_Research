import os
import random
import pandas as pd
import numpy as np

def read_sensor_data(sensor_root):
    all_sensor_data = []
    for i,sensor in enumerate(os.listdir(sensor_root)):
        file_path = os.path.join(sensor_root , sensor)
        series = pd.read_csv(file_path, header=None) 
        all_sensor_data.append(series.values.tolist())
        print ('loaded file : {}'.format(file_path))
        if i == 100:
            return all_sensor_data



class data_generator:
    '''    
    return (batch_data , batch_label)
    shape of (batch_data , batch_label) : 
        [batch_size , int(sample_feq/sample_skip) , number of attrubute]    
    '''
    def __init__(self ,  data_path ,\
                         batch_size  = 3,\
                         sample_feq  = 20 ,\
                         sample_skip = 2 ,\
                         sample_times = 1000 ,\
                         data_format = 'sensor'):

        self.batch_size = batch_size
        self.sample_feq = sample_feq
        self.sample_skip = sample_skip
        self.sample_times = sample_times
        
        self.current_time = 0
        
        if data_format == 'sensor':
            self.all_datas = read_sensor_data(data_path)
    
    def __next__(self): 
        if self.current_time > self.sample_times:
            raise StopIteration
        else:
            self.current_time += 1
        
        num_dev = len(self.all_datas)
        
        batch_data = []
        batch_label = []
        
        while len(batch_data) < self.batch_size:
            
            dev_idx = random.choice(range(num_dev))
            dev_datas = self.all_datas[dev_idx]


            num_rows = len(dev_datas)
            row_idx = random.choice(range(num_rows))
            while (row_idx - self.sample_feq -1 ) < 0 :
                row_idx = random.choice(range(num_rows))


            row_seq = [i for i in range(row_idx - self.sample_feq  , row_idx  , self.sample_skip)]
            
            data = [[float(j)/3000 for j in dev_datas[i][2:]] for i in row_seq]
            label =[float(j)/3000 for j in dev_datas[row_idx][1:2]]
            
            # check if non zero
#             if all(data) and len(data[0]) == 7: 
            if len(data[0]) == 6: 
                batch_data.append(data)
                batch_label.append(label)
        
        
        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)
            
        
        
        return (batch_data , batch_label)
    def __iter__(self):
        return self
    
        
def test_data_generator():
    data_root = '/root/data/pm25_data/datas/sensors/train/'
    generator = data_generator(data_root)
    
    for i , (a,b) in enumerate(generator):
        print (a.shape)
        print (b.shape)
        quit()
    
if __name__ == '__main__':
    test_data_generator()
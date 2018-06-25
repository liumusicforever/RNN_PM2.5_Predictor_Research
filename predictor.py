import numpy as np
import tensorflow as tf
from networks import model

from config import timesteps, num_input

def Predictor(model_path , network = 'basic_lstm'):
    
    X = tf.placeholder("float", [None, timesteps, num_input])
    
    if network == 'basic_lstm' :
        prediction = model(X , network = 'basic_lstm' , training = False)
    elif network == 'muti_lstm' :
        prediction = model(X , network = 'muti_lstm' , training = False)
    
    sess = tf.Session()
    saver = tf.train.Saver() 
    saver.restore(sess , model_path)
    
    def predict( batch_x ):
        return sess.run(prediction , feed_dict={X: batch_x})
        
    return predict



def predict_taiwan_csv():
    import os
    import csv
    def read_csv(path):
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            data_list = list(reader)
        return data_list
    
    file_path = '/root/data/pm25_data/datas/Taiwan_data/201703_Taiwan_sol.csv'
    dev_path = '/root/data/pm25_data/datas/sensors/test/'
    model_path = 'models/basic_lstm_387.ckpt'
    predictor = Predictor(model_path , network = 'basic_lstm')
    
    all_data = read_csv(file_path)
    
    for i , data in enumerate(all_data[1:]):
        date , time , dev , *content = data[:]
        print (date , time , dev , content)
        dev = os.path.join(dev_path , dev + '.csv')
        dev_data = read_csv(dev)
        print (len(dev_data))
        
        
        break
    
    

def predict_devices():
    import os
    import csv
    import config
    
    model_path = 'models/basic_lstm_387.ckpt'
    predictor = Predictor(model_path , network = 'basic_lstm')
    
    
    device_roots = '/root/data/pm25_data/datas/sensors/test/'
    devs = os.listdir(device_roots)
    
    predict_out = '/root/data/pm25_data/datas/sensors/prediction_basic_lstm/'
    for num_file , file_name in enumerate(devs):
        result = []
        file_path = os.path.join(device_roots , file_name)
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            data_list = list(reader)
            
        
        for cur in range(config.sample_feq , len(data_list)):
            
            row_seq = [i for i in range(cur - config.sample_feq  , cur  , config.sample_skip)]
            data = [[float(j)/3000 for j in data_list[i][2:8]] for i in row_seq]
            
            pred = predictor([data])
            data_list[cur].append(pred[0][0] * 3000)
            
        with open(predict_out + '/' + file_name, "w", encoding = "utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(data_list)
        print ('finish : {} , total : {}'.format(num_file , len(devs)))
        


def test_Predictor():
    model_path = 'models/basic_lstm_387.ckpt'
    predictor = Predictor(model_path , network = 'basic_lstm')
    
    import numpy as np
    import config
    x =  np.zeros([config.batch_size , config.timesteps , config.num_input])
    print (predictor(x))

if __name__ == '__main__':
    predict_taiwan_csv()
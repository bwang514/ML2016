import numpy as np
import csv

def running_test_data(Coef,data_size):
    data = np.transpose(np.genfromtxt("test_X.csv",delimiter = ","))
    output = []
    out_fin = []  
    for i in xrange(0,2):
        data = np.delete(data,0,0)   # delete non-related infos
    for i in xrange(0,9):  
        output.append(np.split(data[i],len(data[i]) / data_size))
    for i in xrange(0,len(output[0])) :
        for j in xrange(0,9) :
            out_fin.append(output[j][i]) 
    out_fin = np.array(out_fin)
    out_fin[np.isnan(out_fin)] = 0             
    ## now output is a list of testing data
   ## print "hihi"
    predict = predictData(Coef,out_fin)
    return predict
def data_visualiztion():
    pass


def predictData(Coef,data):
    totalError = []
    validate_factor = 0
    N = float(len(data))
    ##print N
    for i in xrange(0,int(N),9):
        ##print i
        zone = np.array([1])
        data_slot = data[i:i+9]
##        target_pm = data[i+9][9]       
        for j in xrange(0,9):
            zone = np.concatenate((zone,data_slot[j])) 
        innerproduct = np.dot(Coef,zone)
        totalError.append(innerproduct)   
    return totalError

def readCSV(data_size):
    data = np.delete(np.transpose(np.genfromtxt("train.csv",delimiter = ",")),0,1)
    output = []
    out_fin = []  
    for i in xrange(0,3):
    	data = np.delete(data,0,0)   # delete non-related infos
    for i in xrange(0,24):	
    	output.append(np.split(data[i],len(data[i]) / data_size))
    for i in xrange(0,len(output[0])) :
        for j in xrange(0,24) :
            out_fin.append(output[j][i]) 
    out_fin = np.array(out_fin)
    out_fin[np.isnan(out_fin)] = 0   
 ##   print "debug nan"
 ##   print len(out_fin) 
    return out_fin

def step_gradient(Coef, data, learning_rate):
    gradient = np.zeros(len(Coef))
    data_iterator = 0
    validate_factor = 3
    N = float(len(data))
    for i in range(0,int(N * validate_factor / 3) - 10):
        zone = np.array([1])
        data_slot = data[i:i+9]
        target_pm = data[i+9][9]       
        for j in xrange(0,9):
            zone = np.concatenate((zone,data_slot[j])) 
        innerproduct = np.dot(Coef,zone)   
        gradient[0] += -(2/N) * (target_pm - innerproduct)
        for k in xrange(1,len(gradient)):                                           
            gradient[k] += -(2/N) * zone[k] * (target_pm - innerproduct)    
      ##  print learning_rate * gradient[1]
    Coef = Coef - learning_rate * gradient
    #print "Coef = " + str(Coef)         
    return Coef
	
def gradient_descent_runner(data,Coef_initial,learning_rate, num_iterations):
    Coef = Coef_initial
    for i in range(num_iterations):
        print i
        Coef = step_gradient(Coef,data,learning_rate)
        
    return Coef
def main():
    data_size = 18
    training_features = 163
    learning_rate = 0.001
    num_iterations = 500
    Coef = np.zeros(training_features) # coefficients
    data = readCSV(data_size)
    ## start runing gradient descent
    print "Running Gradient Descent with learning rate = " + str(learning_rate) + " iterations = " + str(num_iterations)
    Coef = gradient_descent_runner(data,Coef,learning_rate,num_iterations)
    predict = running_test_data(Coef,data_size)
    print predict
    f = open('predict.csv', 'w')
    f.write("id,value\n")
    for i in xrange(0,240) :
        tmp = "id_" + str(i) + "," + str(predict[i]) + "\n"  
        f.write(tmp)



if __name__ == '__main__':
    main()

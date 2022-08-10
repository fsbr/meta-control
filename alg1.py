# Input an Anytime algorithm A, Performa
from giving_in import BIT_STAR, Visualizer
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def sigmoid(x, tht1, tht2, tht3):
    #return (tht1/(np.exp(-tht2*x) + 1)) + tht3
    return tht1*np.arctan(x + tht2) + tht3
    #return tht1*np.log(x + tht2) + tht3


# for 50x50 i liked a = 1000, b = 0.1
# for 10x10 i like ....
def utility(t,q,a=1e3,b=1):
    return a*q-np.exp(b*t)
    #return np.exp(a*q) - b*t
      
if __name__ == "__main__":

    test_length =25 
    bs = BIT_STAR()
    #bs.readEnvironment("test_environments/grid_envs/environment104.txt")
    #bs.readEnvironment("test_environments/grid_envs/environment3.txt")
    #bs.readEnvironment("test_environments/grid_envs50/environment50_74.txt")
    bs.readEnvironment("test_environments/grid_envs50/environment50_7.txt")
    #bs.readEnvironment("test_environments/grid_envs1000/environment1000_74.txt")
    #bs.readEnvironment("test_environments/grid_envs50/environment50_57.txt")
    #bs.readEnvironment("test_environments/grid_envs50/environment50_53.txt")

    # 1. T <- 0
    t_start = time.time()   
    t_end = time.time() + test_length
    delta_t = 0.1 

    # performance history
    performance_history_t = np.array([])
    performance_history_q = np.array([])
    print("delta t", delta_t)
    bs.BIT_STAR_MAIN(t_start, t_end, False)
    stopCondition = True 

    # while A.running() ....
    t_start_iteration = t_start 
    datapoints = 0
    while stopCondition:
        V, E = bs.BIT_STAR_MAIN_LOOP(t_start, t_end, stopCondition)

        # end it if the timer times out
        if time.time() > t_end:
            stopCondition = False 
        t_end_iteration = time.time()
        if t_end_iteration - t_start_iteration >= delta_t:
            print("time and cost check")
            t = t_end_iteration - t_start
            #q = 1/bs.c
            q = bs.start.fHat / bs.c
            performance_history_t = np.append(performance_history_t, t)
            performance_history_q = np.append(performance_history_q, q)
            print("time = %s || cost = %s "%(t, bs.c))
            print("quality = ", (q))

            # curve fitting
            # datapoints = 4
            if datapoints > 50:
                popt, pcov = curve_fit(sigmoid, performance_history_t, performance_history_q,
                                p0=[1, 1, 1], maxfev=50000)
                # there needs to be a comparison of the utility and the prediction
                U = utility(t, q) 
                q_predicted = sigmoid(t+delta_t, *popt)
                U_predicted = utility(t+delta_t, q_predicted)
                print("Utility = %s"%(U))
                print("Utility Prediction= %s"%(U_predicted))
                if bs.c < np.Inf:
                    print("Solution Found")
                    delta_U = U_predicted - U
                    print("Difference in Prediction == ", delta_U)
                    if delta_U > 0:
                        stopCondition = True 
                    else:
                        stopCondition = False 
            t_start_iteration = time.time()
            datapoints+=1


    gv = Visualizer(bs)
    gv.plotMotionTree()


    print("popt", *popt)
    #plt.plot(t, sigmoid(t, *popt))

    # i think this doesnt quite work becasue the prediction is changing throughout
    y_pred = sigmoid(performance_history_t, *popt) 
    print("y_pred", y_pred)

    # plotting
    plt.plot(performance_history_t, y_pred, label = "prediction")
    #plt.plot(performance_history_t, predicted_q_vector, label = "prediction")
    plt.scatter(performance_history_t, performance_history_q, label= "performance history")
    plt.grid()
    plt.legend()
    plt.title("Solution Quality vs. Time tht=%s"%(popt))
    plt.xlabel("time (s)")
    plt.ylabel("Solution Quality")
    plt.show()

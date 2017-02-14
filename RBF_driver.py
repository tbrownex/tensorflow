import RBF
import pandas as pd
import os
import sys
import time

clusters = ['km']
nodes    = [5, 6, 7]
sigma    = [2.0, 2.2, 2.4]
test     = [.21, .23, .25]
loops    = 3

def get_data():
    if sys.platform[:3] =='win': data_loc = 'D:/Data/Loyalty Vision/'
    else: data_loc = "/home/tom/data/"

    filenm = "rbf_data.csv"
    return(pd.read_csv(data_loc+filenm, delimiter=','))

if __name__ == "__main__":
    parms = [[a,b,c,d] for a in sigma for b in clusters for c in nodes for d in test]
    out = open('/home/tom/data/rbf_results.csv', 'w')
    rec = '{}{}{}{}{}{}{}'.format('Sigma,','Cluster,','Nodes,','TestPct,','RMSE,','Time,','\n')
    out.write(rec)

    df = get_data()
    
    sigma=2.
    cluster='random'
    nodes=8
    t0 = time.time()

    for x in parms:
        start = time.time()
        for y in range(loops):
            rmse = RBF.process(df, x)   # Pass the data and the parameters
            end = time.time()
            rec = str(x[0])+','+x[1]+','+str(x[2])+','+str(x[3])+','+str(rmse)+','+str(end-start)+'\n'
            out.write(rec)
        
    t1 = time.time()
    print('all done, elapsed time: {:.1f} minutes'.format((t1-t0)/60))
    os.system("aplay /usr/share/sounds/bicycle_bell.wav")
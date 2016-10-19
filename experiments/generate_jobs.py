import itertools
import sys
import numpy as np
def generate_jobs(learning_rate, filter_dim, hidden_dim, num_epochs):  
    parts_job = ["python", "exp_conv_vae.py", str(learning_rate), \
    str(filter_dim), str(hidden_dim),str(num_epochs)]
    s1 = " ".join(parts_job) 
    return s1
 



def main():
    N = 18
    lrlist = 10 ** np.random.uniform(-3.5,-2.5,N)
    fdlist = np.random.choice([16,32,64],N)
    hdlist = np.random.choice([50,100,200],N)
    with open('job.sh','w') as f:
        for t in range(N): 
            f.write(generate_jobs(lrlist[t],fdlist[t],hdlist[t],10) + '\n')

if __name__ == "__main__":
    main()
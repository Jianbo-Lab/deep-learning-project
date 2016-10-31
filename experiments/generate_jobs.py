import itertools
import sys
import numpy as np

 



def generate_all_jobs(model):
    if model == 'convvae':
        # For conv VAE.
        def generate_jobs(learning_rate, filter_dim, hidden_dim, num_epochs):  
            parts_job = ["python", "exp_conv_vae.py", str(learning_rate), \
            str(filter_dim), str(hidden_dim),str(num_epochs)]
            s1 = " ".join(parts_job) 
            return s1
        N = 18
        lrlist = 10 ** np.random.uniform(-3.5,-2.5,N)
        fdlist = np.random.choice([16,32,64],N)
        hdlist = np.random.choice([50,100,200],N)
        with open('job.sh','w') as f:
            for t in range(N): 
                f.write(generate_jobs(lrlist[t],fdlist[t],hdlist[t],10) + '\n')
    elif model == 'fcvae':
        # For fc VAE.
        def generate_jobs(learning_rate, hidden_dim, z_dim, num_epochs):  
            parts_job = ["python", "exp_vae.py", str(learning_rate), \
            str(hidden_dim),str(z_dim), str(num_epochs)]
            s1 = " ".join(parts_job) 
            return s1        
        N = 30
        lrlist = 10 ** np.random.uniform(-3.5,-2.5,N)
        hdlist = np.random.choice([500,700,1000],N)
        zdlist = np.random.choice([20,30,50],N)
        with open('fcvae_job.sh','w') as f:
            for t in range(N): 
                f.write(generate_jobs(lrlist[t],hdlist[t],zdlist[t], 30) + '\n')

 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',type = str) 
    args = parser.parse_args()   
    generate_all_jobs(args.model) 

if __name__ == "__main__":
    main()
import tensorflow as tf

def linear(x, output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    
    w = tf.get_variable("w", [x.get_shape()[1], output_dim]) 
    b = tf.get_variable("b", [output_dim], initializer = tf.constant_initializer(0.0))
    # print '!!!!!!!!!!!!!!!', w.get_shape(), b.get_shape()
    return tf.matmul(x,w)+b


def filterbank(gx, gy, sigma2, delta, N, A = 28, B = 28, eps = 1e-8):
    # Compute Fx, Fy
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
    Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True), eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True), eps)
    return Fx,Fy



def attn_window(scope, h_dec, N,  A, B, reuse, eps):
    # Compute Fx, Fy and gamma
    with tf.variable_scope(scope,reuse = reuse):
        params=linear(h_dec, 5)
        gx_,gy_,log_sigma2,log_delta,log_gamma = tf.split(1,5,params)
        gx = (A + 1) / 2*( gx_ + 1)
        gy = (B + 1) / 2*( gy_ + 1)
        sigma2 = tf.exp(log_sigma2)
        delta = (max(A, B)-1)/(N-1) * tf.exp(log_delta) # batch x N
        return filterbank(gx,gy,sigma2,delta,N, A, B, eps)+(tf.exp(log_gamma),)



## READ ## 
def read_no_attn(x,x_hat,h_dec_prev, A, B, read_n, DO_SHARE, eps):
    return tf.concat(1,[x,x_hat])

def read_attn(x,x_hat,h_dec_prev, A, B, read_n, DO_SHARE, eps):
    # Compute x, x_hat
    Fx,Fy,gamma=attn_window("read", h_dec_prev, read_n, A, B, DO_SHARE, eps)

    def filter_img(img,Fx, Fy,gamma,N):
        Fxt=tf.transpose(Fx,perm=[0,2,1])
        img=tf.reshape(img,[-1, B, A])
        glimpse=tf.batch_matmul(Fy,tf.batch_matmul(img,Fxt))
        glimpse=tf.reshape(glimpse,[-1,N*N])
        return glimpse*tf.reshape(gamma,[-1,1])

    x=filter_img(x,Fx,Fy,gamma, read_n) # batch x (read_n*read_n)
    x_hat=filter_img(x_hat,Fx,Fy,gamma, read_n)
    return tf.concat(1,[x,x_hat]) # concat along feature axis



# Q sample

def sampleQ(h_enc, reuse, batch_size, z_size):
    """
    Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
    mu is (batch,z_size)
    """
    e = tf.random_normal([batch_size, z_size])

    with tf.variable_scope("mu", reuse = reuse):
        mu = linear(h_enc, z_size)
    with tf.variable_scope("sigma", reuse = reuse):
        logsigma = linear(h_enc, z_size)
        sigma = tf.exp(logsigma)
    return (mu + sigma*e, mu, logsigma, sigma)
            


## WRITER ## 
def write_no_attn(h_dec, reuse, write_n, A, B, eps, batch_size):
    with tf.variable_scope("write",reuse = reuse):
        return linear(h_dec, A* B)


def write_attn(h_dec, reuse, write_n, A, B, eps, batch_size):
    with tf.variable_scope("writeW",reuse = reuse):
        w = linear(h_dec, write_n * write_n) # batch x (write_n*write_n)
        N = write_n
        w=tf.reshape(w,[batch_size, N, N])
        Fx,Fy,gamma = attn_window("write",h_dec, write_n,  A, B, reuse, eps)
        Fyt = tf.transpose(Fy, perm=[0,2,1])
        wr = tf.batch_matmul(Fyt, tf.batch_matmul(w,Fx))
        wr = tf.reshape(wr,[batch_size, B * A])
        #gamma=tf.tile(gamma,[1,B*A])
        return wr * tf.reshape(1.0/gamma,[-1,1])


       

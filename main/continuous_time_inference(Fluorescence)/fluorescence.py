"""
@author: Maziar Raissi

Modifications by Ram Ananth
"""

"""
The whole domain is split as 6 regions as defined below
r = right boundary
l = left boundary
t = top boundary
b = bottom boundary
ts = tissue
fl = fluorophore

f is governing equation 
g is boundary condition

x1 and x2 are dimensions
"""


import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
from constants import *
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, layers):
        self.ga=100
        
        self.x1_r = X['r'][:,0:1]
        self.x2_r = X['r'][:,1:2]
        
        self.x1_l = X['l'][:,0:1]
        self.x2_l = X['l'][:,1:2]

        self.x1_t = X['t'][:,0:1]
        self.x2_t = X['t'][:,1:2]

        self.x1_b = X['b'][:,0:1]
        self.x2_b = X['b'][:,1:2]

        self.x1_ts = X['ts'][:,0:1]
        self.x2_ts = X['ts'][:,1:2]

        self.x1_fl = X['fl'][:,0:1]
        self.x2_fl = X['fl'][:,1:2]

        self.x1_so = X['so'][:,0:1]
        self.x2_so = X['so'][:,1:2]

        


        
           
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf Placeholders        

        self.x1_r_tf = tf.placeholder(tf.float32, shape=[None, self.x1_r.shape[1]])
        self.x2_r_tf = tf.placeholder(tf.float32, shape=[None, self.x2_r.shape[1]])

        self.x1_l_tf = tf.placeholder(tf.float32, shape=[None, self.x1_l.shape[1]])
        self.x2_l_tf = tf.placeholder(tf.float32, shape=[None, self.x2_l.shape[1]])
        
        self.x1_t_tf = tf.placeholder(tf.float32, shape=[None, self.x1_t.shape[1]])
        self.x2_t_tf = tf.placeholder(tf.float32, shape=[None, self.x2_t.shape[1]])

        self.x1_b_tf = tf.placeholder(tf.float32, shape=[None, self.x1_b.shape[1]])
        self.x2_b_tf = tf.placeholder(tf.float32, shape=[None, self.x2_b.shape[1]])

        self.x1_ts_tf = tf.placeholder(tf.float32, shape=[None, self.x1_ts.shape[1]])
        self.x2_ts_tf = tf.placeholder(tf.float32, shape=[None, self.x2_ts.shape[1]])

        self.x1_fl_tf = tf.placeholder(tf.float32, shape=[None, self.x1_fl.shape[1]])
        self.x2_fl_tf = tf.placeholder(tf.float32, shape=[None, self.x2_fl.shape[1]])

        self.x1_so_tf = tf.placeholder(tf.float32, shape=[None, self.x1_so.shape[1]])
        self.x2_so_tf = tf.placeholder(tf.float32, shape=[None, self.x2_so.shape[1]])

        self.x1_tf = tf.placeholder(tf.float32, shape=[None, self.x1_so.shape[1]])
        self.x2_tf = tf.placeholder(tf.float32, shape=[None, self.x2_so.shape[1]])
        

        # tf Graphs
        self.phi_x_r_pred, self.phi_m_r_pred,self.phi_x_x1_r_pred, self.phi_m_x1_r_pred, self.phi_x_x2_r_pred, self.phi_m_x2_r_pred = self.net_uv(self.x1_r_tf, self.x2_r_tf)
        self.g_x_r_pred,self.g_m_r_pred = self.net_g_x1(self.x1_r_tf, self.x2_r_tf,'pos')

        self.phi_x_l_pred, self.phi_m_l_pred,self.phi_x_x1_l_pred, self.phi_m_x1_l_pred, self.phi_x_x2_l_pred, self.phi_m_x2_l_pred = self.net_uv(self.x1_l_tf, self.x2_l_tf)
        self.g_x_l_pred,self.g_m_l_pred = self.net_g_x1(self.x1_l_tf, self.x2_l_tf,'neg')

        self.phi_x_t_pred, self.phi_m_t_pred,self.phi_x_x1_t_pred, self.phi_m_x1_t_pred, self.phi_x_x2_t_pred, self.phi_m_x2_t_pred = self.net_uv(self.x1_t_tf, self.x2_t_tf)
        self.g_x_t_pred,self.g_m_t_pred = self.net_g_x2(self.x1_t_tf, self.x2_t_tf,'pos')

        self.phi_x_b_pred, self.phi_m_b_pred,self.phi_x_x1_b_pred, self.phi_m_x1_b_pred, self.phi_x_x2_b_pred, self.phi_m_x2_b_pred = self.net_uv(self.x1_b_tf, self.x2_b_tf)
        self.g_x_b_pred,self.g_m_b_pred= self.net_g_x2(self.x1_b_tf, self.x2_b_tf,'neg')

        self.f_x_pred, self.f_m_pred = self.net_f_uv(self.x1_ts_tf, self.x2_ts_tf)
        self.f_x_fl_pred, self.f_m_fl_pred = self.net_fl_uv(self.x1_fl_tf, self.x2_fl_tf)

        self.f_x_so_pred, self.f_m_so_pred = self.net_so_uv(self.x1_so_tf, self.x2_so_tf)       

        self.phi_x_pred,self.phi_m_pred=self.net_u(self.x1_tf,self.x2_tf) 
        
        # Loss
        self.loss = tf.reduce_mean(tf.square(self.f_x_pred)) + \
                    tf.reduce_mean(tf.square(self.f_m_pred))+ \
                    tf.reduce_mean(tf.square(self.f_x_fl_pred)) + \
                    tf.reduce_mean(tf.square(self.f_m_fl_pred)) + \
                    tf.reduce_mean(tf.square(self.g_x_r_pred)) + \
                    tf.reduce_mean(tf.square(self.g_m_r_pred)) + \
                    tf.reduce_mean(tf.square(self.g_x_l_pred)) + \
                    tf.reduce_mean(tf.square(self.g_m_l_pred)) + \
                    tf.reduce_mean(tf.square(self.g_x_t_pred)) + \
                    tf.reduce_mean(tf.square(self.g_m_t_pred)) + \
                    tf.reduce_mean(tf.square(self.g_x_b_pred)) + \
                    tf.reduce_mean(tf.square(self.g_m_b_pred)) + \
                    self.ga*tf.reduce_mean(tf.square(self.f_x_so_pred))+ \
                    self.ga*tf.reduce_mean(tf.square(self.f_m_so_pred))   
        
        # # Optimizers
        # self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
        #                                                         method = 'L-BFGS-B', 
        #                                                         options = {'maxiter': 50000,
        #                                                                    'maxfun': 50000,
        #                                                                    'maxcor': 50,
        #                                                                    'maxls': 50,
        #                                                                    'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
              
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    def net_u(self, x1,x2):
        X=tf.concat([x1,x2],1)
        uv = self.neural_net(X, self.weights, self.biases)
        u = uv[:,0:1]
        v = uv[:,1:2]
        return u, v

    def net_uv(self, x1, x2):
        X = tf.concat([x1,x2],1)
        
        uv = self.neural_net(X, self.weights, self.biases)
        u = uv[:,0:1]
        v = uv[:,1:2]
        
        u_x1 = tf.gradients(u, x1)[0]
        v_x1 = tf.gradients(v, x1)[0]

        u_x2 = tf.gradients(u, x2)[0]
        v_x2 = tf.gradients(v, x2)[0]

        return u, v, u_x1, v_x1, u_x2, v_x2

    def net_f_uv(self, x1, x2):
        u, v, u_x1, v_x1, u_x2, v_x2  = self.net_uv(x1,x2)
        
        u_xx1 = tf.gradients(u_x1, x1)[0]
        u_xx2 = tf.gradients(u_x2, x2)[0]
        
        v_xx1 = tf.gradients(v_x1, x1)[0]
        v_xx2 = tf.gradients(v_x2, x2)[0]
        
        f_x = -k_x*(u_xx1+u_xx2) + mu_x*u 
        f_m = -k_m*(v_xx1+v_xx2) + mu_m*v - gamma*u
        
        return f_x, f_m

    def net_so_uv(self, x1, x2):
        u, v, u_x1, v_x1, u_x2, v_x2  = self.net_uv(x1,x2)
        
        u_xx1 = tf.gradients(u_x1, x1)[0]
        u_xx2 = tf.gradients(u_x2, x2)[0]
        
        v_xx1 = tf.gradients(v_x1, x1)[0]
        v_xx2 = tf.gradients(v_x2, x2)[0]
        
        f_x = -k_x*(u_xx1+u_xx2) + mu_x*u - q
        f_m = -k_m*(v_xx1+v_xx2) + mu_m*v - gamma*u
        
        return f_x, f_m

    def net_fl_uv(self, x1, x2):
        u, v, u_x1, v_x1, u_x2, v_x2  = self.net_uv(x1,x2)
        
        u_xx1 = tf.gradients(u_x1, x1)[0]
        u_xx2 = tf.gradients(u_x2, x2)[0]
        
        v_xx1 = tf.gradients(v_x1, x1)[0]
        v_xx2 = tf.gradients(v_x2, x2)[0]
        
        f_x = -k_f_x*(u_xx1+u_xx2) + mu_f_x*u 
        f_m = -k_f_m*(v_xx1+v_xx2) + mu_f_m*v - gamma_f*u
        
        return f_x, f_m

    def net_g_x1(self, x1, x2, sign):
        u, v, u_x1, v_x1, u_x2, v_x2  = self.net_uv(x1,x2)
        
        if sign == 'pos':
            g_x = k_x * u_x1 + rho_x*u
            g_m = k_m * v_x1 + rho_m*v
        elif sign== 'neg':
            g_x = -k_x * u_x1 + rho_x*u
            g_m = -k_m * v_x1 + rho_m*v

        return g_x,g_m

    def net_g_x2(self, x1, x2, sign):
        u, v, u_x1, v_x1, u_x2, v_x2  = self.net_uv(x1,x2)
        
        if sign == 'pos':
            g_x = k_x * u_x2 + rho_x*u
            g_m = k_m * v_x2 + rho_m*v
        elif sign== 'neg':
            g_x = -k_x * u_x2 + rho_x*u
            g_m = -k_m * v_x2 + rho_m*v

        return g_x,g_m

  
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        
        tf_dict = {

                   self.x1_r_tf: self.x1_r, self.x2_r_tf: self.x2_r,
                   self.x1_l_tf: self.x1_l, self.x2_l_tf: self.x2_l,
                   self.x1_t_tf: self.x1_t, self.x2_t_tf: self.x2_t,
                   self.x1_b_tf: self.x1_b, self.x2_b_tf: self.x2_b,

                   self.x1_ts_tf: self.x1_ts, self.x2_ts_tf: self.x2_ts,
                   self.x1_fl_tf: self.x1_fl, self.x2_fl_tf: self.x2_fl,
                   self.x1_so_tf: self.x1_so, self.x2_so_tf: self.x2_so,

                   }
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
                                                                                                                          
        """
        self.optimizer_Adam.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        
        """                           
    """
    To do
    """

    def predict(self, X_star):
        tf_dict = {self.x1_tf: X_star[:,0:1], self.x2_tf : X_star[:,1:2]}

        phi_x_pred = self.sess.run(self.phi_x_pred, tf_dict)
        phi_m_pred = self.sess.run(self.phi_m_pred, tf_dict)

        return phi_x_pred,phi_m_pred


        
    #     u_star = self.sess.run(self.u0_pred, tf_dict)  
    #     v_star = self.sess.run(self.v0_pred, tf_dict)  
        
        
    #     tf_dict = {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]}
        
    #     f_u_star = self.sess.run(self.f_u_pred, tf_dict)
    #     f_v_star = self.sess.run(self.f_v_pred, tf_dict)
               
    #     return u_star, v_star, f_u_star, f_v_star
    
if __name__ == "__main__": 
    """
    noise = 0.0        
    
    # Doman bounds
    lb = np.array([-5.0, -5.0])
    ub = np.array([5.0, 5.0])

    N0 = 50
    N_b = 50
    N_f = 20000
    
        
    data = scipy.io.loadmat('../Data/NLS.mat')
    
    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    print(t.shape)

    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)
    
    X, T = np.meshgrid(x,t)
    """
    dim=2
    k=5     #-k to +k in both directions
    #fluorophore position definition
    layers = [2, 100, 100, 100, 100, 2]
    so_pts=[[-k,0],[-k,2],[-k,-2],[-k,4],[-k,-4]]
    v_s=0.25 
    v_x1=0
    v_x2=0
    N_tissue=2000
    t_set=-k+(2*k*lhs(dim,N_tissue))
    for i in range(t_set.shape[0]-1,-1,-1):
        if ((t_set[i,0]<=(v_x1+v_s) and t_set[i,0]>=(v_x1-v_s)) and (t_set[i,1]<=(v_x2+v_s) and t_set[i,0]>=(v_x2-v_s))) or ([t_set[i,0],t_set[i,1]] in so_pts):
            t_set=np.delete(t_set,i,axis=0)
    N_f=1000
    f_set=lhs(dim,N_f)
    f_set[:,0]=v_x1-v_s+(2*v_s*f_set[:,0])
    f_set[:,1]=v_x2-v_s+(2*v_s*f_set[:,1])
    N_b=100

    #Left boundary data
    lb_set=np.random.uniform(low=-k,high=k,size=N_b).reshape(N_b,1)
    x1_lb=-k*np.ones(N_b).reshape(N_b,1)
    lb_set=np.append(x1_lb,lb_set,axis=1)

    #Right boundary data
    rb_set=np.random.uniform(low=-k,high=k,size=N_b).reshape(N_b,1)
    x1_rb=k*np.ones(N_b).reshape(N_b,1)
    rb_set=np.append(x1_rb,rb_set,axis=1)

    #Top Boundary
    ub_set=np.random.uniform(low=-k,high=k,size=N_b).reshape(N_b,1)
    x2_ub=k*np.ones(N_b).reshape(N_b,1)
    ub_set=np.append(ub_set,x2_ub,axis=1)

    #Bottom Boundary
    bb_set=np.random.uniform(low=-k,high=k,size=N_b).reshape(N_b,1)
    x2_bb=-k*np.ones(N_b).reshape(N_b,1)
    bb_set=np.append(bb_set,x2_bb,axis=1)

    X={'r':rb_set,'l':lb_set,'t':ub_set,'b':bb_set,'ts':t_set,'fl':f_set}

    X['so']=np.array(so_pts)

    model = PhysicsInformedNN(X,layers)
    model.train(5)
    x_pred,m_pred=model.predict(rb_set)
    print(x_pred)


    # X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    # u_star = Exact_u.T.flatten()[:,None]
    # v_star = Exact_v.T.flatten()[:,None]
    # h_star = Exact_h.T.flatten()[:,None]
    
#     ###########################
    
#     idx_x = np.random.choice(x.shape[0], N0, replace=False)
#     x0 = x[idx_x,:]
#     u0 = Exact_u[idx_x,0:1]
#     v0 = Exact_v[idx_x,0:1]
    
    # idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    # tb = t[idx_t,:]
    
#     X_f = lb + (ub-lb)*lhs(2, N_f)
            
#     model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub)
             
#     start_time = time.time()                
#     model.train(50000)
#     elapsed = time.time() - start_time                
#     print('Training time: %.4f' % (elapsed))
    
        
#     u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
#     h_pred = np.sqrt(u_pred**2 + v_pred**2)
            
#     error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
#     error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
#     error_h = np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
#     print('Error u: %e' % (error_u))
#     print('Error v: %e' % (error_v))
#     print('Error h: %e' % (error_h))

    
#     U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
#     V_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')
#     H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')

#     FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')
#     FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method='cubic')     
    

    
#     ######################################################################
#     ############################# Plotting ###############################
#     ######################################################################    
    
#     X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
#     X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
#     X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
#     X_u_train = np.vstack([X0, X_lb, X_ub])

#     fig, ax = newfig(1.0, 0.9)
#     ax.axis('off')
    
#     ####### Row 0: h(t,x) ##################    
#     gs0 = gridspec.GridSpec(1, 2)
#     gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
#     ax = plt.subplot(gs0[:, :])
    
#     h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu', 
#                   extent=[lb[1], ub[1], lb[0], ub[0]], 
#                   origin='lower', aspect='auto')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     fig.colorbar(h, cax=cax)
    
#     ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)
    
#     line = np.linspace(x.min(), x.max(), 2)[:,None]
#     ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)
#     ax.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)
#     ax.plot(t[125]*np.ones((2,1)), line, 'k--', linewidth = 1)    
    
#     ax.set_xlabel('$t$')
#     ax.set_ylabel('$x$')
#     leg = ax.legend(frameon=False, loc = 'best')
# #    plt.setp(leg.get_texts(), color='w')
#     ax.set_title('$|h(t,x)|$', fontsize = 10)
    
#     ####### Row 1: h(t,x) slices ##################    
#     gs1 = gridspec.GridSpec(1, 3)
#     gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
#     ax = plt.subplot(gs1[0, 0])
#     ax.plot(x,Exact_h[:,75], 'b-', linewidth = 2, label = 'Exact')       
#     ax.plot(x,H_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
#     ax.set_xlabel('$x$')
#     ax.set_ylabel('$|h(t,x)|$')    
#     ax.set_title('$t = %.2f$' % (t[75]), fontsize = 10)
#     ax.axis('square')
#     ax.set_xlim([-5.1,5.1])
#     ax.set_ylim([-0.1,5.1])
    
#     ax = plt.subplot(gs1[0, 1])
#     ax.plot(x,Exact_h[:,100], 'b-', linewidth = 2, label = 'Exact')       
#     ax.plot(x,H_pred[100,:], 'r--', linewidth = 2, label = 'Prediction')
#     ax.set_xlabel('$x$')
#     ax.set_ylabel('$|h(t,x)|$')
#     ax.axis('square')
#     ax.set_xlim([-5.1,5.1])
#     ax.set_ylim([-0.1,5.1])
#     ax.set_title('$t = %.2f$' % (t[100]), fontsize = 10)
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)
    
#     ax = plt.subplot(gs1[0, 2])
#     ax.plot(x,Exact_h[:,125], 'b-', linewidth = 2, label = 'Exact')       
#     ax.plot(x,H_pred[125,:], 'r--', linewidth = 2, label = 'Prediction')
#     ax.set_xlabel('$x$')
#     ax.set_ylabel('$|h(t,x)|$')
#     ax.axis('square')
#     ax.set_xlim([-5.1,5.1])
#     ax.set_ylim([-0.1,5.1])    
#     ax.set_title('$t = %.2f$' % (t[125]), fontsize = 10)
    
    # savefig('./figures/NLS')  
    

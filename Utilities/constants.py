# Throughout this file x denotes excitation and m denotes emission

# Tissue region
dim=2
mu_a_x_t=0.15
mu_f_x_t=0.05
mu_a_m_t=0.09
mu_f_m_t=0.01
mu_s_x_t=14
mu_s_m_t=10
eta=1
mu_x = mu_a_x_t+mu_f_x_t
mu_m =  mu_a_m_t+mu_f_m_t
k_x =1/(dim*(mu_f_x_t+mu_a_x_t+mu_s_x_t))
k_m =1/(dim*(mu_f_m_t+mu_a_m_t+mu_s_m_t))
gamma = eta*mu_f_x_t
rho_x =0.1695
rho_m = 0.17

# Fluorophore region
mu_a_x_f=0.15
mu_f_x_f=0.15
mu_a_m_f=0.09
mu_f_m_f=0.11
mu_s_x_f=14
mu_s_m_f=10
mu_f_x =mu_a_x_f+mu_f_x_f
mu_f_m = mu_a_m_f+mu_f_m_f
k_f_x = 1/(dim*(mu_f_x_f+mu_a_x_f+mu_s_x_f))
k_f_m = 1/(dim*(mu_f_m_f+mu_a_m_f+mu_s_m_f))
gamma_f = eta*mu_f_x_f 
rho_f_x = 0.1695
rho_f_m = 0.2
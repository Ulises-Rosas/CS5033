
import random
import numpy as np

random.seed(1345)

cov_file = './vcv_betweenAreas.csv'
spps_names = np.loadtxt('./spps_names.csv', dtype=str).tolist()

data = np.loadtxt('./betwee_areas_Xy.csv', delimiter=',')
# n,p = data.shape
# data = np.c_[np.ones(n),data]
X0,y0 = data[:,:-1], data[:,-1]

n,p = X0.shape
num_test = round(0.4*n)


def P_mat(spps, cov_file):

    raw_data = []
    row_spps = []
    with open(cov_file, 'r') as f:
        for i in f.readlines()[1:]:
            line = i.strip().split(',')
            row_spps.append(line[0].replace('"', ''))
            raw_data.append( line[1:] )

    vcv = np.array(raw_data).astype(float)
    idxs = [row_spps.index(i) for i in spps]

    row_aug = vcv[idxs,:]
    vcv_aug = row_aug[:,idxs]

    Q,L,Qt = np.linalg.svd(vcv_aug)
    # inverse of the 
    # square root of lambda
    LH = np.diag( np.sqrt( 1/abs(L) ) )

    return Q.dot(LH).dot(Qt)

def rmse(K, alpha, Y):
    return np.sqrt( np.mean( (K.dot(alpha) - Y)**2 ) )

def distance_matrix(a, b):
    """
    l2 norm squared matrix
    """
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)**2

def RBF_kernel(a, b, gamma):
    """
    Radial Basis Function
    """
    # import sklearn.metrics    
    tmp_rbf = -gamma*distance_matrix(a, b)
    np.exp(tmp_rbf, tmp_rbf) # RBF kernel. Inplace exponentiation
    return tmp_rbf
    # return sklearn.metrics.pairwise.rbf_kernel(a, b, gamma=gamma)

def linear_kernel(a, b):
    """
    Linear Kernel
    """
    return a.dot(b.T)

def opt_alpha(K, y, reg_lam = None):
    # Y = y
    # X = X
    # m = None
    K_Idx = K + reg_lam * np.diag( np.ones( K.shape[0] ) )
    return np.linalg.inv( K_Idx ).dot( y )
    # return np.linalg.pinv( K_Idx ).dot( y )

def gls_error(X, y, pseudo_inv = False):
    if pseudo_inv:
        B_gls = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
    else:
        B_gls = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

    return rmse(X, B_gls, y)

def standard_scale(u):
    return (u - np.mean(u, axis=0))/np.std( u )

P = P_mat(spps_names, cov_file)
X = standard_scale( P.dot(X0) )
y = standard_scale( P.dot(y0) )

reg_lam = 0.00005
reg_gamma = 1e-5

K_linear = linear_kernel( X, X )
alpha_linear = opt_alpha(K_linear, y, reg_lam)

K_rbf = RBF_kernel(X, X, gamma = reg_gamma)
alpha_rbf = opt_alpha(K_rbf, y, reg_lam)

error_rbf = rmse(K_rbf, alpha_rbf, y)
error_lk = rmse(K_linear, alpha_linear, y)
error_gls = gls_error(X, y)

print('RBF kernel = %s' % round(error_rbf, 5))
print('linear kernel = %s' % round(error_lk, 5))
print('GLS = %s' % round(error_gls, 5))

def cv_hype(X, y, max_iter, 
              reg_lambdas, 
              kernel_type = 'linear', 
              gamma = None):

    n,p = X.shape

    if not gamma:
        gamma = 1/2*p

    # kernel_func = linear_kernel
    # max_iter = 15
    # reg_lambdas = np.linspace( 0.001, 0.025, 100, dtype=float )

    training_errors = []
    testing_errors = []

    for reg_lam in reg_lambdas:

        tmp_train = []
        tmp_test = []

        for _ in range(max_iter):

            test_idx  = random.sample(range(n), k = num_test)
            train_idx = list( set(range(n)) - set(test_idx) )

            X_train, X_test = X[train_idx,:], X[test_idx,:]
            y_train, y_test = y[train_idx]  , y[test_idx]

            if kernel_type == 'rbf':
                K_train = RBF_kernel( X_train, X_train, gamma )
                K_test = RBF_kernel(  X_test,  X_train, gamma )
            

            elif kernel_type == 'linear':
                K_train = linear_kernel( X_train, X_train)
                K_test  = linear_kernel( X_test, X_train )

            else:
                try:
                    gls_err_train = gls_error(X_train, y_train)
                    gls_err_test  = gls_error(X_test, y_test)

                except np.linalg.LinAlgError:
                    gls_err_train = gls_error(X_train, y_train, pseudo_inv=True)
                    gls_err_test  = gls_error(X_test, y_test, pseudo_inv=True)

                tmp_train.append( gls_err_train )
                tmp_test.append( gls_err_test )
                continue

            alpha_trained = opt_alpha(K_train, y_train, reg_lam)

            tmp_train.append( rmse(K_train, alpha_trained, y_train) )
            tmp_test.append( rmse(K_test , alpha_trained, y_test) )

        training_errors.append( np.mean(tmp_train) )
        testing_errors.append( np.mean(tmp_test) )

    return training_errors, testing_errors

def cv_errors(X, y, max_iter, 
              reg_lambda, 
              kernel_type = 'linear', 
              gamma = None):

    n,p = X.shape

    if not gamma:
        gamma = 1/2*p

    # kernel_func = linear_kernel
    # max_iter = 15
    # reg_lambdas = np.linspace( 0.001, 0.025, 100, dtype=float )

    training_errors = []
    testing_errors = []

    for _ in range(max_iter):

        test_idx  = random.sample(range(n), k = num_test)
        train_idx = list( set(range(n)) - set(test_idx) )

        X_train, X_test = X[train_idx,:], X[test_idx,:]
        y_train, y_test = y[train_idx]  , y[test_idx]

        if kernel_type == 'rbf':
            K_train = RBF_kernel( X_train, X_train, gamma )
            K_test = RBF_kernel(  X_test,  X_train, gamma )
        

        elif kernel_type == 'linear':
            K_train = linear_kernel( X_train, X_train)
            K_test  = linear_kernel( X_test, X_train )

        else:
            try:
                gls_err_train = gls_error(X_train, y_train)
                gls_err_test  = gls_error(X_test, y_test)

            except np.linalg.LinAlgError:
                gls_err_train = gls_error(X_train, y_train, pseudo_inv=True)
                gls_err_test  = gls_error(X_test, y_test, pseudo_inv=True)

            training_errors.append( gls_err_train )
            testing_errors.append( gls_err_test )
            continue

        alpha_trained = opt_alpha(K_train, y_train, reg_lam)

        training_errors.append( rmse(K_train, alpha_trained, y_train) )
        testing_errors.append( rmse(K_test , alpha_trained, y_test) )

    return training_errors, testing_errors



max_iter = 20
reg_lambdas_linear = np.linspace( 1e-10, 1e-6, 100, dtype = float )
# reg_lambdas_linear = np.linspace( 5e-13, 1e-10, 100, dtype = float )
(train_error_lk, 
 test_error_lk ) = cv_hype(X, y, max_iter = max_iter , 
                            reg_lambdas = reg_lambdas_linear, 
                            kernel_type = 'linear',
                            gamma = None)


reg_gamma = 1e-5
# reg_lambdas_rbf = np.linspace( 1e-5, 1e-4, 100, dtype = float )
reg_lambdas_rbf = np.linspace( 1e-5, 5e-3, 100, dtype = float )
(train_error_rbf, 
  test_error_rbf ) = cv_hype(X, y, max_iter = max_iter , 
                            reg_lambdas = reg_lambdas_rbf, 
                            kernel_type = 'rbf', 
                            gamma = reg_gamma)

(train_error_gls, 
  test_error_gls ) = cv_hype(X, y, max_iter = max_iter , 
                            reg_lambdas = reg_lambdas_rbf, 
                            kernel_type = 'gls', 
                            gamma = None)




"""
Copyright (c) 2019 Olga Mula, JA
"""

import numpy as np

def ODEsolve2(mesh, f, y0, solve_method='RK4', backward = False):
    """
    Integrates y' =f(y) with y(0) = y0
    where y0 is an aribtrary array and spits out all_x and all_xp of size [N, d, K] where N is the time steps
    
    """
    d, K = y0.shape # assume y0 is d x K with K>=1
    N = mesh.n
    all_x  = np.zeros(( N, d, K ))
    all_xp = np.zeros(( N, d, K ))

    tmin, tmax = mesh.points[0], mesh.points[-1]    

    # Initial conditions
    initvalue = y0
    if backward:
        all_x[N-1, :, :]   = y0     # fix at N-1
        all_xp[N-1, : , :] = f(tmax, y0)  # fix at N-1
    else:        
        all_x[0, :, :]   = y0     # fix at N-1
        all_xp[0, : , :] = f(tmin, y0)  # fix at N-1
    
    
    # Explicit Euler 
    # ------------------------------------------------------ #

    if solve_method == 'euler':
        
        # Backward
        if backward:
            
            for i in reversed(range(N-1)):
                t        = mesh.points[i]    # ti
                t_right  = mesh.points[i+1]  # ti+1
                h = np.abs(t - t_right)
                all_x[i, :, :] = all_x[i+1, :, :] - h * f(t_right, all_x[i+1, :, :])
                all_xp[i, :, :] = f(t, all_x[i, :, : ])

        # Forward
        else:
            
            for i in range(1,N):
                t        = mesh.points[i]    # ti
                t_left  = mesh.points[i-1]  # ti-1
                h = np.abs(t - t_left)
                all_x[i, :, :] = all_x[i-1, :, :] + h * f(t_left, all_x[i-1, :, :])
                all_xp[i, :, :] = f(t, all_x[i, :, : ])


    # Non adaptive RK4
    # ------------------------------------------------------ #		      

    elif solve_method == 'RK4':

        # RK 4 values
        gamma = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
        alpha = np.array([ 0, 1/4, 3/8, 12/13, 1, 1/2 ])
        beta = np.array([[	      0,	  0,	      0,	 0,	 0, 0],
                         [	    1/4,	  0,	      0,	 0,	 0, 0],
                         [	   3/32,       9/32,	      0,	 0,	 0, 0],
                         [    1932/2197, -7200/2197,  7296/2197,	 0,	 0, 0],
                         [	439/216,	 -8,   3680/513, -845/4104,	 0, 0],
                         [	  -8/27,	  2, -3544/2565, 1859/4104, -11/40, 0]])
        m = len(gamma)


        # Backward
        if backward:

            for i in reversed(range(N-1)):
                t        = mesh.points[i]    # ti
                t_right  = mesh.points[i+1]  # ti+1
                h = np.abs(t_right - t)

                # Compute Ks
                Ks = [f(t_right, initvalue)]
                for j in range(1, m):
                    Kj = f(t_right - alpha[j]*h, initvalue - h*sum(b*K for b, K in zip(beta[j,:], Ks)))
                    Ks.append(Kj)

                y_step = initvalue - h * sum(g*K for g,K in zip(gamma, Ks))
                all_x[i, :, :]  = y_step
                all_xp[i, :, :] = f(t, y_step)
                initvalue = y_step

        # Forward
        else:

            for i in range(1, N):
                t       = mesh.points[i]    # ti
                t_left  = mesh.points[i-1]  # ti+1
                h = np.abs(t - t_left)

                # Compute Ks
                Ks = [f(t_left, initvalue)]
                for j in range(1, m):
                    Kj = f(t_left + alpha[j]*h, initvalue + h*sum(b*K for b, K in zip(beta[j,:], Ks)))
                    Ks.append(Kj)

                y_step = initvalue + h * sum(g*K for g,K in zip(gamma, Ks))
                all_x[i, :, :]  = y_step
                all_xp[i, :, :] = f(t, y_step)
                initvalue = y_step
                
    return all_x, all_xp
            


    # Not tested yet
    # 
    # # Adaptive RK45
    # # ------------------------------------------------------ #		      

    # if solve_method == 'RK45' or solve_method == 'DOP853':
        
    #     from scipy.integrate import ode, solve_ivp
        
    #     # Backward
    #     if backward:
    #         tmax = mesh.points[-1]
    #         piecewise_y.append([tmax, initvalue])
    #         piecewise_yp.append([tmax, f(tmax, initvalue)])

    #         for interval in reversed(mesh.intervals):
    #             soln = solve_ivp(f,		   \
        #                              np.flip(interval),		   \
        #                              initvalue, \
        #                              # jac = self.ode.jac, \
        #                              method = solve_method)

    #             for i in range(1, len(soln.t)):
    #                 piecewise_y.append([soln.t[i], soln.y[:,i] ])
    #                 piecewise_yp.append([soln.t[i], f(soln.t[i], soln.y[:,i])])

    #             initvalue = soln.y[:,-1]

    #     # Forward
    #     else:
    #         tmin = mesh.points[0]
    #         piecewise_y.append([tmin, initvalue])	    
    #         piecewise_yp.append([tmin, f(tmin, initvalue)])	  
    #         for interval in mesh.intervals:
    #             soln = solve_ivp(f,	  \
        #                              interval,		  \
        #                              initvalue, \
        #                              # jac = self.ode.jac, \
        #                              method = solve_method)

    #             for i in range(1, len(soln.t)):		       
    #                 piecewise_y.append([soln.t[i], soln.y[:,i]])
    #                 piecewise_yp.append([soln.t[i], f(soln.t[i], soln.y[:,i])])

    #             initvalue = soln.y[:,-1]

    # return piecewise_y, piecewise_yp # list 

import cvxpy as cp
import numpy as np
from scipy import integrate

def CarModel(t, x, Student_Controller, param):
    
    if t <= param["switch_time"]:
        param["v0"] = param["v01"]
    if t > param["switch_time"]:
        param["v0"] = param["v02"]
#the switch time is used to decide when to update the initial velocity of the vehicle model
    
    ## student need to complete this function ##
    A, b, P, q = Student_Controller(t, x, param)
    ## student need to complete this function ##
    
    var = cp.Variable(2)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(var, P)+ q.T @ var),
                     [A @ var <= b])
    prob.solve()
    
    u = var.value[0]        
    u = np.clip(u, -param["Cdg"] * param["m"], param["Cag"] * param["m"])
    
    dx = np.array([param["v0"] - x[1], 
                   u / param["m"]])
    return dx

def sim_vehicle(Student_Controller, param, y0):
    t0, t1 = 0, param["terminal_time"]                # start and end
    t = np.linspace(t0, t1, 200)  # the points of evaluation of solution
    # y0 = [250, 10]                   # initial value
    y = np.zeros((len(t), len(y0)))   # array for solution
    y[0, :] = y0
#Sets up the time grid (t) and initializes an array y to store the solution trajectory. It also initializes the first entry of y with the initial state y0.
    
    r = integrate.ode( lambda t, x:CarModel(t, x, Student_Controller, param) ).set_integrator("dopri5")  # choice of method
    r.set_initial_value(y0, t0)   # initial values
#Defines an ODE solver using the ode function from scipy.integrate. It sets the initial state and the function (CarModel) to integrate.
    
    for i in range(1, t.size):
       y[i, :] = r.integrate(t[i]) # get one more value, add it to the array
       if not r.successful():
           raise RuntimeError("Could not integrate")
#Integrates the system dynamics using the ODE solver over the specified time grid, storing the results in the y array.    
    ### recover control input ###
    u = np.zeros((200, 1))
    for k in range(200):
        if t[k] <= param["switch_time"]:
            param["v0"] = param["v01"]
        if t[k] > param["switch_time"]:
            param["v0"] = param["v02"]
            
        A, b, P, q = Student_Controller(t[k], y[k, :], param)
        var = cp.Variable(2)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(var, P)+ q.T @ var),
                         [A @ var <= b])
        prob.solve()

        u[k] = var.value[0]
    ### recover control input ###
#Computes the control input u for each time step by solving a quadratic optimization problem using CVXPY.
    v0 = t * 0
    v0[t <  param["switch_time"]] = param["v01"]
    v0[t >= param["switch_time"]] = param["v02"]
    Cdg = param["Cdg"]
    B   = y[:, 0] - 1.8 * y[:, 1] - 0.5 * (np.clip(y[:, 1] - v0, 0, np.inf))**2 / Cdg

    return t, B, y, u

#This code simulates a vehicle's behavior using an ODE solver, where the control input is determined by solving a quadratic optimization problem at each time step.

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
class C:
    T = 0.1
    RF = 4.5
    RB = 1.0
    W = 3.0
    WD = 0.7 * W
    WB = 3.5
    LF = 1.5
    LR = WB - LF
    TR = 0.5
    TW = 1
    MAX_STEER = 0.6
    MAX_SPEED = 10/3.6
    MIN_SPEED = 0
    MAX_ACC = 1.5
    MAX_STEERING_RATE = np.deg2rad(20)

class NMPC:
    def __init__(self):
        self.WheelBase = 2.74
        self.dt = 0.1
        self.N = 25
        self.Q = np.diag([1,1,0.5,10,0]) # x,y,yaw,v,steer
        self.R = np.diag([1,1]) # a, steering_rate

        self.BuildModel()
        self.generate_variable()
        self.generate_constraint()
        self.generate_obj()
        self.generate_solver()
        self.ref = False

    def BuildModel(self):
        '''
        func:build the nolinear model and construct the variable
        :return:
        '''
        # 构造符号变量
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        v = ca.SX.sym('v')
        steering = ca.SX.sym('steering')
        a = ca.SX.sym('a')
        # 步长t
        steering_rate = ca.SX.sym('steering_rate')
        self.obj = 0
        self.n_states = 5
        self.n_controls = 2
        self.state = ca.vertcat(x, y, v, theta, steering)
        # control:[acc, delta_f]
        self.control = ca.vertcat(a, steering_rate)

        self.rhs = ca.vertcat(v * ca.cos(theta) ,
                              v * ca.sin(theta) ,
                              a ,
                              v / C.WB * ca.tan(steering) ,
                              steering_rate ) * self.dt
        # 构建f = AX+BU
        self.f = ca.Function('f', [self.state, self.control], [self.rhs])
        # 构建状态量集合 N个时刻 每个时刻5个状态量
        self.X = ca.SX.sym('X', self.n_states, self.N)
        # 构建控制量集合 N-1个间隔， 每个时刻2个控制量
        self.U = ca.SX.sym('U', self.n_controls, self.N - 1)
        self.P = ca.SX.sym('P', self.n_states, self.N)
    def generate_obj(self):
        '''
        func: generate the objective func of path smoother
        :return:
        '''
        R = ca.SX(self.R)
        Q = ca.SX(self.Q)
        self.obj = 0
        for i in range(self.N):
            # state x,y,yaw,v,steering
            st = self.X[:, i]
            # ref_state
            ref_st = self.P[:,i]
            error = st - ref_st

            #cost func  control^T * R * control + (state - ref_state)*Q*(state - ref_state)
            self.obj += (error.T@Q@error)
            if i < self.N -1 :
                con = self.U[:, i]
                self.obj += (con.T @ R @ con)

        self.vio = 0
        self.vio_coe = 1e4
        for i in range(0, self.N - 1):
            st = self.X[:, i]
            st_ = self.X[:, i + 1]
            con = self.U[:, i]
            f_value = self.f(st, con)
            self.vio += ((st_ - st) - f_value).T @ ((st_ - st) - f_value) * self.vio_coe
        self.obj += self.vio

    def generate_variable(self):
        '''
        func: generate the raw variable to be optimized
        :return:
        '''
        self.variable,self.lbx,self.ubx = [], [], []
        for i in range(self.N):
            # 一阶控制约束
            self.variable += [self.X[:, i]]
            self.lbx += [-np.inf, -np.inf, - C.MAX_SPEED, -40 * np.pi, -C.MAX_STEER]
            self.ubx += [np.inf, np.inf, C.MAX_SPEED, 40 * np.pi, C.MAX_STEER]

        for i in range(self.N - 1):
            # 二阶控制约束
            self.variable += [self.U[:, i]]
            self.lbx += [-C.MAX_ACC, -C.MAX_STEERING_RATE]
            self.ubx += [ C.MAX_ACC,  C.MAX_STEERING_RATE]

    def generate_constraint(self):
        '''
        func: generative the constraint of smooth term
        :return:
        todo: replace the nonlinear curvature constraint with linear constraint
        '''
        self.constrains,self.lbg,self.ubg = [],[],[]
        # 增加起始位置约束
        self.constrains += [self.X[:, 0] - self.P[:,0]]
        self.lbg += [0, 0, 0, 0, 0]
        self.ubg += [0, 0, 0, 0, 0]

    def generate_solver(self):
        self.nlp_prob = {'f': self.obj,
                         'x': ca.vertcat(*self.variable),
                         'g': ca.vertcat(*self.constrains),
                         'p': self.P}
        opts_setting = {'ipopt.max_iter': 4000,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-3,
                        'ipopt.acceptable_obj_change_tol': 1e-3,
                        # "jit": True,
                        # "verbose": True,
                        # "linear_solver" : "ma57"
                        }
        self.solver = ca.nlpsol("solver", 'ipopt', self.nlp_prob,opts_setting)

    def Solve(self,ref_p,ref_u):
        x_init = ref_p.flatten()
        # u_init = ref_u.flatten() if self.ref == False else self.ref_control.flatten()
        u_init = ref_u.flatten()
        sol = self.solver(x0=np.hstack((x_init, u_init)),
                          lbx=ca.vertcat(*self.lbx),
                          ubx=ca.vertcat(*self.ubx),
                          lbg=ca.vertcat(*self.lbg),
                          ubg=ca.vertcat(*self.ubg),
                          p=ref_p.T
                          )
        res = sol['x']
        self.x_opt = res[0:self.n_states * (self.N):self.n_states].full().flatten()
        self.y_opt = res[1:self.n_states * (self.N):self.n_states].full().flatten()
        self.v_opt = res[2:self.n_states * (self.N):self.n_states].full().flatten()
        self.theta_opt = res[3:self.n_states * (self.N):self.n_states].full().flatten()
        self.steer_opt = res[4:self.n_states * (self.N):self.n_states].full().flatten()
        self.a_opt = res[
                     self.n_states * (self.N):self.n_states * (self.N) + self.n_controls * (self.N - 1):self.n_controls].full().flatten()
        self.steerate_opt = res[self.n_states * (self.N) + 1:self.n_states * (self.N) + self.n_controls * (
                    self.N - 1):self.n_controls].full().flatten()

        self.ref_state = np.vstack((self.x_opt,self.y_opt,self.theta_opt,self.v_opt,self.steer_opt)).T
        self.ref_control = np.vstack((np.vstack((self.a_opt,self.steerate_opt)).T[1:],[0,0]))
        self.ref = True
        plt.plot(self.x_opt,self.y_opt)


if __name__ == '__main__':
    NMPC_controller = NMPC()
    ref_p = np.array([0,0,0,0,0] + [10,10,0,0,0] * (NMPC_controller.N - 1)).reshape(NMPC_controller.N,5)
    ref_u = np.zeros((2, NMPC_controller.N - 1))
    NMPC_controller.Solve(ref_p, ref_u)
    plt.show()

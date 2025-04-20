import casadi as ca
import numpy as np

class MPC_Controller:
    def __init__(self):
        self.WheelBase = 2.74
        self.dt = 0.1
        self.N = 15
        self.BuildModel()
        self.GetCostFunction()
        self.GenerateVariable()
        self.GenerateConstraint()
        self.ConstructSolver()
        self.Res = None
    def BuildModel(self):
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        yaw = ca.SX.sym('yaw')
        v = ca.SX.sym('v')
        a = ca.SX.sym('a')
        steer = ca.SX.sym('steer')
        self.state = ca.vertcat(x,y,yaw,v)
        self.control = ca.vertcat(a,steer)

        A,B,C = self.GetLinearMatrix(v,yaw,steer)

        self.rhs_linear = ca.Function("LinearizeModel",[self.state,self.control],[A,B,C])

        self.Q = ca.diag([1.0, 1.0, 1.0, 1.0])
        self.R = ca.diag([0.1,0.1])

        self.v_max = 5
        self.v_min = 0
        self.max_steer = np.deg2rad(30)

        # 构建状态量集合 N个时刻 每个时刻5个状态量
        self.X = ca.MX.sym('X', self.state.shape[0], self.N)
        # 构建控制量集合 N-1个间隔， 每个时刻2个控制量
        self.U = ca.MX.sym('U', self.control.shape[0], self.N - 1)
        # 构建参考状态集合
        self.P = ca.MX.sym('P', self.state.shape[0],self.N)
    def GetLinearMatrix(self,v,phi,delta):
        A = np.array([[1.0, 0.0, self.dt * ca.cos(phi), - self.dt * v * ca.sin(phi)],
                      [0.0, 1.0, self.dt * ca.sin(phi), self.dt * v * ca.cos(phi)],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, self.dt * ca.tan(delta) / self.WheelBase, 1.0]])

        B = np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [self.dt, 0.0],
                      [0.0, self.dt * v / (self.WheelBase * ca.cos(delta) ** 2)]])

        C = np.array([self.dt * v * ca.sin(phi) * phi,
                      -self.dt * v * ca.cos(phi) * phi,
                      0.0,
                      -self.dt * v * delta / (self.WheelBase * ca.cos(delta) ** 2)])

        return A, B, C
    def GetCostFunction(self):
        self.Cost = 0
        for i in range(self.N - 1):
            control = self.U[:,i]
            states = self.X[:,i]
            ref_state = self.P[:,i]
            self.Cost += control.T @ self.R @ control
            delta_ = ref_state - states
            self.Cost += delta_.T @ self.Q @ delta_
    def GenerateVariable(self):
        self.variable = []
        self.lbx = []
        self.ubx = []
        for i in range(self.N):
            # 一阶控制约束
            self.variable += [self.X[:, i]]
            self.lbx += [-np.inf, -np.inf, -np.inf,self.v_min]
            self.ubx += [ np.inf,  np.inf,  np.inf,self.v_max]

        for i in range(self.N - 1):
            # 二阶控制约束
            self.variable += [self.U[:, i]]
            self.lbx += [self.v_min, -self.max_steer]  # v and theta lower bound
            self.ubx += [self.v_max,  self.max_steer]  # v and theta upper bound
    def GenerateConstraint(self):
        state = self.X[:,0]
        self.g = [state - self.P[:,0]]
        for i in range(self.N - 1):
            state = self.X[:,i]
            state_next = self.X[:,i+1]
            control = self.U[:,i]
            A,B,C = self.rhs_linear(state,control)
            LinearModelNextState = A @ state + B @ control + C
            self.g += [state_next - LinearModelNextState]
        self.lbg = [0,0,0,0] * len(self.g)
        self.ubg = [0,0,0,0] * len(self.g)
    def ConstructSolver(self):
        OPT_variables = ca.vertcat(ca.reshape(self.X, 4 * self.N, 1),
                                   ca.reshape(self.U, 2 * (self.N - 1), 1))
        self.nlp_prob = {'f': self.Cost,
                         'x': OPT_variables,
                         'g': ca.vertcat(*self.g),
                         'p': ca.vertcat(self.P)}
        # self.nlp_prob = {'f': self.Cost,
        #                  'x': ca.vertcat(*self.variable),
        #                  'g': ca.vertcat(*self.g),
        #                  'p': ca.vertcat(self.P)}
        self.solver = ca.nlpsol("solver",'osqp',self.nlp_prob)
    def Solve(self,ref_p,ref_u):
        # if self.Res == None:
        #     x_init = ca.vertcat(ca.reshape(self.X.T, 4 * self.N, 1),
        #                         ca.reshape(self.U.T, 2 * (self.N - 1), 1))
        # else:
        #     x_init = self.Res
        x_init = ref_p.flatten()
        u_init = ref_u.flatten()
        sol = self.solver(x0 = np.hstack((x_init,u_init)),
                          lbx=ca.vertcat(*self.lbx),
                          ubx=ca.vertcat(*self.ubx),
                          lbg=ca.vertcat(*self.lbg),
                          ubg=ca.vertcat(*self.ubg),
                          p  =ref_p
                          )
        self.X0 = ca.reshape(sol['x'][0:4 * self.N ], 4,self.N).T  # get soln trajectory
        u = ca.reshape(sol['x'][4 * self.N:], 2, self.N - 1).T  # get controls solution




if __name__ == '__main__':

    Controller = MPC_Controller()
    ref_p = np.zeros((4, Controller.N))
    ref_u = np.zeros((2, Controller.N - 1))
    Controller.Solve(ref_p,ref_u)
    
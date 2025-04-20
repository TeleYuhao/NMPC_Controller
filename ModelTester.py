import matplotlib.pyplot as plt
from NMPC import NMPC
import math
import  numpy as np
from draw import draw_car
from CubicSpline import Spline2D
import pandas as pd

def Checkyaw(yaw):
    YawChecked = [yaw[0]]
    for i in range(1,len(yaw)):
        if abs(yaw[i] - YawChecked[-1]) > np.pi:
            if yaw[i] > YawChecked[-1]:
                YawChecked.append(yaw[i] - 2 * np.pi)
            else:
                YawChecked.append(yaw[i] + 2 * np.pi)
        else:
            YawChecked.append(yaw[i])
    return YawChecked
def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi

    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle



class Node:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, steer = 0,v=0.0, direct=1.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.steer = steer
        self.direct = direct
        self.dt = 0.1
        self.WB = 2.74

    def update(self, a, steerate, direct):
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        self.yaw += self.v / self.WB * math.tan(self.steer) * self.dt
        self.direct = direct
        self.v += self.direct * a * self.dt
        self.steer += steerate * self.dt

def round_to_half(num):
    abs_num = abs(num)
    rounded = math.floor(abs_num * 2 + 0.5) / 2
    return math.copysign(rounded, num)

def CalcRefPath(ref_path:Spline2D,state:np.array):
    s = ref_path.cartesian_to_frenet1D(state[:2])
    v = state[2]
    ref = [[state[0],state[1],state[2],state[3],state[4]]]
    ref_v = 15/3.6
    dt = 0.1
    for i in range(24):
        next_s = round_to_half(s[0] + max(v,0.25) * i * dt)
        next_s = next_s if next_s < ref_path.s[-1] else ref_path.s[-1]
        ref_v = ref_v if next_s < ref_path.s[-1] else 0
        pos = ref_path.calc_position(next_s)
        yaw = ref_path.calc_yaw(next_s)
        ref.append([pos[0],pos[1],ref_v,yaw,0])
    ref = np.array(ref)
    ref[:,3] = Checkyaw(ref[:,3])
    return ref



if __name__ == '__main__':

    ax = [0.0, 15.0, 30.0, 50.0, 60.0]
    ay = [0.0, 40.0, 15.0, 30.0, 0.0]
    # ref_path = Spline2D(ax,ay)
    path = np.array(pd.read_csv("downsampled_gauss_path2.csv"))[:-20]

    ref_path = Spline2D(path[:,0],path[:,1])
    p = ref_path.GetPath()

    s = 0
    position_init = ref_path.calc_position(s)
    state = [position_init[0],position_init[1],0,ref_path.calc_yaw(s),0]
    ref = CalcRefPath(ref_path,state)

    # plt.plot(p[:,0],p[:,1])
    # plt.plot(ref[:, 0], ref[:, 1])
    # plt.show()
    veh = Node(x=state[0], y=state[1], yaw=state[3], steer=0, v=state[2])
    traj_x = [veh.x]
    traj_y = [veh.y]

    Controller = NMPC()
    ref_u = np.zeros((Controller.N - 1, 2))
    for i in range(1000):
        plt.plot(p[:, 0], p[:, 1])
        state = [veh.x,veh.y,veh.v,veh.yaw,veh.steer]
        ref = CalcRefPath(ref_path, state)
        Controller.Solve(ref,ref_u)
        veh.update(Controller.a_opt[0],Controller.steerate_opt[0],direct=1)
        traj_x.append(veh.x)
        traj_y.append(veh.y)
        plt.plot(traj_x,traj_y)
        draw_car(veh.x,veh.y,veh.yaw,-veh.steer)
        # print(veh.x,veh.y,veh.yaw,veh.steer)
        plt.pause(0.01)
        plt.cla()







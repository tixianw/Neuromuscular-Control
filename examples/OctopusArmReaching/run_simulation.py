import numpy as np

# import matplotlib.pylab as plt
from tqdm import tqdm
from elastica import *

# from set_arm_environment import ArmEnvironment
from set_environment import Environment

# from neuralctrl_Matsuoka import NeuralCtrl
from neuromuscular_control.neuron_models import NeuralControlMatsuoka

# from neuralctrl_FitzHugh_Nagumo import NeuralCtrl
# from sensoryfeedback import SensoryFeedback
from neuromuscular_control.feedback_controller import SensoryFeedback

# from neuromuscular_control.utils import gaussian

s = np.linspace(0, 0.2, 101)


def data_save(env, controller, sensor=None, desired=None):
    model = {
        "flags": env.flags,
        "numerics": env.numeric_param,
        "arm": env.arm_param,
        "neuron": controller.neuron_param,
        "sensor_mag": controller.mu,
    }
    if env.flags[1]:
        model.update({"target": env.target[:: env.step_skip, :]})
    if env.flags[2]:
        model.update({"obstacle": env.obstacle_param})
    arm = []
    muscle = []
    neuron = []
    sensor_data = []
    arm.append(
        {
            "position": np.array(env.pp_list["position"])[:, :, :],
            "orientation": np.array(env.pp_list["orientation"])[:, :, :, :],
            "velocity": np.array(env.pp_list["velocity"])[:, :2, :],
            "omega": np.array(env.pp_list["angular_velocity"])[:, 0, :],
            "kappa": -np.array(env.pp_list["kappa"])[:, 0, :]
            / np.array(env.pp_list["voronoi_dilatation"])[:, :],
            "nu1": np.array(env.pp_list["strain"])[:, -1, :] + 1,
            "nu2": np.array(env.pp_list["strain"])[:, 1, :],
        }
    )
    muscle.append(
        {
            "u": np.array(env.muscle_list["u"]),
        }
    )
    neuron.append(
        {
            "I": np.array(controller.callback_list["I"]),
            "V": np.array(controller.callback_list["V"]),
        }
    )
    if sensor != None:
        sensor_data.append(
            {
                "dist": np.array(sensor.callback_list["dist"]),
                "angle": np.array(sensor.callback_list["angle"]),
                "s_bar": np.array(sensor.callback_list["s_bar"]),
            }
        )

    data = {
        "t": np.array(env.pp_list["time"]),
        "model": model,
        "arm": arm,
        "muscle": muscle,
        "neuron": neuron,
        "sensor": sensor_data,
        "desired": desired,
    }
    np.save("Data/test.npy", data)


def get_activation(
    time, systems, controller=None, desired_curvature=None, desired_activation=None
):
    # activation = np.zeros([3,systems[0].n_elems+1])
    activation = controller.neural_ctrl(
        time, systems[0], desired_curvature, desired_activation
    )
    return activation


def main(filename):
    ### Create arm and simulation environment
    final_time = 2.0
    flag_shooting = 1
    flag_target = True  # False #
    flag_obstacle = False  # True #
    flags = [flag_shooting, flag_target, flag_obstacle]

    # env = ArmEnvironment(final_time, flags)
    env = Environment(final_time, flags)
    total_steps, systems = env.reset()

    ### Create neural muscular controller
    neural_list = defaultdict(list)
    sensor_list = defaultdict(list)
    controller = NeuralControlMatsuoka(env, neural_list, env.step_skip)
    sensor = SensoryFeedback(env, sensor_list, env.step_skip)

    # ### Desired muscle activation or curvature
    # # u = np.vstack([gaussian(s, mu=0.1, sigma=0.02, magnitude=0.2), np.zeros([2, len(s)])])
    # V = np.vstack([gaussian(s, mu=0.1, sigma=0.02, magnitude=80), np.zeros([2, len(s)])])
    # u = controller.v_to_u(V)
    # desired_kappa = gaussian(s[1:-1], mu=0.1, sigma=0.02, magnitude=20)
    # desired = {
    # 	'desired_u': u,
    # 	# 'desired_kappa': desired_kappa,
    # }

    ### Start the simulation
    print("Running simulation ...")
    time = np.float64(0.0)
    for k_sim in tqdm(range(total_steps)):
        target = env.target[k_sim, :]
        u = sensor.sensory_feedback_law(time, systems[0], target)
        activation = get_activation(
            time,
            systems,
            controller=controller,
            # desired_curvature=desired_kappa,
            desired_activation=u,
        )
        time, systems, done = env.step(time, activation)
        if done:
            break

    data_save(env, controller, sensor)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument(
        "--filename",
        type=str,
        default="simulation",
        help="a str: data file name",
    )
    args = parser.parse_args()
    main(filename=args.filename)

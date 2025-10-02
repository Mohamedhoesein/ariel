"""Assignment 3 template code."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import random as rd
from multiprocessing import Pool

# Third-party libraries
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

from Neural_Net import Brain, Layer

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)


def show_xpos_history(history: list[float]) -> None:
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Create figure and axis
    plt.figure(figsize=(10, 6))

    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], "b-", label="Path")
    plt.plot(pos_data[0, 0], pos_data[0, 1], "go", label="Start")
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", label="End")
    plt.plot(0, 0, "kx", label="Origin")

    # Add labels and title
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Robot Path in XY Plane")
    plt.legend()
    plt.grid(visible=True)

    # Set equal aspect ratio and center at (0,0)
    plt.axis("equal")

    # Show results
    #plt.show()
    plt.savefig("./__data__/test.png")


def random_move(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    # Get the number of joints
    num_joints = model.nu

    # Hinges take values between -pi/2 and pi/2
    hinge_range = np.pi / 2
    return RNG.uniform(
        low=-hinge_range,  # -pi/2
        high=hinge_range,  # pi/2
        size=num_joints,
    ).astype(np.float64)


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu

    # Initialize the networks weights randomly
    # Normally, you would use the genes of an individual as the weights,
    # Here we set them randomly for simplicity.
    w1 = RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
    w2 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
    w3 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # Run the inputs through the lays of the network.
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))

    # Scale the outputs
    return outputs * np.pi


def experiment(
    robot: Any,
    controller: Controller,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.spec, spawn_position=[0, 0, 0.1])

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )

    # ------------------------------------------------------------------ #
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(
                model,
                data,
                duration=duration,
            )
        case "frame":
            # Render a single frame (for debugging)
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
            )
        case "launcher":
            # This opens a liver viewer of the simulation
            viewer.launch(
                model=model,
                data=data,
            )
        case "no_control":
            # If mj.set_mjcb_control(None), you can control the limbs manually.
            mj.set_mjcb_control(None)
            viewer.launch(
                model=model,
                data=data,
            )
    # ==================================================================== #

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def get_input_output_sizes(robot_graph) -> tuple[int, int]:
    core = construct_mjspec_from_graph(robot_graph)
    world = OlympicArena()

    world.spawn(core.spec, spawn_position=[0, 0, 0.1])

    model = world.spec.compile()
    data = mj.MjData(model)
    return len(data.qpos), len(data.ctrl)

def learn_brain(genotypes):
    mj.set_mjcb_control(None)

    num_modules = 20
    robot_graph = genotypes_to_phenotypes(genotypes, num_modules)
    
    input_size, output_size = get_input_output_sizes(robot_graph)

    population_size = 64
    max_gens = 100
    population = [
        Brain(
            [
                Layer(input_size, 50, sigmoid),
                Layer(50, 30, sigmoid),
                Layer(30, output_size, sigmoid),
            ],
            mutation_rate=rd.random(),
        ).random()
        for _ in range(population_size)
    ]

    fitness = np.zeros((max_gens, len(population)))
    best_brain = population[0]

    for gen in range(max_gens):
        for controller in population:
            core = construct_mjspec_from_graph(robot_graph)
            mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
            name_to_bind = "core"
            tracker = Tracker(
                mujoco_obj_to_find=mujoco_type_to_find,
                name_to_bind=name_to_bind,
            )

            ctrl = Controller(
                controller_callback_function=controller.control,
                # controller_callback_function=random_move,
                tracker=tracker,
            )

            experiment(robot=core, controller=ctrl, mode="simple")

            #show_xpos_history(tracker.history["xpos"][0])
            controller.set_history(tracker.history["xpos"][0])
        population.sort(key=lambda c: c.fitness(), reverse=True)
        best_brain = population[0]

        fitness[gen, :] = [c.fitness() for c in population]

        scaled_fitnesses = np.array(
            [c.fitness() - population[-1].fitness() for c in population]
        )
        scaled_fitnesses /= sum(scaled_fitnesses)

        next_gen = []
        for _ in range(round(len(population) / 4)):
            p1, p2 = rd.choices(population, weights=scaled_fitnesses, k=2)
            c1, c2 = p1.crossover(p2)
            c1.mutate()
            c2.mutate()
            next_gen.append(c1)
            next_gen.append(c2)

        next_gen.extend([c.reset() for c in population[: len(population) // 2]])
        population = next_gen
    return fitness, best_brain, genotypes

def genotypes_to_phenotypes(genotypes: list[list[float]], num_modules: int):
    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    p_matrices = nde.forward(genotypes)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    return hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

CROSSOVER_THRESHOLD = 0.5
MUTATION_THRESHOLD = 0.05

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rd.random() < CROSSOVER_THRESHOLD:
        return parent1.copy(), parent2.copy()
    d = 0.25
    beta = rd.uniform(-d, 1 + d)
    child_1 = beta * parent1 + (1 - beta) * parent2
    child_2 = beta * parent2 + (1 - beta) * parent1
    return child_1, child_2

def mutate(genes: np.ndarray) -> np.ndarray:
    for i in range(len(genes)):
        if rd.random() < MUTATION_THRESHOLD:
            genes[i] += rd.gauss(0, 0.1)
    return genes

def main2() -> None:
    genotype_size = 64
    population_size = 64
    generations = 100
    type_p_genes = RNG.random((population_size, genotype_size)).astype(np.float32)
    conn_p_genes = RNG.random((population_size, genotype_size)).astype(np.float32)
    rot_p_genes = RNG.random((population_size, genotype_size)).astype(np.float32)

    best_brain = []
    best_robot = []
    fitness = []
    for _ in range(generations):
        with Pool(processes=8) as pool:
            genotypes = [[
                    type_p_genes[i],
                    conn_p_genes[i],
                    rot_p_genes[i],
                ] for i in range(population_size)]
            results = pool.map(learn_brain, genotypes)
            sorted_results = sorted(results, key=lambda x: x[1].fitness(), reverse=True)
            best_brain = sorted_results[0][1]
            best_robot = sorted_results[0][2]
            fitness.append([x[1].fitness() for x in results])
            scaled_fitness = np.array(
                [c[1].fitness() - sorted_results[-1][1].fitness() for c in sorted_results]
            )
            scaled_fitness /= sum(scaled_fitness)

            type_next_gen = []
            conn_next_gen = []
            rot_next_gen = []
            for _ in range(round(len(genotypes) / 4)):
                p1, p2 = rd.choices(genotypes, weights=scaled_fitness, k=2)
                child1, child2 = crossover(np.array(p1), np.array(p2))
                child1 = mutate(child1)
                child2 = mutate(child2)
                type_next_gen.append(child1[0])
                type_next_gen.append(child2[0])
                conn_next_gen.append(child1[1])
                conn_next_gen.append(child2[1])
                rot_next_gen.append(child1[2])
                rot_next_gen.append(child2[2])
            type_next_gen.extend([c[0] for c in genotypes[: len(genotypes) // 2]])
            conn_next_gen.extend([c[1] for c in genotypes[: len(genotypes) // 2]])
            rot_next_gen.extend([c[2] for c in genotypes[: len(genotypes) // 2]])

            type_p_genes = np.array(type_next_gen)
            conn_p_genes = np.array(conn_next_gen)
            rot_p_genes = np.array(rot_next_gen)
    
    show_xpos_history(best_brain.history)

def main() -> None:
    """Entry point."""
    # ? ------------------------------------------------------------------ #
    # System parameters
    num_modules = 20

    # ? ------------------------------------------------------------------ #
    genotype_size = 64
    type_p_genes = RNG.random(genotype_size).astype(np.float32)
    conn_p_genes = RNG.random(genotype_size).astype(np.float32)
    rot_p_genes = RNG.random(genotype_size).astype(np.float32)

    genotype = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
    ]

    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    p_matrices = nde.forward(genotype)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    # ? ------------------------------------------------------------------ #
    # Save the graph to a file
    save_graph_as_json(
        robot_graph,
        DATA / "robot_graph.json",
    )

    # ? ------------------------------------------------------------------ #
    # Print all nodes
    core = construct_mjspec_from_graph(robot_graph)

    # ? ------------------------------------------------------------------ #
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    # ? ------------------------------------------------------------------ #
    # Simulate the robot
    ctrl = Controller(
        controller_callback_function=nn_controller,
        # controller_callback_function=random_move,
        tracker=tracker,
    )

    experiment(robot=core, controller=ctrl, mode="launcher")

    # show_xpos_history(tracker.history["xpos"][0])


if __name__ == "__main__":
    main2()

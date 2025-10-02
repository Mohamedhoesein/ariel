import numpy as np
from typing import Any, Self

CROSSOVER_THRESHOLD = 0.5
MUTATION_THRESHOLD = 0.05

rng = np.random.default_rng()


class Layer:
    def __init__(self, input_size: int, output_size: int, function) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros(shape=(input_size, output_size))
        self.function = function

    def random(self) -> Self:
        self.weights = rng.standard_normal((self.input_size, self.output_size))
        return self

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.function(np.dot(inputs, self.weights))


class Brain:
    def __init__(self, layers: list[Layer], mutation_rate: float) -> None:
        self.layers = layers
        self.history: list[list[float]] = []
        self._fitness: None | float = None
        self.mutation_rate = mutation_rate

    def set_history(self, history: list[list[float]]) -> None:
        self.history = history

    def random(self) -> Self:
        self.layers = [layer.random() for layer in self.layers]
        return self

    def control(self, model: Any, data: Any):
        # Get outputs, in this case the positions of the actuator motore (hinges)
        inputs = data.qpos

        # Run the inputs through the lays of the network
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)

        # Scale the outputs to [-pi/2, pi/2]
        #data.ctrl = np.clip(outputs, -np.pi / 2, np.pi / 2)

        # Save movement to history
        #self.history.append(data.xpos.copy())
        return outputs * np.pi

    def crossover(self, other: "Brain") -> list["Brain"]:
        """
        Create two children using crossover with `other`. Uses uniform crossover
        with a probability of `CROSSOVER_THRESHOLD`.

        :param self: Description
        :param other: The other parent
        :type other: "Brain"
        :return: A list containing the two children.
        :rtype: list[Brain]
        """
        left = self.copy()
        right = self.copy()

        P = CROSSOVER_THRESHOLD

        selection = []
        for l in self.layers:
            selection.append(rng.random(size=l.weights.shape))
        for i in range(len(self.layers)):
            left.layers[i].weights[selection[i] > P] = self.layers[i].weights[
                selection[i] > P
            ]
            left.layers[i].weights[selection[i] < P] = other.layers[i].weights[
                selection[i] < P
            ]
            right.layers[i].weights[selection[i] < P] = self.layers[i].weights[
                selection[i] < P
            ]
            right.layers[i].weights[selection[i] > P] = other.layers[i].weights[
                selection[i] > P
            ]

        return [left, right]

    def fitness(self) -> float:
        if self._fitness:
            return self._fitness
        self._fitness = self.history[0][1] - self.history[-1][1]
        return self._fitness


    def mutate(self):
        n = sum(layer.weights.size for layer in self.layers)

        tau = 1 / np.sqrt(n)
        self.mutation_rate *= np.exp(rng.normal(scale=tau))

        P = MUTATION_THRESHOLD

        for layer in self.layers:
            mutation_mask = rng.random(size=layer.weights.shape) < P
            layer.weights[mutation_mask] += rng.normal(
                scale=self.mutation_rate, size=layer.weights.shape
            )[mutation_mask]

    def copy(self) -> "Brain":
        return Brain(
            [Layer(l.input_size, l.output_size, l.function) for l in self.layers],
            self.mutation_rate,
        )

    def reset(self) -> "Brain":
        return Brain(self.layers, self.mutation_rate)

    def export(self) -> dict:
        return {
            "name": type(self).__name__,
            "mutation_rate": self.mutation_rate,
            "layers": [
                {
                    "input_size": layer.input_size,
                    "output_size": layer.output_size,
                    "weights": layer.weights.tolist(),
                    "function": layer.function.__name__,
                }
                for layer in self.layers
            ],
        }

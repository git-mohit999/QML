"""
prepare /00?>, XX and YY on the two qubits, encode dataset as basis states of X and Y axes. X for petals, Y for sepals, entangle em using idk smth, use H(-RY) to get back to computational states, train loop n_shots, collapse the state to get y_preds.
? : to add depth and increase expressivity of the circuit, is entangled with first and 2nd qubit befor encoding then reverse it ig.

ok so sepal on 0, width on RX, length on RY same for petal 1, input normalised length and width to angles, this is data mapped to states thru RX,RY. so the data is contained in the plane perpendicular to the z axis of the bloch sphere, initialise the cost_func, train through the split data, where the length and width are now angles.

	phi : [[sl,sw][pl,pw]] ----> |RX(sl)RY(sw)> {*} |RY(-pw)RX(-pl)> :: so for a flower with a set of such params, will give a unique point on the bloch sphere, a region will correspond to class0, and another as class1. define axis through this region with an error of idk smth acceptale, create observable corresponding to this axis. Measure wrt to this, one of the eigenvalues will correspond to class 0 and other to class1.

So essentially record a decision plane, take axis perpendicular to it thats the observale, say O, thus, <psi|O|psi> whichll give two eigenvalues will correspond to the binary classes, determine the [eigenvalue ~ binary_class] while encoding data into the bloch sphere.
"""

import pennylane as qml
from pennylane import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --------------------
# 1. Dataset prep
# --------------------
iris = datasets.load_iris()
X = iris.data[:100, :2]  # sepal length, sepal width
y = iris.target[:100]    # binary classes: 0 = setosa, 1 = versicolor

# Normalize features to [0, pi] for RX/RY angles
scaler = MinMaxScaler(feature_range=(0, np.pi))
X = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=67)

# --------------------
# 2. Device setup
# --------------------
n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)

# --------------------
# 3. Feature encoding + circuit definition
# --------------------
def encode_features(x):
    # x = [sepal_length, sepal_width]
    qml.RX(x[0], wires=0)
    qml.RY(x[1], wires=0)

    # petal encoded as mirrored (to simulate second subspace)
    qml.RX(-x[1], wires=1)
    qml.RY(-x[0], wires=1)

@qml.qnode(dev)
def circuit(params, x):
    encode_features(x)

    # entanglement between data qubits
    qml.CNOT(wires=[0,1])

    # entangle ancilla
    qml.CNOT(wires=[0,2])
    qml.CNOT(wires=[1,2])

    # apply trainable layer
    qml.RY(params[0], wires=2)

    # measurement axis defines the "decision plane"
    return qml.expval(qml.PauliZ(2))

# --------------------
# 4. Loss and training
# --------------------
def cost(params, X, y):
    loss = 0
    for xi, yi in zip(X, y):
        pred = circuit(params, xi)
        #converting classical y_i to "eigenvalues" kinda +/-1
        loss += (pred - (1 - 2*yi)) ** 2
    return loss / len(X)

opt = qml.GradientDescentOptimizer(stepsize=0.2)
params = np.array([0.1], requires_grad=True)

for step in range(50):
    params = opt.step(lambda v: cost(v, X_train, y_train), params)
    if step % 10 == 0:
        print(f"Step {step}: loss = {cost(params, X_train, y_train):.4f}")

# --------------------
# 5. Test accuracy
# --------------------
preds = [np.sign(circuit(params, xi)) for xi in X_test]
preds = [(1 if p < 0 else 0) for p in preds]
acc = np.mean(preds == y_test)
print(f"Test accuracy: {acc*100:.2f}%")

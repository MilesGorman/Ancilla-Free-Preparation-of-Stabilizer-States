import numpy as np
import random as rand
import qiskit as qi
import matplotlib.pyplot as plt

class Code:
    def __init__(self, stabilizers: np.array):
        """
        Initializes a stabilizer code object with given stabilizer generators.

        :param stabilizers: The code's stabilizer generators represented as a NumPy array where each row defines a
        generator, the first column represents the phase of the stabilizers, 0 for +1 and 2 for -1, while the remaining
        columns define the supported operators: 0, 1, 2, and 3 for an identity, Pauli-X, -Y, and -Z respectively.
        """
        self.stabilizers = stabilizers.copy()
        self.initial_stabilizers = stabilizers.copy()
        self.encoding_circuit = None

    def read(self, now_or_initial: str = "now") -> None:
        """
        Prints the stabilizer generators of the code in a more human-readable format: products of single-qubit operators
        (I, X, Y, and Z) with a phase (+1 or -1).

        :param now_or_initial: If 'initial', prints the original stabilziers inputted. If 'now', prints the current
                               stabilziers, perhaps after operations have been applied.
        """

        def reader(operators: np.array) -> str:
            """
            Converts a NumPy array of operators into a readable string of phases (+1 or -1) and products of operators
            (I, X, Y, and Z).

            :param operators: A NumPy array in the format described in the Code object initialization.
            :return: A formatted string listing each stabilizer in a more human-readable form.
            """

            readable = ''

            # Iterates through each operator.
            for row in operators:

                phase_col = True

                readable_row = ''

                for element in row:

                    # Displays the phase.
                    if phase_col == True:
                        if element == 0:
                            readable_row += '+1 '

                        elif element == 1:
                            readable_row += '+i '

                        elif element == 2:
                            readable_row += '-1 '

                        elif element == 3:
                            readable_row += '-i '

                        else:
                            raise Exception("Invalid Phase (Reader)")

                        phase_col = False

                    # Displays the single-qubit operators.
                    else:
                        if element == 0:
                            readable_row += 'I '

                        elif element == 1:
                            readable_row += 'X '

                        elif element == 2:
                            readable_row += 'Y '

                        elif element == 3:
                            readable_row += 'Z '

                        else:
                            raise Exception("Invalid Operator (Reader)")

                readable += readable_row + '\n'

            return readable

        if now_or_initial == 'now':
            stabilziers = self.stabilizers.copy()
        elif now_or_initial == 'initial':
            stabilziers = self.initial_stabilizers.copy()
        else:
            raise Exception("Invalid Operator: Please Choose 'now' Or 'initial' (reader).")

        print("Stabilziers: \n" + reader(stabilziers))

        return None

    def CNOT(self, control: int, target: int) -> None:
        """
        Applies a desired CNOT to the stabilizer state.

        :param control: Control qubit index for the CNOT, starting from 1.
        :param target: Target qubit index for the CNOT, starting from 1.
        """

        def CNOT_gate(row: np.array, control: int, target: int):
            """
            Implements how the CNOT gate propogates single-qubit Paulis in a given multi-qubit Pauli operator.

            :param row: A single-row NumPy array in the format described in the Code object initialization.
            :param control: Control qubit index for the CNOT, starting from 1.
            :param target: Target qubit index for the CNOT, starting from 1.
            """

            if int(row[control]) == 0: # Control is I.

                if int(row[target]) == 0: # Target is I.
                    # Nothing changes.
                    pass

                elif int(row[target]) == 1: # Target is X.
                    # Nothing changes.
                    pass

                elif int(row[target]) == 2: # Target is Y.
                    # Control becomes Z.
                    row[control] = 3

                elif int(row[target]) == 3: # Target is Z.
                    # Control becomes Z.
                    row[control] = 3

                else:
                    raise Exception("Invalid Target (CNOT: Control I).")


            elif int(row[control]) == 1: # Control is X.

                if int(row[target]) == 0: # Target is I.
                    # Target becomes X.
                    row[target] = 1

                elif int(row[target]) == 1: # Target is X.
                    # Target becomes I.
                    row[target] = 0

                elif int(row[target]) == 2:  # Target is Y.
                    # Control becomes Y
                    row[control] = 2
                    # Target becomes Z.
                    row[target] = 3

                elif int(row[target]) == 3: # Target is Z.
                    # Control becomes Y.
                    row[control] = 2
                    # Target becomes Y.
                    row[target] = 2
                    # Picks up a -1 phase.
                    row[0] = (row[0] + 2) % 4


                else:
                    raise Exception("Invalid Target (CNOT: Control X).")


            elif int(row[control]) == 2:  # Control is Y.

                if int(row[target]) == 0:  # Target is I.
                    # Target becomes X.
                    row[target] = 1

                elif int(row[target]) == 1:  # Target is X.
                    # Target becomes I.
                    row[target] = 0

                elif int(row[target]) == 2:  # Target is Y.
                    # Control becomes X.
                    row[control] = 1
                    # Target becomes Z.
                    row[target] = 3
                    # Picks up a -1 phase.
                    row[0] = (row[0] + 2) % 4

                elif int(row[target]) == 3:  # Target is Z.
                    # Control becomes X.
                    row[control] = 1
                    # Target becomes Y.
                    row[target] = 2

                else:
                    raise Exception("Invalid Target (CNOT: Control Y).")


            elif int(row[control]) == 3: # Control is Z.

                if int(row[target]) == 0: # Target is I.
                    # Nothing changes.
                    pass

                elif int(row[target]) == 1: # Target is X.
                    # Nothing changes.
                    pass

                elif int(row[target]) == 2:  # Target is Y.
                    # Control becomes I.
                    row[control] = 0

                elif int(row[target]) == 3:  # Target is Z.
                    # Control becomes I.
                    row[control] = 0

                else:
                    raise Exception("Invalid Target (CNOT: Control Z).")


            else:
                raise Exception("Invalid Control (CNOT).")

        for stabilizer in self.stabilizers:
            CNOT_gate(stabilizer, control, target)

    def H(self, target: int) -> None:
        """
        Applies a desired Hadamard to the stabilizer state.

        :param target: Index of the qubit that the Hadamard acts on, starting from 1.
        """

        def H_gate(row: np.array, target: int):
            """
            Implements how the Hadamard gate propogates single-qubit Paulis in a given multi-qubit Pauli operator.

            :param row: A single-row NumPy array in the format described in the Code object initialization.
            :param target: Index of the qubit that the Hadamard acts on, starting from 1.
            """

            if int(row[target]) == 0: # Target is I.
                # Target stays I.
                pass

            elif int(row[target]) == 1: # Target is X.
                # Target becomes Z.
                row[target] = 3

            elif int(row[target]) == 2: # Target is Y.
                # Target becomes -Y.
                row[0] = (row[0] + 2) % 4

            elif int(row[target]) == 3: # Target is Z.
                # Target becomes X.
                row[target] = 1

            else:
                raise Exception("Invalid Target (H).")

        for stabilizer in self.stabilizers:
            H_gate(stabilizer, target)

    def S(self, target: int) -> None:
        """
        Applies a desired Phase gate to the stabilizer state.

        :param target: Index of the qubit that the phase gate acts on, starting from 1.
        """

        def S_gate(row: np.array, target: int):
            """
            Implements how the phase gate propogates single-qubit Paulis in a given multi-qubit Pauli operator.

            :param row: A single-row NumPy array in the format described in the Code object initialization.
            :param target: Index of the qubit that the Hadamard acts on, starting from 1.
            """

            if int(row[target]) == 0: # Target is I.
                # Target stays I.
                pass

            elif int(row[target]) == 1: # Target is X.
                # Target becomes Y.
                row[target] = 2

            elif int(row[target]) == 2: # Target is Y.
                # Target becomes -X.
                row[target] = 1
                row[0] = (row[0] + 2) % 4

            elif int(row[target]) == 3: # Target is Z.
                # Target stays Z.
                pass

            else:
                raise Exception("Invalid Target (S).")

        for stabilizer in self.stabilizers:
            S_gate(stabilizer, target)

    def info(self) -> list[tuple[int, list, list, list, list]]:
        """
        Summarises the position of the single-qubit Pauli operators in each stabilizer generator, by type.

        :return: A list of tuples, one for each stabilizer. Each tuple contains
                    - the index of the stabilizer i.e. the first is 0, the second is 1, and so on,
                    - a list of qubit indices Pauli-Xs act on,
                    - a list of qubit indices Pauli-Ys act on,
                    - a list of qubit indices Pauli-Zs act on,
                    - the concatenation of the above lists - the locations of all the non-trivial operations in each
                      stabilizer.
                 The qubit indices start from 1.
        """

        row_info = []  # The list we return at the end summarizing everything.
        row_num = 0  # Counts the row in the matrix we are considering starting from 0.

        operators = self.stabilizers

        for row in operators:

            X_locations = []  # Locations of X operators in a stabilizer.
            Y_locations = []  # Locations of Y operators in a stabilizer.
            Z_locations = []  # Locations of Z operators in a stabilizer.
            column_num = 1  # Counts the column in the matrix / qubit we are considering.

            for element in row[1:]:

                if element == 0:  # Qubit has an I in this stabilizer.
                    pass

                elif element == 1:  # Qubit has an X in this stabilizer.
                    X_locations += [column_num]

                elif element == 2:  # Qubit has an Y in this stabilizer.
                    Y_locations += [column_num]

                elif element == 3:  # Qubit has an Z in this stabilizer.
                    Z_locations += [column_num]

                else:
                    raise Exception("Invalid Element (Cost)")

                column_num += 1

            row_info += [(row_num, X_locations, Y_locations, Z_locations, X_locations + Y_locations + Z_locations)]

            row_num += 1

        return row_info

    def summary(self) -> list[tuple[int, int, int, int, int]]:
        """
        Summarises the number of the single-qubit Pauli operators in each stabilizer generator, by type.

        :return: A list of tuples, one for each stabilizer. Each tuple contains
                    - the index of the stabilizer i.e. the first is 0, the second is 1, and so on,
                    - the number of Pauli-Xs the stabilizer contains,
                    - the number of Pauli-Ys the stabilizer contains,
                    - the number of Pauli-Zs the stabilizer contains.
        """

        row_info = self.info()
        row_summary = []

        for row in row_info:

            row_num = row[0]
            X_count = len(row[1])
            Y_count = len(row[2])
            Z_count = len(row[3])

            row_summary += [(row_num,X_count,Y_count,Z_count)]

        return row_summary

    def find_cheapest(self, H_cost: int = 0, S_dagger_cost: int = 0, CNOT_cost: int = 1) -> int:
        """
        Finds the cheapest stabilizer to encode of the stabilizers not already encoded: the cheapest stabilizer to
        transform into a product of Pauli-Zs and identities, and then reduce to a weight-1 operator using CNOTs. Ties
        between equal-cost stabilizers are broken randomly.

        :param H_cost: Cost assigned to each Hadamard gate.
        :param S_dagger_cost: Cost assigned to each phase gate. Note: the conjugate phase gate - and not the phase gate
                              - is used here because the encoding circuit is computed as the Hermitian conjugate of the
                              de-encoding circuit.
        :param CNOT_cost: Cost assigned to each CNOT gate..
        :return: The index of the stabilizer (among unencoded ones) with the lowest encoding cost.
        """

        row_summary = self.summary()
        row_numbers = [] # Holds lowest cost row (or rows when there is degeneracy in the local minimum cost).
        min_cost = float('inf') # Upper bound on the global minimum cost.

        for row in row_summary:

            num_H = row[1] + row[2] # Each X and Y needs a H gate to become a Z.
            num_S = row[2] # Each Y needs an S gate to become an X (and then later a Z).
            num_CNOT = row[1]+row[2]+row[3]-1 # Each CNOT reduces the stabilzier's weight by 1.

            cost = num_H * H_cost + num_S * S_dagger_cost + num_CNOT * CNOT_cost

            # Records stabilizers not already encoded if they set a new minimum cost.
            if ( cost < min_cost and not ( row[1] == 0 and row[2] == 0 and row[3] == 1 ) ):
                min_cost = cost
                row_numbers = [row[0]]

            # Records stabilizers if they match the currently known minimum cost.
            elif cost == min_cost:
                row_numbers += [row[0]]

        # Random selection among minimum cost stabilziers to break ties.
        optimal_row_num = rand.randint(0, len(row_numbers) - 1)

        return row_numbers[optimal_row_num]

    def find_complete(self) -> list[tuple[int, int]]:
        """
        Identifies all stabilizers that have already been fully encoded. A stabilizer is considered encoded if it has
        weight 1 and consists of a single Pauli-Z operator: it is consistent with a zero state on some qubit.

        :return: A list of tuples, one for each fully encoded stabilizer. Each tuple contains
                    - the index of the stabilizer,
                    - the index of the qubit on which the lone Pauli-Z acts.
                 Both indices start from 0.
        """

        row_info = self.info()
        complete_rows = []

        for row in row_info:

            row_num, X_locations, Y_locations, Z_locations, all_locations = row

            if len(Z_locations) == 1 and len(all_locations) == 1:
                complete_rows += [(row_num, Z_locations[0])]

        return complete_rows

    def rearange(self) -> None:
        """
        Updates the stabilizer generating set by taking products of existing generators to simplify future encoding
        steps. In particular, for every weight-1 stabilizer with a Pauli-Z on qubit *i*, we effectively erase any
        other Pauli-Zs on qubit *i* in the other stabilziers. This ensures that all these other stabilziers act
        trivially on qubit *i* and so qubit *i* never needs to be acted on again in the encoding circuit, simplifying
        the encoding. This transformation is always possible due to the mutual commutativity of stabilizer generators.
        """

        operators = self.stabilizers

        complete_stabilziers = self.find_complete()

        for complete_stabilzier in complete_stabilziers:

            row_num, Z_location = complete_stabilzier
            stabilizer = self.stabilizers[row_num]
            stabilizer_phase = stabilizer[0]
            operator_num = 0

            for operator in operators:

                # If the single-qubit operator is an identity.
                if operator[Z_location] == 0:
                    pass

                # If the single-qubit operator is Z and the stabilizer is not the weight-1 stabilzier being considered.
                elif operator[Z_location] == 3 and operator_num != row_num:
                    operator[Z_location] = 0
                    operator[0] = ( operator[0] + stabilizer_phase ) % 4

                elif operator[Z_location] == 3:
                    pass

                else:
                    raise Exception('Operator Does Not Commute (rearange)')

                operator_num += 1

    def done_check(self) -> bool:
        """
        Checks whether all stabilizers have been successfully encoded. This is true only when each stabilizer has been
        reduced to a weight-1 operator containing a Pauli-Z.

        :return: True if all stabilizers are encoded; False otherwise.
        """

        complete_rows = self.find_complete()

        operators = self.stabilizers

        num_operators, num_qubits = np.shape(operators)

        if len(complete_rows) == num_operators:
            return True
        else:
            return False

    def reduce_once(self, H_cost: int = 0, S_dagger_cost: int = 0, CNOT_cost: int = 1) -> tuple[list, list, list[tuple]]:
        """
        Finds the gates needed to encode the locally optimal stabilizer: the cheapest stabilizer according to the
        `find_cheapest` method. At the same time as the gates are found, they are also applied to all the current
        stabilizers.

        :param H_cost: Cost assigned to each Hadamard gate.
        :param S_dagger_cost: Cost assigned to each conjugate phase gate.
        :param CNOT_cost: Cost assigned to each CNOT gate.
        :return: A tuple of lists describing the gates used in encoding the selected stabilizer. The tuple has the form
                    - a list of qubit indices where Hadamards are applied,
                    - a list of qubit indices where S† gates are applied,
                    - a list of (control, target) tuples for each CNOT gate applied.
        """

        optimal_row_num = self.find_cheapest(H_cost, S_dagger_cost, CNOT_cost)

        row_num, X_locations, Y_locations, Z_locations, non_trivial_locations = self.info()[optimal_row_num]

        Ss = [] # The phase gates to apply.
        for location in Y_locations:
            Ss += [location]
            self.S(location)

        Hs = [] # The Hadamard gates to apply.
        for location in X_locations + Y_locations:
            Hs += [location]
            self.H(location)

        CNOTs = [] # The CNOTs to apply, done in a way to reduce depth: each time step uses the maximum number of CNOTs.
        while len(non_trivial_locations) > 1:

            random_pairings = np.random.permutation(len(non_trivial_locations))
            left_over = None

            if len(random_pairings) % 2 == 1:
                left_over = non_trivial_locations[random_pairings[-1]]
                random_pairings = random_pairings[:-1]

            pairs = []
            for i in [2 * j for j in range(0, int(len(random_pairings) / 2))]:
                pairs += [(non_trivial_locations[random_pairings[i]], non_trivial_locations[random_pairings[i + 1]])]

            non_trivial_locations = []
            for pair in pairs:
                CNOTs += [pair]
                self.CNOT(pair[0], pair[1])
                non_trivial_locations += [pair[1]]
            if left_over != None:
                non_trivial_locations += [left_over]


        return (Hs, Ss, CNOTs)

    def reduce(self, H_cost: int = 0, S_dagger_cost: int = 0, CNOT_cost: int = 1) -> tuple[list[tuple[list, list, list]], tuple[list, list, list[tuple]]]:
        """
        Greedy algorithm for finding efficient encoding circuit of a code: at each step, the algorithm selects and
        encodes the cheapest stabilizer using the specified gate costs, updating the stabilizer set as it proceeds.
        The encoding circuit found is stored in `self.encoding_circuit`.

        :param H_cost: Cost assigned to each Hadamard gate.
        :param S_dagger_cost: Cost assigned to each conjugate phase gate.
        :param CNOT_cost: Cost assigned to each CNOT gate.
        :return: A tuple of two elements:
                    1. a list of tuples that specify the de-encoding circuit's gates, where each tuple has
                        - a list of Hadamard gate locations,
                        - a list of phase gate locations,
                        - a list of CNOTs as (control, target) pairs,
                       where the order of the tuples specifies the order of the gates;
                    2. a list of tuples that specify the encoding circuit's gates: the tuples of (1) in reverse where
                In each tuple, for un-encoding, the single-qubit operations in that step are applied before the CNOTs.
                Conversly, for encoding, CNOTs are applied before the single qubit operations in each tuple.
        """

        de_encoding_gates = []

        while not self.done_check():
            de_encoding_gates += [self.reduce_once(H_cost, S_dagger_cost, CNOT_cost)]
            self.rearange()

        encoding_gates = []
        de_encoding_gates.reverse()
        for step in de_encoding_gates:
            Hs, Ss, CNOTs = step
            CNOTs.reverse()
            encoding_gates += [(Hs, Ss, CNOTs)]

        self.encoding_circuit = encoding_gates

        return encoding_gates

    def encoding_cost(self, H_cost: int = 0, S_dagger_cost: int = 0, CNOT_cost: int = 1) -> int:
        """
        computes the total cost of the encoding circuit given user-defined gate costs.

        :param H_cost: Cost assigned to each Hadamard gate.
        :param S_dagger_cost: Cost assigned to each conjugate phase gate.
        :param CNOT_cost: Cost assigned to each CNOT gate.
        :return: The total cost of the encoding circuit using the provided gate cost weights.
        """

        circuit = self.encoding_circuit

        if circuit != None:

            num_Hs = 0
            num_Ss = 0
            num_CNOTs = 0

            for row in circuit:

                Hs, Ss, CNOTs = row

                num_Hs += len(Hs)
                num_Ss += len(Ss)
                num_CNOTs += len(CNOTs)

            return num_Hs * H_cost + num_Ss * S_dagger_cost + num_CNOTs * CNOT_cost

        else:
            return None

    def circ_and_QASM(self, QASM: bool = True, print_circ_png: bool = True, print_circ_pdf: bool = True,  SAVE_PATH: str = "") -> qi.QuantumCircuit:
        """"
        Converts the encoding circuit found into a Qiskit `QuantumCircuit` object and optionally exports the
        corresponding QASM file and/or circuit diagram. It also, crucially, includes Pauli-X gates so that the starting
        state is all zeros - to account for -1 phases in some stabilizers.

        :param QASM: If True, exports the circuit as a QASM text file.
        :param print_circ_png: If True, saves the circuit diagram as a PNG image.
        :param print_circ_pdf: If True, saves the circuit diagram as a PDF image.
        :param SAVE_PATH: Path used for saving the PNG/PDF/QASM outputs, formatted as `/Users/[Username]/[Folder Name]`.
        :return: A Qiskit `QuantumCircuit` object representing the encoding circuit.
        """

        if self.encoding_circuit == None:
            return None

        num_stabilizers, num_qubits = np.shape(self.stabilizers)
        circ = qi.QuantumCircuit(num_qubits - 1)

        negative_phases = []
        for stabilizer in self.stabilizers:
            if stabilizer[0] == 2: # Phase is -1.
                for qubit in range(1, num_qubits):
                    if stabilizer[qubit] == 3:  # The element is a Z operator.
                        negative_phases += [qubit]

        for qubit in negative_phases:
            circ.x(qubit - 1)

        for step in self.encoding_circuit:

            Hs, Ss, CNOTs = step

            for qubits in CNOTs:
                control, target = qubits
                circ.cx(control - 1, target - 1)

            for qubit in Hs:
                circ.h(qubit - 1)

            for qubit in Ss:
                circ.sdg(qubit - 1)

        # Gives the files a unique, identifiable name.
        unique_ID = rand.randint(0, 1048576)
        print(f"Your file's unique identifier is: {unique_ID}.")
        if QASM == True:
            circ.qasm(formatted=False, filename=( SAVE_PATH + f"QASM_{unique_ID}.txt") )
        if print_circ_png == True:
            circ.draw(output='mpl', fold=-1, filename=( SAVE_PATH + f"PNG_{unique_ID}.png"), initial_state=True)
        if print_circ_pdf == True:
            circ.draw(output='latex', filename= ( SAVE_PATH + f"PDF_{unique_ID}.pdf"), initial_state=True)
            # Compatible with Inkscape using poppler import option.

        return circ

    def verify(self) -> None:
        """
        Applies the encoding circuit computed to an appropriate computational basis state and prints the stabilizers
        the encoding circuits produces and the initial, user-inputted stabilizers. Note: the two sets of stabilizers may
        differ by trivial products, i.e., they may generate the same stabilizer group.
        """

        stabilizers = self.stabilizers.copy()

        for step in self.encoding_circuit:

            Hs, Ss, CNOTs = step

            for qubits in CNOTs:
                control, target = qubits
                self.CNOT(control, target)

            for qubit in Hs:
                self.H(qubit)

            for qubit in Ss: # S^\dagger = S^3.
                self.S(qubit)
                self.S(qubit)
                self.S(qubit)

        print("After encoding \n_____________________________")
        self.read('now')
        print("Initial stabilizers \n_____________________________")
        self.read('initial')

        self.stabilizers = stabilizers

        return None

    def reduce_iterations(self, SAVE_PATH: str = "", number_of_iterations: int = 100, H_cost: int = 0, S_dagger_cost: int = 0, CNOT_cost: int = 1) -> int:
        """
        Greedy algorithm for finding efficient encoding circuit of a code: at each step, the algorithm selects and
        encodes the cheapest stabilizer using the specified gate costs, updating the stabilizer set as it proceeds.
        Because the algorithm is stochastic (e.g., uses random tie-breaking), it is repeated for a specified number of
        iterations. The lowest-cost encoding circuit found across all iterations is stored in `self.encoding_circuit`.

        :param SAVE_PATH: Path used for saving a graph of the distribution of costs of encoding circuits found,
                          formatted as `/Users/[Username]/[Folder Name]/plot.png`.
        :param number_of_iterations: Number of times to run, i.e., number of candidate circuits to generate.
        :param H_cost: Cost assigned to each Hadamard gate.
        :param S_dagger_cost: Cost assigned to each conjugate phase gate.
        :param CNOT_cost: Cost assigned to each CNOT gate.
        :return: The lowest cost across all the encoding circuits found after all the iterations.
        """

        encoding_costs = []
        iteration_num = [iteration for iteration in range(1, number_of_iterations + 1)]
        min_cost = float('inf')

        print("Progress: ")
        for iteration in iteration_num:

            if iteration % 10 == 0:
                print(f"{iteration}/{number_of_iterations}")

            self.stabilizers = self.initial_stabilizers.copy()

            encoding_gates = self.reduce(H_cost, S_dagger_cost, CNOT_cost)
            cost_of_encoding = self.encoding_cost(H_cost, S_dagger_cost, CNOT_cost)
            encoding_costs += [cost_of_encoding]

            if cost_of_encoding < min_cost:
                min_cost = cost_of_encoding
                min_stabilizers = self.stabilizers.copy()
                min_encoding = encoding_gates

        self.stabilizers = min_stabilizers
        self.encoding_circuit = min_encoding

        if SAVE_PATH != "":

            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = 17

            x = iteration_num
            y = encoding_costs

            x_save, y_save = x.copy(), y.copy()

            fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,
                                           gridspec_kw={"width_ratios": [3, 1], "wspace": 0})

            ax1.plot(x_save, y_save, ".", color='tab:blue')
            ax1.set_xlabel('Iteration Number')
            ax1.set_ylabel('Cost')

            y_lims = ax1.get_ylim()
            x_lims = ax1.get_xlim()

            ax1.hlines(y=max(y_save), xmin=x_lims[0], xmax=x_lims[1], linewidth=2, color='tab:blue', alpha=0.5)
            ax1.hlines(y=min(y_save), xmin=x_lims[0], xmax=x_lims[1], linewidth=2, color='tab:blue', alpha=0.5)

            ax1.set_ylim(y_lims)
            ax1.set_xlim(x_lims)

            ax2.hist(y_save, bins=20, ec="tab:blue", fc='tab:blue', density=True, orientation="horizontal", alpha=0.5)

            ax2.tick_params(axis="y", left=False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.set_xticks([])

            ax2.text(0, max(y_save), f'  Max {max(y_save)}')
            ax2.text(0, min(y_save), f'  Min {min(y_save)}')

            fig.set_size_inches(15, 7)

            plt.savefig(SAVE_PATH, dpi=300)

            plt.show()

        return encoding_costs

def import_code(file_name: str) -> Code:
        """
        Imports a stabilizer code from a text file (e.g. copied from codetables.de) into a `Code` object. This can be
        used to import other parity check matrices of the same form as those from codetables.de.

        :param file_name: Address of the text file containing the code's parity-check matrix. Example format:
                          '/Users/[Folder Name]/[File Name].txt'.
        :return: A `Code` object initialized with the stabilizers specified in the file.
        """

        file = open(file_name, 'r')
        words = file.read().splitlines()
        file.close()

        X_lines = []
        Z_lines = []
        for line in words:
            X_line, Z_line = line.split('|')

            X_line = X_line.lstrip(' ')
            X_line = X_line.strip('[')
            Z_line = Z_line.strip(']')

            X_line = [int(b) for b in X_line.split(' ')]
            Z_line = [3 * int(b) for b in Z_line.split(' ')]

            X_lines += [X_line]
            Z_lines += [Z_line]

        Hx = np.array([np.array(X_line) for X_line in X_lines])
        Hz = np.array([np.array(Z_line) for Z_line in Z_lines])

        H = Hx + Hz

        row_num = 0
        for row in H:
            col_num = 0
            for element in row:
                if element == 4:
                    H[row_num][col_num] = 2
                col_num += 1
            row_num += 1

        # +1 phases.
        phases = np.zeros((row_num, 1))
        H = np.c_[phases, H]

        code = Code(H, np.array([[]]))

        return code































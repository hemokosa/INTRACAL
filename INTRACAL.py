import numpy as np
import random
import re
import io
from contextlib import redirect_stdout
from quri_parts.circuit import QuantumCircuit
from quri_parts.qulacs.simulator import run_circuit
from quri_parts.circuit.utils.circuit_drawer import draw_circuit

class INTRACAL:
    """
    An interpreter for INTRACAL, an esoteric quantum programming language
    inspired by the infamous INTERCAL. It combines quantum computation with
    arbitrary rules about politeness, despair, and unreliability.
    """
    DEFAULT_SNARK = [
        "That was unnecessarily polite. Please be less courteous.",
        "You must say PLEASE sometimes, but not always.",
        "A little bit of courtesy never killed anyone... or did it?",
        "Hmm, I see you forgot your manners. How rude.",
        "Excessive politeness detected. Don't overdo it.",
        "Politeness is a virtue, but in moderation.",
        "You are treating this computer far too kindly.",
        "This is a quantum computer, not a tea party.",
    ]

    def __init__(self, num_qubits, debug=True, init=None, snark=None):
        self.num_qubits = num_qubits
        self.debug = debug
        
        # --- Session State ---
        self.state = self._init_state(init)
        self.circuit = QuantumCircuit(num_qubits)
        self.circuit_all = QuantumCircuit(num_qubits)
        self.state_history = [self.state.copy()]
        self.command_history = []
        self.log_msgs = []

        # --- INTRACAL Language-Specific State ---
        self.please_count = 0
        self.total_lines = 0
        self.polite_lines = []
        self.rude_lines = []
        self.giveup = False
        self.abstained = set()
        self.remembered = set()
        self.ignored_qubits = set()
        self.unreliable_mode = False
        self.unreliable_prob = 0.15
        
        self.snark = list(self.DEFAULT_SNARK)
        if snark:
            self.snark += list(snark)

        # --- Code parsing cache ---
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.come_from_target_map = {}

    def _init_state(self, init):
        """Initializes the quantum state vector."""
        dim = 2**self.num_qubits
        if init is None:
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0
        elif isinstance(init, str):
            state = np.zeros(dim, dtype=complex)
            state[int(init, 2)] = 1.0
        else:
            state = np.array(init, dtype=complex)
            # Normalize the initial state
            norm = np.linalg.norm(state)
            if norm > 1e-9:
                state /= norm
        return state

    def log(self, message):
        """Logs a message for debugging and user feedback."""
        if self.debug:
            print(message)
        self.log_msgs.append(message)

    def _prepare_code(self, code):
        """Converts code string to a list of lines."""
        if isinstance(code, str):
            return code.strip().splitlines()
        return code

    def _scan_labels_and_comefroms(self, code_lines):
        """
        Scans the code to create maps for labels and COME FROM statements.
        This pre-processing step simplifies the main execution loop.
        """
        # Clear maps before each scan to ensure a clean state for parsing
        self.label_to_idx.clear()
        self.idx_to_label.clear()
        self.come_from_target_map.clear()

        for idx, line in enumerate(code_lines):
            stripped = line.strip()
            
            # Match labels like (LABEL) or LABEL:
            m_label = re.match(r'\((\w+)\)', stripped) or re.match(r'(\w+):', stripped)
            if m_label:
                label = m_label.group(1)
                if label in self.label_to_idx:
                    self.log(f"WARNING: Label '{label}' redefined at line {idx+1}. Using the new definition.")
                self.label_to_idx[label] = idx
                self.idx_to_label[idx] = label

            # Match COME FROM statements
            m_comefrom = re.match(r'COME FROM (\w+)', stripped)
            if m_comefrom:
                target_label = m_comefrom.group(1)
                if target_label in self.come_from_target_map:
                    self.log(f"WARNING: Multiple COME FROMs target label '{target_label}'. Only the one at line {idx+1} will be effective.")
                self.come_from_target_map[target_label] = idx

    def _reset_session_vars(self):
        """Resets variables for a new parsing session."""
        self.please_count = 0
        self.total_lines = 0
        self.polite_lines = []
        self.rude_lines = []
        self.giveup = False
        self.abstained = set()
        self.remembered = set()
        self.ignored_qubits = set()
        self.unreliable_mode = False
        self.circuit = QuantumCircuit(self.num_qubits)
        self.circuit_all = QuantumCircuit(self.num_qubits)
        self.state_history = [self.state.copy()]
        self.command_history = []
        self.log_msgs = []

    def parse(self, code, max_steps=500):
        """
        Parses and executes the INTRACAL code.

        Returns:
            A tuple containing the final state, state history, command history,
            the complete quantum circuit, and log messages.
        """
        code_lines = self._prepare_code(code)
        self._reset_session_vars()
        self._scan_labels_and_comefroms(code_lines)
        
        i = 0
        steps = 0
        while i < len(code_lines) and steps < max_steps:
            # --- COME FROM hijacking ---
            current_label = self.idx_to_label.get(i)
            if current_label and current_label in self.come_from_target_map:
                come_from_line_idx = self.come_from_target_map[current_label]
                self.log(f"[COME FROM] (Line {i+1}) -> Execution hijacked by COME FROM at line {come_from_line_idx+1}.")
                i = come_from_line_idx + 1
                steps += 1
                continue

            line = code_lines[i].strip()

            # --- Strip label definitions ---
            m_label = re.match(r'\((\w+)\)\s*(.*)', line) or re.match(r'(\w+):\s*(.*)', line)
            if m_label:
                line = m_label.group(2).strip()

            # Skip empty or pure COME FROM lines
            if not line or line.startswith("COME FROM"):
                i += 1
                steps += 1
                continue
            
            # --- Politeness check ---
            has_please = False
            if line.startswith("PLEASE"):
                has_please = True
                line = line[6:].strip()
            
            self.total_lines += 1
            if has_please:
                self.please_count += 1
                self.polite_lines.append(i + 1)
            else:
                self.rude_lines.append(i + 1)
            
            # --- Command Execution ---
            jump_target = self._run_command(line, i)
            
            if jump_target is not None:
                i = jump_target
            else:
                i += 1
            steps += 1

        if steps >= max_steps:
            self.log(f"WARNING: Maximum step count ({max_steps}) reached. Possible infinite loop.")

        self._print_politeness_summary()
        self.log("INTRACAL Execution Completed.")

        # Capture circuit diagram as a string and add to logs
        if self.circuit_all.gates:
            with io.StringIO() as buf, redirect_stdout(buf):
                draw_circuit(self.circuit_all)
                circuit_diagram = buf.getvalue()
            self.log("\n--- Quantum Circuit Diagram ---")
            self.log(circuit_diagram)
        else:
            self.log("\n--- Quantum Circuit Diagram ---")
            self.log("No quantum gates were applied in this program.")

        return self.state, self.state_history, self.command_history, self.circuit_all, self.log_msgs

    def _run_command(self, line, idx):
        """
        Parses and executes a single line of INTRACAL code.
        Returns a target line index for GOTO, or None otherwise.
        """
        # --- Handle GIVE UP / RESUME states ---
        if line.startswith("GIVE UP"):
            self.giveup = True
            self.log(f"[GIVE UP] (Line {idx+1}) -- The program has lost all hope.")
            return None
        if line.startswith("RESUME"):
            if self.giveup and random.random() < 0.25:
                self.log(f"[RESUME] (Line {idx+1}) -- The program refuses to resume. Still giving up.")
            else:
                self.log(f"[RESUME] (Line {idx+1}) -- Hope restored. (Barely.)")
                self.giveup = False
            return None

        # If giving up, maybe skip the line
        if self.giveup and random.random() < 0.7:
            self.log(f"[SKIP] (Line {idx+1}) -- GIVE UP: skipping this line in despair.")
            return None

        # --- Handle ABSTAIN / REINSTATE / REMEMBER states ---
        m_abstain = re.match(r"ABSTAIN (\w+)", line)
        if m_abstain:
            thing = m_abstain.group(1).upper()
            self.abstained.add(thing)
            self.log(f"[ABSTAIN] (Line {idx+1}) -- {thing} is now forbidden.")
            if thing == "RELIABILITY": self.unreliable_mode = True
            return None
            
        m_reinstate = re.match(r"REINSTATE (\w+)", line)
        if m_reinstate:
            thing = m_reinstate.group(1).upper()
            if thing in self.abstained: self.abstained.remove(thing)
            self.log(f"[REINSTATE] (Line {idx+1}) -- {thing} is back in business.")
            if thing == "RELIABILITY": self.unreliable_mode = False
            return None

        m_remember = re.match(r"REMEMBER (\w+)", line)
        if m_remember and not re.match(r"@\d+", m_remember.group(1)):
            thing = m_remember.group(1).upper()
            self.remembered.add(thing)
            if random.random() < 0.5:
                self.log(f"[REMEMBER] (Line {idx+1}) -- The program immediately forgets {thing}. Oops.")
            else:
                self.log(f"[REMEMBER] (Line {idx+1}) -- {thing}? Never heard of it.")
            return None

        # --- GOTO flow control ---
        m_goto = re.match(r"GOTO (\w+)", line)
        if m_goto:
            label = m_goto.group(1)
            if label in self.label_to_idx:
                target = self.label_to_idx[label]
                self.log(f"[GOTO] (Line {idx+1}) --> Jumping to '{label}' (line {target+1})")
                return target
            else:
                self.log(f"ERROR: (Line {idx+1}) -- Label '{label}' not found for GOTO. Continuing.")
                return None

        # --- Unreliable mode check ---
        if self.unreliable_mode and random.random() < self.unreliable_prob:
            self.log(f"[UNRELIABLE] (Line {idx+1}) -- Hardware failed to execute command. Try wishing harder.")
            return None
        
        # --- Extract command and qubits ---
        parts = line.split()
        cmd = parts[0].upper()
        qubits = [int(q) for q in re.findall(r'@(\d+)', line)]

        # --- Check for abstained commands or ignored qubits ---
        if cmd in self.abstained:
            self.log(f"[ABSTAINED] (Line {idx+1}) -- Command '{cmd}' is abstained. Skipping.")
            return None
        if any(q in self.ignored_qubits for q in qubits):
            self.log(f"[IGNORED] (Line {idx+1}) -- Command {cmd} on ignored qubit(s). Skipping.")
            return None

        # --- Process Quantum and State-modifying Commands ---
        gate_applied = False
        if cmd == "QUANTIZE":
            self._apply_quantum_gate(self.circuit.add_H_gate, qubits, "H")
            gate_applied = True
        elif cmd == "QNOT":
            self._apply_quantum_gate(self.circuit.add_X_gate, qubits, "X")
            gate_applied = True
        elif cmd == "QSWAP":
            self._apply_quantum_gate2(self.circuit.add_SWAP_gate, qubits, "SWAP")
            gate_applied = True
        elif cmd == "QROTATE":
            self._apply_quantum_gate(self.circuit.add_T_gate, qubits, "T")
            gate_applied = True
        elif cmd == "QUNROTATE":
            self._apply_quantum_gate(self.circuit.add_Tdag_gate, qubits, "Tdag")
            gate_applied = True
        elif cmd == "QPHASE":
            self._apply_quantum_gate(self.circuit.add_Z_gate, qubits, "Z")
            gate_applied = True
        elif cmd == "TRANSFORM":
            self._apply_quantum_gate(self.circuit.add_H_gate, qubits, "H")
            gate_applied = True
        elif cmd == "QCONTROL":
            self._apply_quantum_gate2(self.circuit.add_CNOT_gate, qubits, "CNOT")
            gate_applied = True
        elif cmd == "QCCONTROL":
            self._apply_multi_controlled_X_gate(qubits)
            gate_applied = True
        elif cmd == "QMEASURE" or cmd == "OBSERVE":
            if qubits:
                res = self.measure_and_collapse(qubits[0])
                self.log(f"[{cmd}] (Line {idx+1}) -- Qubit @{qubits[0]} collapsed to {res}")
                self.command_history.append(f"{cmd}@{qubits[0]}={res}")
        elif cmd == "QRESET":
            if qubits: self.reset_qubit(qubits[0])
        elif cmd == "QRANDOM":
            self._qrandom(qubits, idx)
            gate_applied = True
        elif cmd == "QCHANCE":
            self._qchance(qubits, idx)
            gate_applied = True
        elif cmd == "IGNORE":
            if qubits: self.ignored_qubits.add(qubits[0])
        elif cmd == "REMEMBER" and qubits:
            if qubits[0] in self.ignored_qubits: self.ignored_qubits.remove(qubits[0])
        elif cmd == "FORGET":
            label_to_forget = parts[1] if len(parts) > 1 else None
            if label_to_forget and label_to_forget in self.label_to_idx:
                # Find the index associated with the label to also clear idx_to_label
                idx_to_remove = self.label_to_idx[label_to_forget]
                del self.label_to_idx[label_to_forget]
                
                # Remove the reverse mapping as well to maintain consistency
                if idx_to_remove in self.idx_to_label and self.idx_to_label[idx_to_remove] == label_to_forget:
                    del self.idx_to_label[idx_to_remove]
                
                self.log(f"[FORGET] (Line {idx+1}) -- Forgot label '{label_to_forget}'.")
        elif cmd == "READ" and len(parts) > 2 and parts[1].upper() == "OUT":
            target = parts[2] if len(parts) > 2 else None
            if target and target.startswith('@'):
                q = int(target[1:])
                res = self.measure_and_collapse(q)
                self.log(f"[READ OUT] (Line {idx+1}) -- Qubit @{q} measured as {res}")
            else:
                probs = np.abs(self.state)**2
                self.log(f"[READ OUT] (Line {idx+1}) -- State probabilities: {probs.tolist()}")
        elif cmd not in ["DO", "PLEASE"]:
             self.log(f"[NOTE] (Line {idx+1}) -- Unrecognized or non-operational command: {cmd}")
        
        # If any gate was applied, run the circuit and update state
        if gate_applied:
            self.state = run_circuit(self.circuit, self.state)
            self.circuit_all += self.circuit
            self.circuit = QuantumCircuit(self.num_qubits) # Reset for the next op
            self.state_history.append(self.state.copy())
        
        return None # No jump by default

    # --- Quantum Gate Applicators ---
    def _apply_quantum_gate(self, func, qubits, opname):
        if qubits:
            func(qubits[0])
            self.command_history.append(f"{opname}@{qubits[0]}")
        else: self.log(f"WARNING: No qubit specified for {opname}")

    def _apply_quantum_gate2(self, func, qubits, opname):
        if len(qubits) >= 2:
            func(qubits[0], qubits[1])
            self.command_history.append(f"{opname}@{qubits[0]}@{qubits[1]}")
        else: self.log(f"WARNING: Need 2 qubits for {opname}, got {len(qubits)}")

    def _apply_multi_controlled_X_gate(self, qubits):
        if len(qubits) >= 3:
            c1, c2, t = qubits[0], qubits[1], qubits[2]
            self.circuit.add_TOFFOLI_gate(c1, c2, t)
            self.command_history.append(f"CCNOT@{c1}@{c2}@{t}")
        else: self.log(f"WARNING: Need 3 qubits for QCCONTROL, got {len(qubits)}")

    # --- Special Quantum Commands ---
    def _qrandom(self, qubits, idx):
        if not qubits: return
        q = qubits[0]
        gate_map = { "X": self.circuit.add_X_gate, "H": self.circuit.add_H_gate, "Z": self.circuit.add_Z_gate, "T": self.circuit.add_T_gate }
        gate_name = random.choice(list(gate_map.keys()))
        gate_map[gate_name](q)
        self.command_history.append(f"QRANDOM:{gate_name}@{q}")
        self.log(f"[QRANDOM] (Line {idx+1}) -- Randomly inflicted a {gate_name} gate on qubit {q}.")

    def _qchance(self, qubits, idx):
        if not qubits: return
        q = qubits[0]
        if random.random() < 0.5:
            self.circuit.add_X_gate(q)
            self.command_history.append(f"QCHANCE:X@{q}")
            self.log(f"[QCHANCE] (Line {idx+1}) -- Decided to flip qubit {q} on a whim.")
        else:
            self.log(f"[QCHANCE] (Line {idx+1}) -- Left qubit {q} untouched. Such is the majesty of randomness.")

    # --- Measurement and State Collapse ---
    def measure_and_collapse(self, q):
        """Measures a qubit and collapses the state vector accordingly."""
        dim = 2**self.num_qubits
        # Create a mask for indices where qubit q is 1
        mask_1 = np.array([(i >> q) & 1 for i in range(dim)], dtype=bool)
        
        # Calculate the probability of measuring 1
        prob_1 = np.sum(np.abs(self.state[mask_1])**2)
        
        result = 1 if random.random() < prob_1 else 0
        
        # Collapse the state vector
        if result == 1:
            self.state[~mask_1] = 0.0 # Set amplitudes for |0> to zero
        else:
            self.state[mask_1] = 0.0  # Set amplitudes for |1> to zero

        # Re-normalize the state vector
        norm = np.linalg.norm(self.state)
        if norm > 1e-9: # Avoid division by zero
            self.state /= norm
        else:
            self.log(f"WARNING: State norm is zero after measurement of qubit {q}. This shouldn't happen.")

        self.state_history.append(self.state.copy())
        return result

    def reset_qubit(self, q):
        """Resets a qubit to the |0> state."""
        result = self.measure_and_collapse(q)
        if result == 1:
            # If it was |1>, apply an X gate to flip it to |0>
            reset_circ = QuantumCircuit(self.num_qubits)
            reset_circ.add_X_gate(q)
            self.state = run_circuit(reset_circ, self.state)
            self.circuit_all += reset_circ
            self.state_history.append(self.state.copy())
        self.command_history.append(f"RESET@{q}")
        self.log(f"[RESET] -- Qubit @{q} has been reset to |0>.")

    # --- Final Reporting ---
    def _print_politeness_summary(self):
        """Prints a summary of the program's politeness level."""
        if self.total_lines == 0: return
        
        polite_ratio = self.please_count / self.total_lines
        self.log("\n--- Politeness Summary ---")
        if self.please_count == 0:
            self.log("EXECUTION FAILED: No PLEASE detected! (INTRACAL demands at least some politeness!)")
        elif polite_ratio < 0.2:
            self.log("NOTE: This program is quite impolite. Try adding more PLEASE.")
        elif polite_ratio > 0.5:
            self.log(f"ERROR: You are being far too polite! ({self.please_count}/{self.total_lines} lines had PLEASE). INTRACAL disapproves of excessive courtesy.")
            self.log(f"INTRACAL says: \"{random.choice(self.snark)}\"")
        else:
            self.log("Politeness level: Acceptable. (For INTRACAL, anyway.)")


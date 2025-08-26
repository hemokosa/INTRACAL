from quri_parts.circuit import QuantumCircuit
from quri_parts.qulacs.simulator import run_circuit
from quri_parts.circuit.utils.circuit_drawer import draw_circuit

import numpy as np
import random
import re

class INTRACAL:
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
        self.pointer = 0
        self.state = self._init_state(init)
        self.circuit = QuantumCircuit(num_qubits)
        self.circuit_all = QuantumCircuit(num_qubits)
        self.state_history = []
        self.command_history = []
        self.log_msgs = []

        # Politeness tracking
        self.please_count = 0
        self.total_lines = 0
        self.polite_lines = []
        self.rude_lines = []

        # Joke/logic state
        self.giveup = False
        self.abstained = set()
        self.remembered = set()
        self.ignored_qubits = set()      # ← IGNORE/REMEMBER対応
        self.unreliable_mode = False
        self.unreliable_prob = 0.15

        # Snark
        self.snark = list(self.DEFAULT_SNARK)
        if snark:
            self.snark += list(snark)

    def _init_state(self, init):
        dim = 2**self.num_qubits
        if init is None:
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0
        elif isinstance(init, str):
            state = np.zeros(dim, dtype=complex)
            state[int(init, 2)] = 1.0
        else:
            state = np.array(init, dtype=complex)
        return state

    def add_snark(self, phrase):
        self.snark.append(phrase)

    def log(self, message):
        if self.debug:
            print(message)
        self.log_msgs.append(message)

    def set_Haar_random_state(self):
        dim = 2**self.num_qubits
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        self.state = state / np.linalg.norm(state)

    def set_zero_state(self):
        dim = 2**self.num_qubits
        self.state = np.zeros(dim, dtype=complex)
        self.state[0] = 1.0

    def estimate(self):
        """Estimate probability of pointer qubit being 1."""
        dim = 2**self.num_qubits
        prob = sum(np.abs(self.state[i])**2 for i in range(dim) if (i >> self.pointer) & 1)
        result = int(random.random() < prob)
        msg = f"Measured qubit {self.pointer}: 1 with probability {prob:.3f}, result: {result}"
        self.log(msg)
        return result

    def parse(self, code, max_steps=500):
        code_lines = self._prepare_code(code)
        label_to_idx, come_from_map = self._scan_labels(code_lines)
        self.label_to_idx = label_to_idx   # ← FORGETで書き換えるので属性化
        line_visited_from_goto = [False] * len(code_lines)
        self._reset_session_vars()
        last_measured = None

        i = 0
        steps = 0  # ステップカウンタ
        while i < len(code_lines) and steps < max_steps:
            line = code_lines[i]
            stripped = line.strip()

            # --- ラベル判定と分離（(LABEL)やLABEL:を処理） ---
            m1 = re.match(r'\((\w+)\)\s*(.*)', stripped)
            m2 = re.match(r'(\w+):\s*(.*)', stripped)
            if m1:
                label = m1.group(1)
                self.label_to_idx[label] = i
                stripped = m1.group(2).strip()
            elif m2:
                label = m2.group(1)
                self.label_to_idx[label] = i
                stripped = m2.group(2).strip()

            # --- ポリテネス/POLITENESS判定（PLEASE除去もここで） ---
            has_please = False
            if stripped.startswith("PLEASE"):
                has_please = True
                stripped = stripped[6:].strip()
            self.total_lines += 1
            if has_please:
                self.please_count += 1
                self.polite_lines.append(i + 1)
            else:
                self.rude_lines.append(i + 1)

            # --- "PLEASE"だけやラベルだけ行はスキップ ---
            if not stripped:
                i += 1
                steps += 1
                continue

            # --- FORGET LABEL ---
            m = re.match(r"FORGET (\w+)", stripped)
            if m:
                label = m.group(1)
                if label in self.label_to_idx:
                    del self.label_to_idx[label]
                    self.log(f"[FORGET] (Line {i+1}) -- Forgot label '{label}'.")
                else:
                    self.log(f"[FORGET] (Line {i+1}) -- Label '{label}' not found (already forgotten?).")
                i += 1
                steps += 1
                continue

            # --- IGNORE @n ---
            m = re.match(r"IGNORE @(\d+)", stripped)
            if m:
                q = int(m.group(1))
                self.ignored_qubits.add(q)
                self.log(f"[IGNORE] (Line {i+1}) -- Will ignore all operations on qubit {q}.")
                i += 1
                steps += 1
                continue

            # --- REMEMBER @n ---
            m = re.match(r"REMEMBER @(\d+)", stripped)
            if m:
                q = int(m.group(1))
                if q in self.ignored_qubits:
                    self.ignored_qubits.remove(q)
                    self.log(f"[REMEMBER] (Line {i+1}) -- No longer ignoring qubit {q}.")
                else:
                    self.log(f"[REMEMBER] (Line {i+1}) -- Qubit {q} was not ignored.")
                i += 1
                steps += 1
                continue

            # --- READ OUT @n or READ OUT ---
            m = re.match(r"READ OUT(?: @(\d+))?", stripped)
            if m:
                q = int(m.group(1)) if m.group(1) else None
                if q is not None:
                    result = self.measure_qubit(q)
                    self.log(f"[READ OUT] (Line {i+1}) -- Qubit {q} measured: {result}")
                    self.command_history.append(f"READOUT@{q}={result}")  # 履歴にも保存
                elif self.state_history:
                    last_state = self.state_history[-1]
                    probs = [float(abs(x)**2) for x in last_state]
                    self.log(f"[READ OUT] (Line {i+1}) -- State probabilities: {probs}")
                    self.command_history.append(f"READOUT_PROBS={probs}")
                else:
                    self.log(f"[READ OUT] (Line {i+1}) -- No state history to output.")
                    self.command_history.append("READOUT_PROBS=NONE")
                i += 1
                steps += 1
                continue

            # --- GIVE UP, ABSTAIN, REMEMBER（ラベル以外）, etc ---
            if self._handle_giveup_resume(stripped, i): i += 1; steps += 1; continue
            if self._handle_abstain_reinstate_remember(stripped, i): i += 1; steps += 1; continue
            if self.giveup and self._maybe_skip_due_to_giveup(i): i += 1; steps += 1; continue

            # COME FROM
            if self._handle_comefrom(stripped, i, line_visited_from_goto): i += 1; steps += 1; continue
            # GOTO
            goto_jumped = self._handle_goto(stripped, self.label_to_idx, i, line_visited_from_goto)
            if goto_jumped is not None:
                i = goto_jumped
                steps += 1
                continue
            if self._is_label_line(stripped): i += 1; steps += 1; continue

            # ABSTAIN: skip forbidden commands
            cmd_for_abstain, qubits = self._extract_command_and_qubits(stripped)
            if cmd_for_abstain and any(q in self.ignored_qubits for q in qubits):
                self.log(f"[IGNORED] (Line {i+1}) -- Command {cmd_for_abstain} skipped on ignored qubit(s) {set(qubits)&self.ignored_qubits}.")
                i += 1
                steps += 1
                continue
            if cmd_for_abstain and cmd_for_abstain in self.abstained:
                self.log(f"[ABSTAINED] (Line {i+1}) -- Command '{cmd_for_abstain}' is abstained. Skipping.")
                i += 1
                steps += 1
                continue
            if self.unreliable_mode and random.random() < self.unreliable_prob:
                unreliable_snarks = [
                    "The hardware failed to execute this command. Sorry, not sorry.",
                    "Quantum hardware decided not to cooperate. Please try wishing harder.",
                    "This command vanished into the quantum void. Maybe next time.",
                    "Execution failed. Have you tried turning it off and on again?",
                    "Your request was lost in superposition. Try again, or don't.",
                    "Apparently, this hardware is too unreliable for your ambitions.",
                    "The command was ignored, just like your last three emails.",
                    "The universe looked at your command and said 'Nope.'",
                    "Unreliable hardware strikes again. Lucky you.",
                    "This operation self-destructed out of sheer unreliability.",
                ]
                msg = random.choice(unreliable_snarks)
                self.log(f"[UNRELIABLE] (Line {i+1}) -- {msg}")
                i += 1
                steps += 1
                continue
            if self.please_count > 5 and random.random() < 0.3:
                self.log(f"[POLITENESS OVERFLOW] (Line {i+1}) -- The program is overwhelmed by courtesy and needs a break.")
                break

            self._run_command(stripped, i)
            i += 1
            steps += 1

        if steps >= max_steps:
            self.log(f"WARNING: Maximum step count ({max_steps}) reached. Possible infinite loop.")

        self._print_politeness_summary()
        self.log("INTRACAL Execution Completed")
        draw_circuit(self.circuit_all)
        return self.state, self.state_history, self.command_history, self.circuit_all, self.log_msgs

    # --- Utility methods and logic handlers（以降、前回と同様、関数名・構造も同じ） ---
    def _handle_giveup_resume(self, line, i):
        if line.startswith("GIVE UP"):
            self.giveup = True
            self.log(f"[GIVE UP] (Line {i+1}) -- The program has lost all hope. Further execution will be halfhearted at best.")
            return True
        if line.startswith("RESUME"):
            if self.giveup:
                if random.random() < 0.25:
                    self.log(f"[RESUME] (Line {i+1}) -- The program refuses to resume. Still giving up.")
                    return True
                else:
                    self.log(f"[RESUME] (Line {i+1}) -- Hope restored. (Barely.)")
            self.giveup = False
            return True
        return False

    def _handle_abstain_reinstate_remember(self, line, i):
        m = re.match(r"ABSTAIN (\w+)", line)
        if m:
            thing = m.group(1)
            self.abstained.add(thing)
            self.log(f"[ABSTAIN] (Line {i+1}) -- {thing} is now forbidden. (You probably won't notice.)")
            if thing == "RELIABILITY":
                self.unreliable_mode = True
                self.log(f"[ABSTAIN] (Line {i+1}) -- Entering unreliable mode. Expect nonsense.")
            return True
        m = re.match(r"REINSTATE (\w+)", line)
        if m:
            thing = m.group(1)
            if thing in self.abstained:
                self.abstained.remove(thing)
                self.log(f"[REINSTATE] (Line {i+1}) -- {thing} is back in business.")
            else:
                self.log(f"[REINSTATE] (Line {i+1}) -- {thing} wasn't abstained, but okay.")
            if thing == "RELIABILITY":
                self.unreliable_mode = False
                self.log(f"[REINSTATE] (Line {i+1}) -- Reliability has returned. (Maybe.)")
            return True
        # REMEMBER (qubit)は上のREMEMBER @nで処理
        m = re.match(r"REMEMBER (\w+)", line)
        if m and not re.match(r"\d+", m.group(1)):
            thing = m.group(1)
            self.remembered.add(thing)
            if random.random() < 0.5:
                self.log(f"[REMEMBER] (Line {i+1}) -- The program immediately forgets {thing}. Oops.")
            else:
                self.log(f"[REMEMBER] (Line {i+1}) -- {thing}? Never heard of it.")
            return True
        return False

    def _maybe_skip_due_to_giveup(self, i):
        if random.random() < 0.7:
            self.log(f"[SKIP] (Line {i+1}) -- GIVE UP: skipping this line in despair.")
            return True
        else:
            self.log(f"[GIVE UP] (Line {i+1}) -- Fine, I'll do it. But only because you insist.")
            return False

    def _handle_comefrom(self, line, i, line_visited_from_goto):
        if line.startswith("COME FROM"):
            m = re.match(r"COME FROM (\w+)\s*(.*)", line)
            if m:
                label = m.group(1)
                rest = m.group(2).strip()
                if line_visited_from_goto[i]:
                    self._run_command(rest, i)
            return True
        return False

    def _handle_goto(self, line, label_to_idx, i, line_visited_from_goto):
        m = re.match(r"GOTO (\w+)", line)
        if m:
            label = m.group(1)
            if label in label_to_idx:
                target = label_to_idx[label]
                self.log(f"GOTO {label} (line {target+1})")
                line_visited_from_goto[target] = True
                return target
            else:
                self.log(f"ERROR: Label '{label}' not found for GOTO.")
                return i + 1
        return None

    def _is_label_line(self, line):
        return bool(re.match(r'\(\w+\)', line) or re.match(r'\w+:', line))

    def _count_politeness(self, line, i):
        has_please = line.startswith("PLEASE")
        if has_please:
            self.please_count += 1
            self.polite_lines.append(i + 1)
        else:
            self.rude_lines.append(i + 1)

    def _print_politeness_summary(self):
        polite_ratio = self.please_count / max(1, self.total_lines)
        if self.please_count == 0:
            self.log("ERROR: No PLEASE detected! (INTRACAL demands politeness!)")
        elif polite_ratio < 0.1:
            self.log("WARNING: This program is quite impolite. Try adding more PLEASE.")
        elif polite_ratio > 0.5:
            self.log("ERROR: You are being way too polite! INTRACAL disapproves excessive courtesy.")
            self.log(random.choice(self.snark))
        elif polite_ratio > 0.3:
            self.log("WARNING: That's a lot of PLEASEs. INTRACAL values restraint in politeness.")
        else:
            self.log("Politeness level: Acceptable. (For INTRACAL, anyway.)")
        if self.rude_lines:
            self.log(f"Lines without PLEASE: {self.rude_lines} (How rude.)")
        if self.polite_lines:
            self.log(f"Lines with PLEASE: {self.polite_lines} (So courteous...)")

    def _reset_session_vars(self):
        self.please_count = 0
        self.total_lines = 0
        self.polite_lines = []
        self.rude_lines = []
        self.giveup = False
        self.abstained = set()
        self.remembered = set()
        self.ignored_qubits = set()
        self.unreliable_mode = False

    def _prepare_code(self, code):
        if isinstance(code, str):
            return code.strip().splitlines()
        return code

    def _scan_labels(self, code_lines):
        label_to_idx = {}
        come_from_map = {}
        for idx, line in enumerate(code_lines):
            l = line.strip()
            m = re.match(r'\((\w+)\)', l)
            if m:
                label_to_idx[m.group(1)] = idx
            m2 = re.match(r'(\w+):', l)
            if m2:
                label_to_idx[m2.group(1)] = idx
            m3 = re.match(r'COME FROM (\w+)', l)
            if m3:
                come_from_map[m3.group(1)] = idx
        return label_to_idx, come_from_map

    def _extract_command_and_qubits(self, line):
        m = re.match(r"DO (\w+)(.*)", line)
        if m:
            cmd = m.group(1).upper()
            qubits = [int(q.strip()) for q in re.findall(r"@(\d+)", m.group(2))]
            return cmd, qubits
        return None, []

    def _run_command(self, line, idx):
        cmd, qubits = self._extract_command_and_qubits(line)
        if not cmd:
            cmd = line.strip().upper()
        # 無視qubit
        if any(q in self.ignored_qubits for q in qubits):
            self.log(f"[IGNORED] (Line {idx+1}) -- Skipped command {cmd} for ignored qubit(s) {set(qubits) & self.ignored_qubits}.")
            return
        if cmd == "QUANTIZE":
            self._apply_quantum_gate(self.circuit.add_H_gate, qubits, "H")
        elif cmd == "QNOT":
            self._apply_quantum_gate(self.circuit.add_X_gate, qubits, "X")
        elif cmd == "QSWAP":
            self._apply_quantum_gate2(self.circuit.add_SWAP_gate, qubits, "SWAP")
        elif cmd == "QROTATE":
            self._apply_quantum_gate(self.circuit.add_T_gate, qubits, "T")
        elif cmd == "QUNROTATE":
            self._apply_quantum_gate(self.circuit.add_Tdag_gate, qubits, "D")
        elif cmd == "QPHASE":
            if "PHASE" in self.remembered and random.random() < 0.5:
                self.log(f"[REMEMBERED] (Line {idx+1}) -- Oops, forgot to apply PHASE as requested.")
                return
            self._apply_quantum_gate(self.circuit.add_Z_gate, qubits, "Z")
        elif cmd == "TRANSFORM":
            self._apply_quantum_gate(self.circuit.add_H_gate, qubits, "H")
        elif cmd == "QCONTROL":
            self._apply_quantum_gate2(self.circuit.add_CNOT_gate, qubits, "CNOT")
        elif cmd == "QCCONTROL":
            self._apply_multi_controlled_X_gate(qubits)
        elif cmd == "QMEASURE":
            self._measure_command(qubits, idx, "MEASURE")
        elif cmd == "OBSERVE":
            self._measure_command(qubits, idx, "OBSERVE")
        elif cmd == "QRESET":
            self._reset_qubit_cmd(qubits)
        elif cmd == "QCHANCE":
            self._qchance(qubits, idx)
        elif cmd == "QRANDOM":
            if qubits:
                q = qubits[0]
                # 適用するゲートをランダム選択
                gate = random.choice(["X", "H", "Z", "T"])
                if gate == "X":
                    self.circuit.add_X_gate(q)
                elif gate == "H":
                    self.circuit.add_H_gate(q)
                elif gate == "Z":
                    self.circuit.add_Z_gate(q)
                elif gate == "T":
                    self.circuit.add_T_gate(q)
                self.command_history.append(f"QRANDOM:{gate}@{q}")
                self.log(f"[QRANDOM] (Line {idx+1}) -- Randomly chose to inflict a {gate} gate on qubit {q}. Why not?")
        elif cmd == "":
            return
        elif cmd.startswith("READ OUT"):
            self.log(f"[READ OUT] (Line {idx+1}) -- Output requested. Not that it matters.")
        elif cmd.startswith("IGNORE"):
            self.log(f"[IGNORE] (Line {idx+1}) -- Ignoring whatever this is.")
        elif cmd.startswith("FORGET"):
            self.log(f"[FORGET] (Line {idx+1}) -- Pretending to forget something irrelevant.")
        elif cmd == "PLEASE":
            self.log("How polite of you.")
        else:
            self.log(f"[NOTE] (Line {idx+1}) -- Unrecognized or non-operational: {cmd}")

        if cmd in [
            "QUANTIZE", "QNOT", "QSWAP", "QROTATE", "QPHASE", "TRANSFORM",
            "QCONTROL", "QCCONTROL", "QRANDOM", "QCHANCE"
        ]:
            result = run_circuit(self.circuit, self.state)
            self.state = result
            self.circuit_all += self.circuit
            self.circuit = QuantumCircuit(self.num_qubits)
            self.state_history.append(self.state.copy())

    def _apply_quantum_gate(self, func, qubits, opname):
        if qubits:
            func(qubits[0])
            self.command_history.append(f"{opname}@{qubits[0]}")

    def _apply_quantum_gate2(self, func, qubits, opname):
        if len(qubits) >= 2:
            func(qubits[0], qubits[1])
            self.command_history.append(f"{opname}@{qubits[0]}@{qubits[1]}")

    def _apply_multi_controlled_X_gate(self, qubits):
        if len(qubits) >= 3:
            c1, c2, t = qubits
            self.circuit.add_TOFFOLI_gate(c1, c2, t)
            self.command_history.append(f"CCNOT@{c1}@{c2}@{t}")

    def _measure_command(self, qubits, idx, label):
        if qubits:
            result = self.measure_qubit(qubits[0])
            self.command_history.append(f"{label}@{qubits[0]}={result}")
            self.log(f"{label} Qubit @{qubits[0]}: {result}")

    def _reset_qubit_cmd(self, qubits):
        if qubits:
            self.reset_qubit(qubits[0])
            self.command_history.append(f"RESET@{qubits[0]}")

    def _qchance(self, qubits, idx):
        if not qubits:
            self.log(f"[CHANCE] (Line {idx+1}) -- No qubit specified. So much for contingency.")
            return
        q = qubits[0]
        applied = False
        # 50%で適用
        if random.random() < 0.5:
            self.circuit.add_X_gate(q)
            self.command_history.append(f"QCHANCE:X@{q}")
            applied = True
            # INTERCAL風
            snarks = [
                f"Decided to flip qubit {q} on a whim. Hope you're happy.",
                f"Applied X to qubit {q} because, honestly, why not?",
                f"Flipping bits randomly, just as the universe intended.",
                f"Applied X to qubit {q}. This is what passes for progress.",
            ]
            self.log(f"[QCHANCE] (Line {idx+1}) -- {random.choice(snarks)}")
        else:
            # 適用しなかった場合
            snarks = [
                f"Left qubit {q} untouched. Sometimes doing nothing is an art.",
                f"Skipped flipping qubit {q}. Don't say I never gave you anything.",
                f"Decided against doing anything to qubit {q}. Seemed appropriate.",
                f"Qubit {q} remains as it was. Such is the majesty of randomness.",
            ]
            self.log(f"[QCHANCE] (Line {idx+1}) -- {random.choice(snarks)}")

    def measure_qubit(self, q):
        dim = 2**self.num_qubits
        prob = sum(np.abs(self.state[i])**2 for i in range(dim) if (i >> q) & 1)
        return int(random.random() < prob)

    def reset_qubit(self, q):
        dim = 2**self.num_qubits
        prob_0 = sum(np.abs(self.state[i])**2 for i in range(dim) if not ((i >> q) & 1))
        measured = int(random.random() < prob_0)
        if measured == 0:  # |1⟩だったらX
            tmp_circ = QuantumCircuit(self.num_qubits)
            tmp_circ.add_X_gate(q)
            self.state = run_circuit(tmp_circ, self.state)

from quri_parts.circuit import QuantumCircuit
from quri_parts.qulacs.simulator import run_circuit
from quri_parts.circuit.utils.circuit_drawer import draw_circuit

import numpy as np
import random
import re
from typing import Iterable, Tuple, List


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

    def __init__(self, num_qubits: int, debug: bool = True, init=None, snark=None, rng=None):
        self.num_qubits = num_qubits
        self.debug = debug
        self.pointer = 0  # reserved (not used yet)
        self.rng = rng if rng is not None else random

        self.state = self._init_state(init)
        self.circuit = QuantumCircuit(num_qubits)
        self.circuit_all = QuantumCircuit(num_qubits)

        self.state_history: List[np.ndarray] = []
        self.command_history: List[str] = []
        self.log_msgs: List[str] = []

        # politeness & flow-control
        self.please_count = 0
        self.total_lines = 0
        self.polite_lines: List[int] = []
        self.rude_lines: List[int] = []
        self.giveup = False
        self.abstained = set()
        self.remembered = set()
        self.ignored_qubits = set()
        self.unreliable_mode = False
        self.unreliable_prob = 0.15

        self.snark = list(self.DEFAULT_SNARK)
        if snark:
            self.snark += list(snark)

        # label maps
        self.label_to_idx = {}

    # -------------------------
    # Utilities & setup
    # -------------------------
    def _init_state(self, init):
        dim = 2 ** self.num_qubits
        if init is None:
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0
        elif isinstance(init, str):
            state = np.zeros(dim, dtype=complex)
            state[int(init, 2)] = 1.0
        else:
            state = np.array(init, dtype=complex)
        return state

    def log(self, message: str):
        if self.debug:
            print(message)
        self.log_msgs.append(message)

    def _prepare_code(self, code) -> List[str]:
        if isinstance(code, str):
            return code.strip().splitlines()
        return list(code)

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
        self.command_history = []
        self.log_msgs = []

    def _scan_labels_and_comefroms(self, code_lines: List[str]) -> Tuple[dict, dict, dict]:
        """
        Returns:
          label_to_idx: label -> line index
          idx_to_label: line index -> label
          come_from_target_map: target label -> 'COME FROM' line index
        """
        label_to_idx = {}
        idx_to_label = {}
        come_from_target_map = {}
        for idx, line in enumerate(code_lines):
            stripped = line.strip()
            m_label1 = re.match(r'\((\w+)\)', stripped)
            m_label2 = re.match(r'(\w+):', stripped)
            if m_label1:
                label = m_label1.group(1)
                label_to_idx[label] = idx
                idx_to_label[idx] = label
            elif m_label2:
                label = m_label2.group(1)
                label_to_idx[label] = idx
                idx_to_label[idx] = label
            m_comefrom = re.match(r'COME FROM (\w+)', stripped)
            if m_comefrom:
                target = m_comefrom.group(1)
                if target not in come_from_target_map:
                    come_from_target_map[target] = idx
        return label_to_idx, idx_to_label, come_from_target_map

    # -------------------------
    # Parser / Executor
    # -------------------------
    def parse(self, code, max_steps: int = 500):
        code_lines = self._prepare_code(code)
        self.label_to_idx, idx_to_label, come_from_target_map = self._scan_labels_and_comefroms(code_lines)

        self._reset_session_vars()

        i = 0
        steps = 0

        # keep initial state in history for analysis convenience
        self.state_history = [self.state.copy()]

        while i < len(code_lines) and steps < max_steps:
            current_label = idx_to_label.get(i)
            if current_label and current_label in come_from_target_map:
                # COME FROM hijack: jump to line after the COME FROM line
                comefrom_idx = come_from_target_map[current_label]
                self.log(f"[COME FROM] (Line {i+1}) Hijacked by COME FROM at line {comefrom_idx+1}.")
                i = comefrom_idx + 1
                steps += 1
                continue

            line = code_lines[i]
            stripped = line.strip()

            # Remove label definition
            m1 = re.match(r'\((\w+)\)\s*(.*)', stripped)
            m2 = re.match(r'(\w+):\s*(.*)', stripped)
            if m1:
                stripped = m1.group(2).strip()
            elif m2:
                stripped = m2.group(2).strip()

            # Politeness check (also track line numbers)
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

            # Skip empty / "COME FROM" only lines
            if not stripped or stripped.startswith("COME FROM"):
                i += 1
                steps += 1
                continue

            # Handle GOTO before anything else
            goto_target = self._handle_goto(stripped, self.label_to_idx, i)
            if goto_target is not None:
                i = goto_target
                steps += 1
                continue

            # FORGET label
            m = re.match(r"FORGET (\w+)", stripped)
            if m:
                label = m.group(1)
                if label in self.label_to_idx:
                    del self.label_to_idx[label]
                    self.log(f"[FORGET] (Line {i+1}) Forgot label '{label}'.")
                else:
                    self.log(f"[FORGET] (Line {i+1}) Label '{label}' not found.")
                i += 1; steps += 1; continue

            # IGNORE @n
            m = re.match(r"IGNORE @(\d+)", stripped)
            if m:
                q = int(m.group(1))
                if self._validate_qubits([q], i):
                    self.ignored_qubits.add(q)
                    self.log(f"[IGNORE] (Line {i+1}) Will ignore all operations on qubit {q}.")
                i += 1; steps += 1; continue

            # REMEMBER @n (un-ignore)
            m = re.match(r"REMEMBER @(\d+)", stripped)
            if m:
                q = int(m.group(1))
                if q in self.ignored_qubits:
                    self.ignored_qubits.remove(q)
                    self.log(f"[REMEMBER] (Line {i+1}) No longer ignoring qubit {q}.")
                else:
                    self.log(f"[REMEMBER] (Line {i+1}) Qubit {q} was not ignored.")
                i += 1; steps += 1; continue

            # READ OUT [@n]
            m = re.match(r"READ OUT(?: @(\d+))?", stripped)
            if m:
                q = int(m.group(1)) if m.group(1) else None
                if q is not None:
                    if self._validate_qubits([q], i):
                        result = self.measure_qubit(q)  # projects & normalizes
                        self.log(f"[READ OUT] (Line {i+1}) Qubit {q} measured: {result}")
                        self.command_history.append(f"READOUT@{q}={result}")
                else:
                    vec = self.state  # latest state
                    probs = [float(abs(x) ** 2) for x in vec]
                    self.log(f"[READ OUT] (Line {i+1}) State probabilities: {probs}")
                    self.command_history.append(f"READOUT_PROBS={probs}")
                i += 1; steps += 1; continue

            # GIVE UP / RESUME
            if self._handle_giveup_resume(stripped, i):
                i += 1; steps += 1; continue

            # ABSTAIN / REINSTATE / REMEMBER (symbolic)
            if self._handle_abstain_reinstate_remember(stripped, i):
                i += 1; steps += 1; continue

            # Maybe skip line under GIVE UP mood
            if self.giveup and self._maybe_skip_due_to_giveup(i):
                i += 1; steps += 1; continue

            # Abstain check for DO-commands
            cmd_for_abstain, qubits = self._extract_command_and_qubits(stripped)
            if cmd_for_abstain:
                if not self._validate_qubits(qubits, i):
                    i += 1; steps += 1; continue
                if any(q in self.ignored_qubits for q in qubits):
                    self.log(f"[IGNORED] (Line {i+1}) Skipped command {cmd_for_abstain} on ignored qubit(s) {set(qubits) & self.ignored_qubits}.")
                    i += 1; steps += 1; continue
                if cmd_for_abstain in self.abstained:
                    self.log(f"[ABSTAINED] (Line {i+1}) Command '{cmd_for_abstain}' is abstained. Skipping.")
                    i += 1; steps += 1; continue

            # Unreliable mode may drop operations
            if self.unreliable_mode and self.rng.random() < self.unreliable_prob:
                self.log(f"[UNRELIABLE] (Line {i+1}) Operation dropped in unreliable mode.")
                i += 1; steps += 1; continue

            # Check if politeness is within the required range (20% to 50%)
            # We add a condition to avoid checking on the very first few lines.
            if self.total_lines > 5:  # Start checking after 5 lines have been processed
                polite_ratio = self.please_count / self.total_lines
                if not (0.20 <= polite_ratio <= 0.50):
                    self.log(
                        f"[POLITENESS DEVIATION] (Line {i+1}) Politeness ratio is {polite_ratio:.2f}, "
                        f"which is outside the acceptable range (0.20-0.50). Aborting."
                    )
                    break
            
            # Politeness overflow gag
            if self.please_count > 20 and self.rng.random() < 0.1:
                self.log(f"[POLITENESS OVERFLOW] (Line {i+1}) Overwhelmed by courtesy; aborting.")
                break

            # Execute
            self._run_command(stripped, i)
            i += 1
            steps += 1

        if steps >= max_steps:
            self.log(f"WARNING: Maximum step count ({max_steps}) reached. Possible infinite loop.")

        self._print_politeness_summary()
        self.log("INTRACAL Execution Completed")
        try:
            draw_circuit(self.circuit_all)
        except Exception as e:
            self.log(f"[DRAW] Failed to draw circuit: {e!r}")

        return self.state, self.state_history, self.command_history, self.circuit_all, self.log_msgs

    # -------------------------
    # Command helpers
    # -------------------------
    def _handle_goto(self, line: str, label_to_idx: dict, i: int):
        m = re.match(r"GOTO (\w+)", line)
        if m:
            label = m.group(1)
            if label in label_to_idx:
                target = label_to_idx[label]
                self.log(f"GOTO {label} (line {target+1})")
                return target
            else:
                self.log(f"ERROR: Label '{label}' not found for GOTO.")
                return i + 1
        return None

    def _extract_command_and_qubits(self, line: str) -> Tuple[str, List[int]]:
        # 1) DO FOO @a @b ...
        m = re.match(r"DO\s+(\w+)(.*)", line, flags=re.IGNORECASE)
        if m:
            cmd = m.group(1).upper()
            qubits = [int(q) for q in re.findall(r"@(\d+)", m.group(2))]
            return cmd, qubits
        # 2) Plain FOO @a @b ...
        m2 = re.match(r"(\w+)(.*)", line)
        if m2:
            cmd = m2.group(1).upper()
            qubits = [int(q) for q in re.findall(r"@(\d+)", m2.group(2))]
            return cmd, qubits
        return None, []

    def _validate_qubits(self, qubits: Iterable[int], idx: int) -> bool:
        ok = True
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                self.log(f"[ERROR] (Line {idx+1}) Qubit index @{q} out of range (0..{self.num_qubits-1}). Command skipped.")
                ok = False
        return ok

    # -------------------------
    # Quantum operations & state updates
    # -------------------------
    def _apply_quantum_gate(self, func, qubits: List[int], opname: str):
        if qubits:
            func(qubits[0])
            self.command_history.append(f"{opname}@{qubits[0]}")

    def _apply_quantum_gate2(self, func, qubits: List[int], opname: str):
        if len(qubits) >= 2:
            func(qubits[0], qubits[1])
            self.command_history.append(f"{opname}@{qubits[0]}@{qubits[1]}")

    def _apply_multi_controlled_X_gate(self, qubits: List[int]):
        if len(qubits) >= 3:
            c1, c2, t = qubits[0], qubits[1], qubits[2]
            self.circuit.add_TOFFOLI_gate(c1, c2, t)
            self.command_history.append(f"CCNOT@{c1}@{c2}@{t}")

    def _flush_block(self):
        """Run current block circuit, merge to circuit_all, reset block, append state history."""
        if self.circuit is None:
            return
        self.state = run_circuit(self.circuit, self.state)
        try:
            self.circuit_all += self.circuit
        except TypeError:
            self.circuit_all = self.circuit_all + self.circuit
        self.circuit = QuantumCircuit(self.num_qubits)
        self.state_history.append(self.state.copy())

    def measure_qubit(self, q: int) -> int:
        """Projective measurement on qubit q with state update."""
        dim = 2 ** self.num_qubits
        mask = np.array([((i >> q) & 1) == 1 for i in range(dim)], dtype=bool)
        amp = self.state.copy()

        p1 = float(np.vdot(amp[mask], amp[mask]).real)
        outcome = 1 if self.rng.random() < p1 else 0

        if outcome == 1:
            amp[~mask] = 0.0
        else:
            amp[mask] = 0.0

        norm = np.linalg.norm(amp)
        if norm > 0:
            self.state = amp / norm
        else:
            # Fallback (numerical safety)
            self.state = np.zeros_like(amp)
            self.state[0] = 1.0

        self.state_history.append(self.state.copy())
        return outcome

    def reset_qubit(self, q: int):
        """Reset qubit q to |0⟩ by measuring then flipping if needed."""
        outcome = self.measure_qubit(q)
        if outcome == 1:
            tmp = QuantumCircuit(self.num_qubits)
            tmp.add_X_gate(q)
            self.state = run_circuit(tmp, self.state)
            self.state_history.append(self.state.copy())

    # -------------------------
    # High-level command runner
    # -------------------------
    def _run_command(self, line: str, idx: int):
        cmd, qubits = self._extract_command_and_qubits(line)
        if not cmd:
            return

        # respect ignored qubits early
        if any(q in self.ignored_qubits for q in qubits):
            self.log(f"[IGNORED] (Line {idx+1}) Skipped {cmd} on ignored qubit(s) {set(qubits) & self.ignored_qubits}.")
            return

        # Command and Memory Keyword Correspondence Dictionary
        command_to_remember_key = {
            "QUANTIZE": "H",
            "TRANSFORM": "H",
            "QNOT": "NOT",
            "QSWAP": "SWAP",
            "QROTATE": "T",
            "QUNROTATE": "TDG",
            "QPHASE": "PHASE",
            "QCONTROL": "CNOT",
            "QCCONTROL": "CCNOT",
        }

        # Common "Forget" Check
        remember_key = command_to_remember_key.get(cmd)
        if remember_key and remember_key in self.remembered and self.rng.random() < 0.5:
            self.log(f"[REMEMBERED] (Line {idx+1}) Oops, forgot to apply {remember_key}.")
            return 

        # Mapping of mnemonics
        if cmd in ("QUANTIZE", "TRANSFORM"):       # H
            self._apply_quantum_gate(self.circuit.add_H_gate, qubits, "H")
        elif cmd == "QNOT":                         # X
            self._apply_quantum_gate(self.circuit.add_X_gate, qubits, "X")
        elif cmd == "QSWAP":                        # SWAP
            self._apply_quantum_gate2(self.circuit.add_SWAP_gate, qubits, "SWAP")
        elif cmd == "QROTATE":                      # T
            self._apply_quantum_gate(self.circuit.add_T_gate, qubits, "T")
        elif cmd == "QUNROTATE":                    # T†
            if qubits:
                q = qubits[0]
                # environment compatibility
                try:
                    self.circuit.add_Tdag_gate(q)
                except AttributeError:
                    try:
                        self.circuit.add_T_dag_gate(q)
                    except AttributeError:
                        # fallback: Z^(1/4) inverse is not in base; log instead
                        self.log(f"[WARN] (Line {idx+1}) T† gate not available; skipped on @{q}.")
                        return
                self.command_history.append(f"Tdg@{q}")
        elif cmd == "QPHASE":                       # Z
            self._apply_quantum_gate(self.circuit.add_Z_gate, qubits, "Z")
        elif cmd == "QCONTROL":                     # CNOT
            self._apply_quantum_gate2(self.circuit.add_CNOT_gate, qubits, "CNOT")
        elif cmd == "QCCONTROL":                    # Toffoli
            self._apply_multi_controlled_X_gate(qubits)
        elif cmd in ("QMEASURE", "OBSERVE"):
            self._measure_command(qubits, idx, cmd.replace("Q", ""))  # "MEASURE"/"OBSERVE"
            return  # measurement updates state immediately; do not flush empty block
        elif cmd == "QRESET":
            if qubits:
                self.reset_qubit(qubits[0])
                self.command_history.append(f"RESET@{qubits[0]}")
            return
        elif cmd == "QCHANCE":
            self._qchance(qubits, idx)
        elif cmd == "QRANDOM":
            if qubits:
                q = qubits[0]
                gate = self.rng.choice(["X", "H", "Z", "T"])
                if gate == "X":
                    self.circuit.add_X_gate(q)
                elif gate == "H":
                    self.circuit.add_H_gate(q)
                elif gate == "Z":
                    self.circuit.add_Z_gate(q)
                elif gate == "T":
                    self.circuit.add_T_gate(q)
                self.command_history.append(f"QRANDOM:{gate}@{q}")
                self.log(f"[QRANDOM] (Line {idx+1}) Randomly chose {gate} on qubit {q}.")
        elif cmd == "PLEASE":
            self.log("How polite of you.")
            return
        elif cmd.startswith("READ"):
            # handled earlier in parse()
            return
        elif cmd.startswith("IGNORE") or cmd.startswith("FORGET") or cmd.startswith("REMEMBER"):
            # handled earlier
            return
        else:
            self.log(f"[NOTE] (Line {idx+1}) Unrecognized or non-operational: {cmd}")
            return

        # If we reached here, a gate was appended to the current block; flush it
        self._flush_block()

    def _measure_command(self, qubits: List[int], idx: int, label: str):
        if not qubits:
            self.log(f"[{label}] (Line {idx+1}) No qubit specified.")
            return
        q = qubits[0]
        if not self._validate_qubits([q], idx):
            return
        result = self.measure_qubit(q)
        self.command_history.append(f"{label}@{q}={result}")
        self.log(f"[{label}] (Line {idx+1}) Qubit @{q}: {result}")

    def _qchance(self, qubits: List[int], idx: int):
        if not qubits:
            self.log(f"[CHANCE] (Line {idx+1}) No qubit specified.")
            return
        q = qubits[0]
        if self.rng.random() < 0.5:
            self.circuit.add_X_gate(q)
            self.command_history.append(f"QCHANCE:X@{q}")
            self.log(f"[QCHANCE] (Line {idx+1}) Applied X to @{q} by chance.")
            self._flush_block()
        else:
            self.log(f"[QCHANCE] (Line {idx+1}) Did nothing to @{q} by chance.")

    # -------------------------
    # Narrative gimmicks
    # -------------------------
    def _handle_giveup_resume(self, line: str, i: int) -> bool:
        if line.startswith("GIVE UP"):
            self.giveup = True
            self.log(f"[GIVE UP] (Line {i+1}) The program has lost all hope.")
            return True
        if line.startswith("RESUME"):
            if self.giveup:
                if self.rng.random() < 0.25:
                    self.log(f"[RESUME] (Line {i+1}) Refuses to resume. Still giving up.")
                    return True
                else:
                    self.log(f"[RESUME] (Line {i+1}) Hope restored. Barely.")
            self.giveup = False
            return True
        return False

    def _handle_abstain_reinstate_remember(self, line: str, i: int) -> bool:
        m = re.match(r"ABSTAIN (\w+)", line)
        if m:
            thing = m.group(1).upper()
            self.abstained.add(thing)
            self.log(f"[ABSTAIN] (Line {i+1}) {thing} is now forbidden.")
            if thing == "RELIABILITY":
                self.unreliable_mode = True
                self.log(f"[ABSTAIN] (Line {i+1}) Entering unreliable mode.")
            return True

        m = re.match(r"REINSTATE (\w+)", line)
        if m:
            thing = m.group(1).upper()
            if thing in self.abstained:
                self.abstained.remove(thing)
                self.log(f"[REINSTATE] (Line {i+1}) {thing} reinstated.")
            else:
                self.log(f"[REINSTATE] (Line {i+1}) {thing} wasn't abstained.")
            if thing == "RELIABILITY":
                self.unreliable_mode = False
                self.log(f"[REINSTATE] (Line {i+1}) Reliability restored.")
            return True

        # REMEMBER (symbolic, not @n)
        m = re.match(r"REMEMBER (\w+)", line)
        if m and not re.match(r"\d+", m.group(1)):
            thing = m.group(1).upper()
            if self.rng.random() < 0.5: # 50% chance of failure (forgetting)
                self.log(f"[REMEMBER] (Line {i+1}) Failed to remember {thing}.")
            else: # The remaining 50% chance of success (remembering)
                self.remembered.add(thing)
                self.log(f"[REMEMBER] (Line {i+1}) Okay, I will try to remember {thing}.")
            return True

        return False

    def _maybe_skip_due_to_giveup(self, i: int) -> bool:
        if self.rng.random() < 0.7:
            self.log(f"[SKIP] (Line {i+1}) GIVE UP: skipping this line in despair.")
            return True
        else:
            self.log(f"[GIVE UP] (Line {i+1}) Fine, I'll do it.")
            return False

    def _print_politeness_summary(self):
        polite_ratio = self.please_count / max(1, self.total_lines)
        if self.please_count == 0:
            self.log("ERROR: No PLEASE detected! (INTRACAL demands politeness!)")
        elif polite_ratio < 0.1:
            self.log("WARNING: This program is quite impolite. Try adding more PLEASE.")
        elif polite_ratio > 0.5:
            self.log("ERROR: You are being way too polite! INTRACAL disapproves excessive courtesy.")
            self.log(self.rng.choice(self.snark))
        elif polite_ratio > 0.3:
            self.log("WARNING: That's a lot of PLEASEs. INTRACAL values restraint in politeness.")
        else:
            self.log("Politeness level: Acceptable. (For INTRACAL, anyway.)")

        if self.rude_lines:
            self.log(f"Lines without PLEASE: {self.rude_lines}")
        if self.polite_lines:
            self.log(f"Lines with PLEASE: {self.polite_lines}")

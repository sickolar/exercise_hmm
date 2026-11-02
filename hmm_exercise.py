import itertools
import json
import math
from Bio import SeqIO


def log2(p):
    if p == 0:
        return -math.inf
    return math.log2(p)


class HMM:
    def __init__(self, states, alphabet, pi, a, e):
        state_set = set(states)
        alphabet_set = set(alphabet)

        assert set(pi.keys()) == state_set
        assert set(a.keys()) == state_set
        assert set(e.keys()) == state_set
        assert math.isclose(sum(pi.values()), 1.0)

        for s in states:
            assert set(a[s].keys()) == state_set
            assert math.isclose(sum(a[s].values()), 1.0)
            assert set(e[s].keys()) == alphabet_set
            assert math.isclose(sum(e[s].values()), 1.0)

        self.states = states
        self.alphabet = alphabet
        self.pi = pi
        self.a = a
        self.e = e

        self.log_pi = {s: log2(pi[s]) for s in states}
        self.log_a = {s1: {s2: log2(a[s1][s2]) for s2 in states} for s1 in states}
        self.log_e = {s: {c: log2(e[s][c]) for c in alphabet} for s in states}

    def joint_probability(self, x, z):
        prob = self.pi[z[0]] * self.e[z[0]][x[0]]
        for i in range(1, len(x)):
            prob *= self.a[z[i - 1]][z[i]] * self.e[z[i]][x[i]]
        return prob

    def log_joint(self, x, z):
        score = self.log_pi[z[0]] + self.log_e[z[0]][x[0]]
        for i in range(1, len(x)):
            score += self.log_a[z[i - 1]][z[i]] + self.log_e[z[i]][x[i]]
        return score

    def brute_force(self, x):
        best_path = ""
        best_prob = -1.0
        total_prob = 0.0

        for path_tuple in itertools.product(self.states, repeat=len(x)):
            z = "".join(path_tuple)
            p = self.joint_probability(x, z)
            total_prob += p
            if p > best_prob:
                best_prob = p
                best_path = z

        return best_path, best_prob, total_prob

    def viterbi(self, x):
        v = []
        back = []

        v0 = {}
        back0 = {}
        for k in self.states:
            v0[k] = self.log_pi[k] + self.log_e[k][x[0]]
            back0[k] = None
        v.append(v0)
        back.append(back0)

        for i in range(1, len(x)):
            vi = {}
            backi = {}
            for k in self.states:
                best_score = -math.inf
                best_prev = self.states[0]
                for l in self.states:
                    score = v[i - 1][l] + self.log_a[l][k] + self.log_e[k][x[i]]
                    if score > best_score:
                        best_score = score
                        best_prev = l
                vi[k] = best_score
                backi[k] = best_prev
            v.append(vi)
            back.append(backi)

        best_last_score = -math.inf
        best_last_state = self.states[0]
        for k in self.states:
            if v[-1][k] > best_last_score:
                best_last_score = v[-1][k]
                best_last_state = k

        path = [best_last_state]
        curr = best_last_state
        for i in range(len(x) - 1, 0, -1):
            curr = back[i][curr]
            path.append(curr)
        path.reverse()

        return "".join(path), best_last_score

    def viterbi_plain(self, x):
        v = [{k: self.pi[k] * self.e[k][x[0]] for k in self.states}]
        back = [{k: None for k in self.states}]

        for i in range(1, len(x)):
            vi = {}
            backi = {}
            for k in self.states:
                best_prob = -1.0
                best_prev = self.states[0]
                for l in self.states:
                    prob = v[i - 1][l] * self.a[l][k] * self.e[k][x[i]]
                    if prob > best_prob:
                        best_prob = prob
                        best_prev = l
                vi[k] = best_prob
                backi[k] = best_prev
            v.append(vi)
            back.append(backi)

        best_last_prob = -1.0
        best_last_state = self.states[0]
        for k in self.states:
            if v[-1][k] > best_last_prob:
                best_last_prob = v[-1][k]
                best_last_state = k

        path = [best_last_state]
        curr = best_last_state
        for i in range(len(x) - 1, 0, -1):
            curr = back[i][curr]
            path.append(curr)
        path.reverse()

        return "".join(path), best_last_prob


def read_model(path):
    with open(path, "r") as f:
        data = json.load(f)
    return HMM(
        data["states"],
        data["alphabet"],
        data["pi"],
        data["a"],
        data["e"],
    )


if __name__ == "__main__":
    weather = read_model("weather_model.json")
    toy = read_model("toy_model.json")

    print("States:", weather.states, "Alphabet:", weather.alphabet)
    print('joint_probability("sr", "HL"):', weather.joint_probability("sr", "HL"))
    print('log_joint("sr", "HL"):', weather.log_joint("sr", "HL"))
    print('brute_force("ssrssrrrss"):', weather.brute_force("ssrssrrrss"))
    print('viterbi("ssrssrrrss"):', weather.viterbi("ssrssrrrss"))

    for model in (weather, toy):
        for length in range(1, 5):
            for x_tuple in itertools.product(model.alphabet, repeat=length):
                x = "".join(x_tuple)
                for z_tuple in itertools.product(model.states, repeat=length):
                    z = "".join(z_tuple)
                    assert math.isclose(
                        model.log_joint(x, z),
                        log2(model.joint_probability(x, z)),
                    )

        for x_tuple in itertools.product(model.alphabet, repeat=5):
            x = "".join(x_tuple)
            v_path, v_score = model.viterbi(x)
            bf_path, bf_prob, _ = model.brute_force(x)
            assert v_path == bf_path
            assert math.isclose(2 ** v_score, bf_prob)
            assert math.isclose(v_score, model.log_joint(x, v_path))

    for record in SeqIO.parse("sequences.fasta", "fasta"):
        seq = str(record.seq)
        v_path, v_score = toy.viterbi(seq)
        p_path, p_prob = toy.viterbi_plain(seq)
        bits_per_pos = v_score / len(seq)
        print(
            f"{record.id} (len={len(seq)}): "
            f"log_score={v_score:.2f}, bits/pos={bits_per_pos:.2f}, "
            f"plain_prob={p_prob:.3e}, paths_match={v_path == p_path}"
        )

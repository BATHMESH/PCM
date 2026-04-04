import numpy as np
from gnuradio import gr

class quantizer(gr.interp_block):
    def __init__(self, n_bits=8, v_max=1.0, v_min=-1.0, dtype="int"):
        self.n_bits = int(n_bits)
        self.v_max = float(v_max)
        self.v_min = float(v_min)
        self.dtype = dtype

        interp = self.n_bits if self.dtype == "int" else 1

        gr.interp_block.__init__(
            self,
            name="PCM Binary Encoder",
            in_sig=[np.float32],
            out_sig=[np.int32 if self.dtype == "int" else np.float32],
            interp=interp
        )

        self._update_params()

    def _update_params(self):
        self.levels = 2 ** self.n_bits
        self.delta = (self.v_max - self.v_min) / (self.levels - 1)

    def set_n_bits(self, n_bits):
        self.n_bits = int(n_bits)
        self._update_params()

    def work(self, input_items, output_items):
        in_data = input_items[0]
        out_data = output_items[0]

        q_signal = self.delta * np.round(in_data / self.delta)
        q_signal = np.clip(q_signal, self.v_min, self.v_max)

        if self.dtype == "int":
            indices = np.round((q_signal - self.v_min) / self.delta).astype(np.int32)
            indices = np.clip(indices, 0, self.levels - 1)

            bit_matrix = np.zeros((len(indices), self.n_bits), dtype=np.int32)

            for i in range(self.n_bits):
                bit_matrix[:, i] = (indices >> (self.n_bits - 1 - i)) & 1

            out_data[:] = bit_matrix.flatten()
        else:
            out_data[:] = q_signal

        return len(out_data)
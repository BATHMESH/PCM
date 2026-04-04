import numpy as np
from gnuradio import gr
from scipy import signal

class pcm_decoder(gr.decim_block):
    def __init__(self, n_bits=8, v_max=1.0, v_min=-1.0, samp_rate=32000, cutoff_freq=400):
        self.n_bits = int(n_bits)
        self.v_max = float(v_max)
        self.v_min = float(v_min)
        self.samp_rate = samp_rate
        self.cutoff_freq = cutoff_freq
        
        gr.decim_block.__init__(
            self,
            name="PCM Binary Decoder",
            in_sig=[np.int32],
            out_sig=[np.float32],
            decim=self.n_bits
        )
        
        self.zi = None
        self._update_params()

    def _update_params(self):
        self.levels = 2 ** self.n_bits
        self.delta = (self.v_max - self.v_min) / (self.levels - 1)
        
        
        # Bessel filters minimize phase distortion
        b, a = signal.bessel(4, self.cutoff_freq / (self.samp_rate / 2.0), btype='low')
        self.b = b
        self.a = a
        self.zi = signal.lfilter_zi(b, a) * 0

    def work(self, input_items, output_items):
        in_bits = input_items[0]
        out_signal = output_items[0]
        
        # Calculate how many full PCM samples we can reconstruct
        n_samples = len(in_bits) // self.n_bits
        
        # Reshape bits and convert to indices
        binary_bits = in_bits[:n_samples * self.n_bits].reshape(-1, self.n_bits)
        powers = 2**np.arange(self.n_bits - 1, -1, -1)
        indices = np.sum(binary_bits * powers, axis=1)
        
        # Reconstruct quantized values
        q_reconstructed = (indices.astype(float) * self.delta) + self.v_min
        
        # Apply low-pass filter to smooth the output
        filtered, self.zi = signal.lfilter(self.b, self.a, q_reconstructed, zi=self.zi)
        
        out_signal[:len(filtered)] = filtered
        return len(filtered)

class config:
    def __init__(self):
        self.lr= 0.01
        self.batch_size = 32
        self.n_epochs = 10
        self.n_fft= 512
        self.hop_length_fft= 256
        self.fs= 8000
        self.height= 257
        self.width= 313
        self.sig_legnth= 80000

        # Data parameters
        self.data_path = "/path/to/data"


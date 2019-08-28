#
# Settings for SEGAN
#

class settings:

    def __init__(self):

        # General Settings
        self.len                = 2 ** 14           # Input Size (len = 16384)
        self.device_id          = 0                 # GPU ID (init:0)
        self.random_seed        = 0
        self.halfprec           = False              # 16Bit or not

        # Parameters
        self.batch_size 	    = 16               # Batch size
        self.epoch              = 20               # Epoch
        self.learning_rate      = 0.0001            # Learning Rate (Generator)
        self.weight_decay       = 0.0001
        self.decay_rate1        = 0.9
        self.decay_rate2        = 0.99

        # Retrain
        self.epoch_from         = 0                   # Epoch No. from that Retraining starts (init:0)

        # Save path
        self.model_save_path    = 'params'          # Network model path
        self.model_save_cycle   = 20                 # Epoch cycle for saving model (init:1)

        # Save wav path
        self.wav_save_path      = ('pred_%d'%(self.epoch))

        # Wave files
        self.clean_train_path   = 'C:/Users/InternShip/Documents/CPS/segan/data/clean_trainset_wav_16k'     # Folder containing clean wav (train)
        self.noisy_train_path   = 'C:/Users/InternShip/Documents/CPS/segan/data/noisy_trainset_wav_16k'     # Folder containing noisy wav (train)
        self.clean_test_path    = 'C:/Users/InternShip/Documents/CPS/segan/data/clean_testset_wav_16k'      # Folder containing clean wav (test)
        self.noisy_test_path    = 'C:/Users/InternShip/Documents/CPS/segan/data/noisy_testset_wav_16k'      # Folder containing noisy wav (test)

        # Pkl files for train
        self.train_pkl_path     = 'C:/Users/InternShip/Documents/CPS/segan/pkl'             # Folder of pkl files for train
        self.train_pkl_clean    = 'train_clean.pkl' # File name of "Clean" pkl for train
        self.train_pkl_noisy    = 'train_noisy.pkl' # File name of "Noisy" pkl for train

        # Pkl files for test
        self.test_pkl_path      = 'C:/Users/InternShip/Documents/CPS/segan/pkl'             # Folder of pkl files for test
        self.test_pkl_clean     = 'test_clean.pkl'  # File name of "Clean" pkl for test
        self.test_pkl_noisy     = 'test_noisy.pkl'  # File name of "Noisy" pkl for test
        self.test_pkl_length    = 'test_length.pkl' # File name of "Length" pkl for test

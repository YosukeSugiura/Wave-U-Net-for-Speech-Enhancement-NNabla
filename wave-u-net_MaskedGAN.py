# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from six.moves import range

import os
import time
import numpy as np

# NNabla
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.initializer import ConstantInitializer as Init_cnst
from nnabla.initializer import NormalInitializer as Init_nml

from nnabla.ext_utils import get_extension_context

# Display
import pyqtgraph as pg
import pyqtgraph.exporters as pgex

# Sound
from scipy.io import wavfile
from pypesq import pypesq

# Original Functions
from settings import settings
import data as dt

### ================================================================
###
###   Sub Functions
###
### ================================================================
##  Convolution
def conv(x, output_ch, karnel=(15,), pad=(7,), stride=(1,), name=None):
    return PF.convolution(x, output_ch, karnel, pad=pad, stride=stride,
                        w_init=Init_nml(sigma=0.02), b_init=Init_cnst(), name=name)

##  Deconvolution
def deconv(x, output_ch, karnel=(1,), pad=(0,), stride=(1,), name=None):
    return PF.deconvolution(x, output_ch, karnel, pad=pad, stride=stride,
                        w_init=Init_nml(sigma=0.02), b_init=Init_cnst(), name=name)

##  Activation Function
def af(x, alpha=0.2):
    return F.leaky_relu(x, alpha)

##  Crop & Concat
def crop_and_concat(x1, x2):

    def crop(tensor, target_times):
        diff = tensor.shape[2] - target_times
        if diff == 0:
            return tensor
        crop_start = diff // 2
        crop_end = diff - crop_start
        return F.slice(tensor, start=(0, 0, crop_start), stop=(tensor.shape[0], tensor.shape[1], tensor.shape[2] - crop_end), step=(1, 1, 1))

    return F.concatenate(crop(x1, x2.shape[2]), x2, axis=1)

##  Display progress in console
class display:

    # Remaining Time Estimation
    class time_estimation:

        def __init__(self, epoch_from, epoch, batch_num):
            self.start = time.time()
            self.epoch = epoch
            self.epoch_from = epoch_from
            self.batch = batch_num
            self.all = batch_num * (epoch - epoch_from)

        def __call__(self, epoch_num, batch_num):
            elapse = time.time() - self.start
            amount = (batch_num + 1) + (epoch_num - self.epoch_from) * self.batch
            remain = elapse / amount * (self.all - amount)

            hours, mins = divmod(elapse, 3600)
            mins, sec = divmod(mins, 60)
            hours_e, mins_e = divmod(remain, 3600)
            mins_e, sec_e = divmod(mins_e, 60)

            elapse_time = [int(hours), int(mins), int(sec)]
            remain_time = [int(hours_e), int(mins_e), int(sec_e)]

            return elapse_time, remain_time

    def __init__(self, epoch_from, epoch, batch_num):

        self.tm = self.time_estimation(epoch_from, epoch, batch_num)
        self.batch = batch_num

    def __call__(self, epoch, trial, losses):

        elapse_time, remain_time = self.tm(epoch, trial)
        print('  ---------------------------------------------------')
        print('  [ Epoch  # {0},    Trials  # {1}/{2} ]'.format(epoch + 1, trial + 1, self.batch))
        print('    +  Mean Squared Loss        = {:.4f}'.format(losses))
        print('    -------------------------')
        print('    +  Elapsed Time            : {0[0]:3d}h {0[1]:02d}m {0[2]:02d}s'.format(elapse_time))
        print('    +  Expected Remaining Time : {0[0]:3d}h {0[1]:02d}m {0[2]:02d}s'.format(remain_time))
        print('  ---------------------------------------------------')

##  Create figure object and plot
class figout:
    def __init__(self):

        ## Create Graphic Window
        self.win = pg.GraphicsWindow(title="")
        self.win.resize(800, 600)
        self.win.setWindowTitle('pyqtgraph example: Plotting')
        self.win.setBackground("#FFFFFFFF")
        pg.setConfigOptions(antialias=True)     # Anti-Aliasing for clear plotting

        ## Graph Layout
        #   1st Col: Speech Waveform
        self.p1 = self.win.addPlot(title="Source 1 Waveform")
        self.p1.addLegend()
        self.c11 = self.p1.plot(pen=(255, 0, 0, 255), name="In")
        self.c12 = self.p1.plot(pen=(0, 255, 0, 150), name="Out1")
        self.c13 = self.p1.plot(pen=(0, 0, 255, 90), name="Clean")
        self.win.nextRow()
        self.p2 = self.win.addPlot(title="Source 2 Waveform")
        self.p2.addLegend()
        self.c21 = self.p2.plot(pen=(255, 0, 0, 255), name="In")
        self.c22 = self.p2.plot(pen=(0, 255, 0, 150), name="Out2")
        self.c23 = self.p2.plot(pen=(0, 0, 255, 90), name="Clean")
        self.win.nextRow()
        #   2st Col-1: Loss
        self.p3 = self.win.addPlot(title="Loss")
        self.p3.addLegend()
        self.c31 = self.p3.plot(pen=(255, 0, 0, 255), name="losses")
        self.win.nextRow()

    def waveform_1(self, noisy, target, clean, stride=10):
        self.c11.setData(noisy[0:-1:stride])
        self.c12.setData(target[0:-1:stride])
        self.c13.setData(clean[0:-1:stride])

    def waveform_2(self, noisy, target, clean, stride=10):
        self.c21.setData(noisy[0:-1:stride])
        self.c22.setData(target[0:-1:stride])
        self.c23.setData(clean[0:-1:stride])

    def loss(self, losses, stride=10):
        self.c31.setData(losses[0:-1:stride])

##  Display PESQ in console
def pesq_score(clean_wavs, reconst_wavs, band='wb'):

    scores    = []

    print('PESQ Calculation...')
    for i, (clean_, reconst_) in enumerate(zip(clean_wavs, reconst_wavs)):
        rate, ref = wavfile.read(clean_)
        rate, deg = wavfile.read(reconst_)
        score = pypesq(rate, ref, deg, band)
        scores.append(score)
        print('Score : {0:.4} ... {1}/{2}'.format(score, i, len(clean_wavs)))

    score = np.average(np.array(scores))


    print('  ---------------------------------------------------')
    print('  Average PESQ score = {0:.4}'.format(score))
    print('  ---------------------------------------------------')

    return 0


### ================================================================ ###
###                                                                  ###
###   Generator ( Encoder + Decoder )
###   - output estimated clean wav
###                                                                  ###
### ================================================================ ###
def Wave_U_Net(Noisy, num_initial_filters = 24, num_layers = 12, merge_filter_size = 5, latent = False):

    ds_outputs = list()

    ## --------------------------------- ##
    ##     Sub-functions                 ##
    ## --------------------------------- ##
    def downsampling_block(x, i):
        with nn.parameter_scope( ('ds_block-%2d' % i) ):
            ds = af(conv(x, (num_initial_filters+num_initial_filters*i), name='conv'))
            ds_slice = F.slice(ds, start=(0,0,0), stop=ds.shape, step=(1,1,2))      # Decimate by factor of 2
            #ds_slice = F.average_pooling(ds, kernel=(1, 1,), stride=(1, 2,), pad=(0, 0,))
            return ds, ds_slice

    def upsampling_block(x, i):
        with nn.parameter_scope( ('us_block-%2d' % i) ):
            up = F.unpooling(af(x), (2,))
            cac_x = crop_and_concat(ds_outputs[-i-1], up)
            us = af(conv(cac_x, num_initial_filters+num_initial_filters*(num_layers-i-1), (merge_filter_size,), (2,), name='conv'))
            return us

    ## --------------------------------- ##
    ##     Network Model                 ##
    ## --------------------------------- ##
    with nn.parameter_scope('Wave-U-Net'):

        ##  Input
        current_layer = Noisy

        ##  Downsampling block
        for i in range(num_layers):
            ds, current_layer = downsampling_block(current_layer, i) 
            ds_outputs.append(ds)

        ##  Latent variable (input: (N,288,4))
        with nn.parameter_scope('latent_variable'):
            current_layer = af(conv(current_layer, num_initial_filters+num_initial_filters*num_layers))

        ##  Upsampling block
        for i in range(num_layers):
            current_layer = upsampling_block(current_layer, i)
            if i==3:
                lv_mask = current_layer

        current_layer = crop_and_concat(Noisy, current_layer)

        ##  Output layer
        target_1 = F.tanh(conv(current_layer, 1, (1,), (0,), name='target_1'))
        target_2 = F.tanh(conv(current_layer, 1, (1,), (0,), name='target_2'))

        ##  Output
        if latent:
            return target_1, target_2, lv_mask
        else:
            return target_1, target_2


### ================================================================ ###
###                                                                  ###
###   Feature Mapping of Latent of Generator                         ###
###                                                                  ###
### ================================================================ ###
def Mapping(Input, alpha=0.5, test=False):
    with nn.parameter_scope('map'):
        lv_cnv1 = af(conv(Input, 64, (1,), (0,), name='mask_cnv1'))         # Point-wise Convolution (for Channel Only)
        lv_cnv2 = F.sigmoid(conv(lv_cnv1, 8, (1,), (0,), name='mask_cnv2'))    # Point-wise Convolution (for Channel Only)
        lv_mask = lv_cnv2                                                   # Latent Variable
        lv_dcv1 = af(deconv(lv_cnv2, 64, (1,), (0,), name='mask_dcv1'))     # Point-wise Convolution (for Channel Only)
        lv_out  = af(deconv(lv_dcv1, 216, (1,), (0,), name='mask_dcv2'))    # Point-wise Convolution (for Channel Only)

    return lv_out, lv_mask  # Output & Latent


### ================================================================ ###
###                                                                  ###
###   Mask Window                                                    ###
###                                                                  ###
### ================================================================ ###
def Mask_Window(batch, L=16384, block=64, alpha=0.5, test=False):
    #   create window-mask
    width    = int(2*L/block)
    width_h  = int(L/block)
    w        = np.pad(np.hanning(width), [0,L-width], 'constant')
    w_multi  = np.concatenate([[np.roll(w, i * (width_h))] for i in range(int(block))])
    window  = nn.Variable.from_numpy_array(np.array([w_multi]).repeat(batch, axis=0))
    return np.array([w_multi]).repeat(batch, axis=0)


### ================================================================ ###
###                                                                  ###
###   Discriminator                                                  ###
###                                                                  ###
### ================================================================ ###
def Discriminator(Input, num_initial_filters = 24, num_layers = 12, merge_filter_size = 5, test=False):
    """
    Building discriminator network
        Noisy : (Batch, 1, 16384)
        Clean : (Batch, 1, 16384)
        Output : (Batch, 1, 16384)
    """

    ## --------------------------------- ##
    ##     Sub-functions                 ##
    ## --------------------------------- ##
    def downsampling_block(x, i):
        with nn.parameter_scope(('ds_block-%2d' % i)):
            ds = af(conv(x, (num_initial_filters + num_initial_filters * i), name='conv'))
            ds_slice = F.slice(ds, start=(0, 0, 0), stop=ds.shape, step=(1, 1, 2))  # Decimate by factor of 2
            # ds_slice = F.average_pooling(ds, kernel=(1, 1,), stride=(1, 2,), pad=(0, 0,))
            return ds, ds_slice

    ## --------------------------------- ##
    ##     Network Model                 ##
    ## --------------------------------- ##
    with nn.parameter_scope("dis"):
        current_layer = Input
        ## downsampling block
        for i in range(num_layers):
            _, current_layer = downsampling_block(current_layer, i)
        f       = PF.affine(current_layer, 1)                    # (1024, 16) --> (1,)

    return f


### ================================================================ ###
###                                                                  ###
###   Loss Functions                                                 ###
###                                                                  ###
### ================================================================ ###
def Loss_dis(dval_real, dval_fake, p=2):

    E_real = F.mean( F.squared_error(dval_real, F.constant(1, dval_real.shape)) )    # real
    E_fake = F.mean( F.squared_error(dval_fake, F.constant(0, dval_fake.shape)) )    # fake
    return E_real + E_fake

def Loss_gen(wave_fake1, wave_true1, wave_fake2, wave_true2, dval_fake, alpha = 0.1, lmd=400, no_fake=False):

    E_fake = F.mean( F.squared_error(dval_fake, F.constant(1, dval_fake.shape)) )	# fake
    E_wave1 =  F.mean(F.absolute_error(wave_fake1, wave_true1)) 	# Reconstruction Performance
    E_wave2 = F.mean(F.absolute_error(wave_fake2, wave_true2))      # Reconstruction Performance

    if no_fake:
        gain = 0
    else:
        gain = 1

    return gain * E_fake/lmd + (E_wave1 + E_wave2)/2

def Loss_rec(wave_fake1, wave_true1):
    return F.mean(F.absolute_error(wave_fake1, wave_true1))


### ================================================================ ###
###                                                                  ###
###   Training                                                       ###
###                                                                  ###
### ================================================================ ###
def train(args):

    ## --------------------------------- ##
    ##     Sub-functions                 ##
    ## --------------------------------- ##
    ##  Save Models
    def save_models(epoch_num, losses):

        # save generator parameter
        with nn.parameter_scope('Wave-U-Net'):
            nn.save_parameters(os.path.join(args.model_save_path, 'param_{:04}.h5'.format(epoch_num + 1)))

        # save results
        np.save(os.path.join(args.model_save_path, 'losses_{:04}.npy'.format(epoch_num + 1)), np.array(losses))

    ##  Load Models
    def load_models(epoch_num, gen=True, dis=True):

        # load generator parameter
        with nn.parameter_scope('Wave-U-Net'):
            nn.load_parameters(os.path.join(args.model_save_path, 'param_{:04}.h5'.format(args.epoch_from)))

    ##  Update parameters
    class updating:

        # solver setting
        def __init__(self, learning_rate, scope=""):
            self.solver = S.Adam(learning_rate)  # solver set

            with nn.parameter_scope(scope):
                self.solver.set_parameters(nn.get_parameters())  # parameter set

            self.scale = 8 if args.halfprec else 1  # 32bit or 16bit

        # forward
        def forward(self, loss):
            loss.forward(clear_no_need_grad=True)
            self.solver.zero_grad()  # initialize

        # backward
        def backward(self, val):
            val.backward(self.scale, clear_buffer=True)

        # update
        def update(self):
            self.solver.scale_grad(1. / self.scale)  # scaling
            self.solver.update()  # update


    ## --------------------------------- ##
    ##      Initialization               ##
    ## --------------------------------- ##
    ##  Clear
    nn.clear_parameters()

    ##  Variables
    noisy 		= nn.Variable([args.batch_size, 1, 16384], need_grad=False)  # Input
    clean 		= nn.Variable([args.batch_size, 1, 16384], need_grad=False)  # Desire

    ## --------------------------------- ##
    ##      Network Model                ##
    ## --------------------------------- ##
    ##  Generator
    target_1, target_2, latent = Wave_U_Net(noisy, latent=True)
    target_1ul  = target_1.get_unlinked_variable()  # Unlinked Parameter
    target_2ul  = target_2.get_unlinked_variable()  # Unlinked Parameter
    latent_ul   = latent.get_unlinked_variable()    # Unlinked Parameter

    ##  Mapping
    map_out, map_mask = Mapping(latent_ul)
    map_mask_ul = map_mask.get_unlinked_variable()  # Unlinked Parameter

    ##  Mask
    window = nn.Variable.from_numpy_array(Mask_Window(args.batch_size))
    mask = F.batch_matmul(map_mask_ul, window)

    ##  Discriminator
    Input_dis_real  = F.concatenate(noisy, mask, clean, axis=1)
    Input_dis_fake  = F.concatenate(noisy, mask, target_1ul, axis=1)
    real_dis        = Discriminator(Input_dis_real)
    fake_dis        = Discriminator(Input_dis_fake)
    fake_dis_ul     = fake_dis.get_unlinked_variable()

    ## --------------------------------- ##
    ##      Loss & Updator               ##
    ## --------------------------------- ##
    ##  Loss ( order of calculation )
    loss_map = Loss_rec(map_out, latent_ul)
    loss_dis = Loss_dis(real_dis, fake_dis)
    loss_gen = Loss_gen(target_1ul, clean, target_2ul, noisy - clean, fake_dis_ul, no_fake=True)
    loss_rec = Loss_rec(target_1ul, clean)      #   Reconstruction Loss

    ##  Optimizer: Adam
    upd_gen = updating(args.learning_rate, scope='Wave-U-Net')
    upd_dis = updating(args.learning_rate_dis, scope='dis')
    upd_map = updating(args.learning_rate_dis, scope='map')

    ## --------------------------------- ##
    ##      Data Loader                  ##
    ## --------------------------------- ##
    ##  Load data & Create batch
    clean_data, noisy_data = dt.data_loader()
    batches   = dt.create_batch(clean_data, noisy_data, args.batch_size)
    del clean_data, noisy_data

    ##  Initial settings for sub-functions
    fig     = figout()
    disp    = display(args.epoch_from, args.epoch, batches.batch_num)

    ## --------------------------------- ##
    ##     Training Process              ##
    ## --------------------------------- ##
    print('== Start Training ==')

    ##  Load "Pre-trained" parameters
    if args.epoch_from > 0:
        print(' Retrain parameter from pre-trained network')
        load_models(args.epoch_from)
        losses  = np.load(os.path.join(args.model_save_path, 'losses_{:04}.npy'.format(args.epoch_from)))
        
        ## Create loss loggers
        point        = args.epoch_from * ((batches.batch_num+1)//10)
        loss_len     = (args.epoch - args.epoch_from) * ((batches.batch_num+1)//10)
        losses   = np.append(losses, np.zeros(loss_len))
    else:
        losses   = []
        ## Create loss loggers
        point        = len(losses)
        loss_len     = (args.epoch - args.epoch_from) * ((batches.batch_num+1)//10)
        losses   = np.append(losses, np.zeros(loss_len))

    ##  Training
    for i in range(args.epoch_from, args.epoch):

        print('')
        print(' =========================================================')
        print('  Epoch :: {0}/{1}'.format(i + 1, args.epoch))
        print(' =========================================================')
        print('')

        #  Batch iteration
        for j in range(batches.batch_num):
            print('  Train (Epoch. {0}) - {1}/{2}'.format(i+1, j+2, batches.batch_num))

            ##  Batch setting
            clean.d, noisy.d = batches.next(j)

            ##  Updating
            target_1.forward() # "latent" is automatically executed
            target_2.forward()

            #upd_map.forward(loss_map)   # "map_out" is automatically executed
            #upd_map.backward(loss_map)
            #upd_map.update()

            #upd_dis.forward(loss_dis)
            #upd_dis.backward(loss_dis)
            #upd_dis.update()

            upd_gen.forward(loss_gen)
            #upd_gen.backward(loss_gen)
            target_1.backward(grad=None)
            target_2.backward(grad=None)
            upd_gen.update()

            ##  Display
            if (j) % 100 == 0:

                # Display text
                loss_rec.forward(clear_no_need_grad=True)
                disp(i, j, loss_rec.d)

                # Data logger
                losses[point] = loss_gen.d
                point = point + 1

                # Plot
                fig.waveform_1(noisy.d[0,0,:], target_1.d[0,0,:], clean.d[0,0,:])
                fig.waveform_2(noisy.d[0,0,:], target_2.d[0,0,:], clean.d[0,0,:])
                fig.loss(losses[0:point-1])
                pg.QtGui.QApplication.processEvents()

        ## Save parameters
        if ((i+1) % args.model_save_cycle) == 0:
            save_models(i, losses)  # save model
            # fig.save(os.path.join(args.model_save_path, 'plot_{:04}.pdf'.format(i + 1))) # save fig
            exporter = pg.exporters.ImageExporter(fig.win.scene())  # exportersの直前に pg.QtGui.QApplication.processEvents() を呼ぶ！
            exporter.export(os.path.join(args.model_save_path, 'plot_{:04}.png'.format(i + 1))) # save fig

    ## Save parameters (Last)
    save_models(args.epoch-1, losses)
    exporter = pg.exporters.ImageExporter(fig.win.scene())  # exportersの直前に pg.QtGui.QApplication.processEvents() を呼ぶ！
    exporter.export(os.path.join(args.model_save_path, 'plot_{:04}.png'.format(i + 1))) # save fig


### ================================================================ ###
###                                                                  ###
###   Test                                                           ###
###                                                                  ###
### ================================================================ ###
def test(args):

    ##  Load data & Create batch
    clean_data, noisy_data, length_data = dt.data_loader(test=True, need_length=True)
    print(clean_data.shape)
    # Batch
    #  - Proccessing speech interval can be adjusted by "start_frame" and "start_frame".
    #  - "None" -> All speech in test dataset.
    
    output_ts = []
    bt_idx    = 0
    test_batch_size = args.batch_size
    for i in range(clean_data.shape[0]//(test_batch_size*2)):
        print(i, "/", clean_data.shape[0]//(test_batch_size*2))
        batches_test = dt.create_batch_test(clean_data[bt_idx:bt_idx+test_batch_size*2], noisy_data[bt_idx:bt_idx+test_batch_size*2], start_frame=0, stop_frame=test_batch_size*2)

        ##  Create network
        # Variables
        noisy_t     = nn.Variable(batches_test.noisy.shape)          # Input
        # Network (Only Generator)
        output_t, _ = Wave_U_Net(noisy_t)
        ##  Load parameter
        # load generator
        with nn.parameter_scope('Wave-U-Net'):
            nn.load_parameters(os.path.join(args.model_save_path, "param_{:04}.h5".format(args.epoch)))
        ##  Validation
        noisy_t.d = batches_test.noisy
        output_t.forward()

        ##  Create wav files
        output = output_t.d.flatten()
        output_ts.append(output)
        bt_idx += (test_batch_size*2)
    if (clean_data.shape[0]%(test_batch_size*2)) != 0:
        last_batch_size_2 = clean_data.shape[0]%(test_batch_size*2)
        print(last_batch_size_2)
        batches_test = dt.create_batch_test(clean_data[bt_idx:bt_idx+last_batch_size_2], noisy_data[bt_idx:bt_idx+last_batch_size_2], start_frame=0, stop_frame=last_batch_size_2)

        ##  Create network
        # Variables
        noisy_t     = nn.Variable(batches_test.noisy.shape)          # Input
        # Network (Only Generator)
        ##  Load parameter
        output_t, _ = Wave_U_Net(noisy_t)
        # load generator
        with nn.parameter_scope('Wave-U-Net'):
            nn.load_parameters(os.path.join(args.model_save_path, "param_{:04}.h5".format(args.epoch)))
        ##  Validation
        noisy_t.d = batches_test.noisy
        output_t.forward()

        ##  Create wav files
        output = output_t.d.flatten()
        output_ts.append(output)
        bt_idx += (last_batch_size_2)

    output = output_ts[0]
    for i in range(1, len(output_ts)):
        output = np.concatenate([output, output_ts[i]], axis=0)
    print(len(output))
    output = np.array(output)
    print(output.shape)
    idx_cnt = 0
    for i in range(len(length_data['name'])):
        print("saving", i, length_data['name'][i], "...")
        outwav = output[idx_cnt:idx_cnt+length_data['length'][i]]
        print(outwav.shape)
        idx_cnt += length_data['length'][i]
        print(idx_cnt)
        dt.wav_write((args.wav_save_path + '/' + 'ests_' + os.path.basename(length_data['name'][i])), np.array(outwav), fs=16000)
    print('finish!')


### ================================================================ ###
###                                                                  ###
###   Main                                                           ###
###                                                                  ###
### ================================================================ ###
if __name__ == '__main__':

    ## Load settings
    args = settings()

    ## GPU connection
    if args.halfprec:
        # - Float 16-bit precision mode : When GPU memory often gets stack, please use it.
        ctx = get_extension_context('cudnn', device_id=args.device_id, type_config='half')
    else:
        # - Float 32-bit precision mode :
        ctx = get_extension_context('cudnn', device_id=args.device_id)

    ## Training or Prediction
    Train = 1
    if Train:
        # Training
        nn.set_default_context(ctx)
        train(args)
    else:
        # Test
        # nn.set_default_context(ctx)
        # test(args)

        import glob
        clean_wavs = glob.glob(args.clean_test_path+'/*.wav')
        reconst_wavs = glob.glob(args.wav_save_path+'/*.wav')
        pesq_score(clean_wavs, reconst_wavs)
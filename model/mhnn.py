import os
import torch
import torch.utils.model_zoo
import torchvision.models as models
from src.model.snn import *
from src.model.cann import CANN
from src.tools.utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class MHNN(nn.Module):
    def __init__(self, **kwargs):
        super(MHNN, self).__init__()

        self.cnn_arch = kwargs.get('cnn_arch')
        self.num_class = kwargs.get('num_class')
        self.cann_num = kwargs.get('cann_num')
        self.rnn_num = kwargs.get('rnn_num')
        self.lr = kwargs.get('lr')
        self.batch_size = kwargs.get('batch_size')
        self.sparse_lambdas = kwargs.get('sparse_lambdas')
        self.r = kwargs.get('r')

        self.reservoir_num = kwargs.get('reservoir_num')
        self.threshold = kwargs.get('spiking_threshold')

        self.num_epoch = kwargs.get('num_epoch')
        self.num_iter = kwargs.get('num_iter')
        self.w_fps = kwargs.get('w_fps')
        self.w_gps = kwargs.get('w_gps')
        self.w_dvs = kwargs.get('w_dvs')
        self.w_head = kwargs.get('w_head')
        self.w_time = kwargs.get('w_time')

        self.seq_len_aps = kwargs.get('seq_len_aps')
        self.seq_len_gps = kwargs.get('seq_len_gps')
        self.seq_len_dvs = kwargs.get('seq_len_dvs')
        self.seq_len_head = kwargs.get('seq_len_head')
        self.seq_len_time = kwargs.get('seq_len_time')
        self.dvs_expand = kwargs.get('dvs_expand')

        self.ann_pre_load = kwargs.get('ann_pre_load')
        self.snn_pre_load = kwargs.get('snn_pre_load')
        self.re_trained = kwargs.get('re_trained')

        self.train_exp_idx = kwargs.get('train_exp_idx')
        self.test_exp_idx = kwargs.get('test_exp_idx')

        self.data_path = kwargs.get('data_path')
        self.snn_path = kwargs.get('snn_path')

        #self.device = kwargs.get('device')
        self.device = device

        if self.ann_pre_load:
            print("=> Loading pre-trained model '{}'".format(self.cnn_arch))
            self.cnn = models.__dict__[self.cnn_arch](pretrained=self.ann_pre_load)
        else:
            print("=> Using randomly inizialized model '{}'".format(self.cnn_arch))
            self.cnn = models.__dict__[self.cnn_arch](pretrained=self.ann_pre_load)

        if self.cnn_arch == "mobilenet_v2":
            """ MobileNet """
            self.feature_dim = self.cnn.classifier[1].in_features
            self.cnn.classifier[1] = nn.Identity()

        elif self.cnn_arch == "resnet50":
            """ Resnet50 """

            self.feature_dim = 512

            # self.cnn.layer1 = nn.Identity()
            self.cnn.layer2 = nn.Identity()
            self.cnn.layer3 = nn.Identity()
            self.cnn.layer4 = nn.Identity()
            self.cnn.fc = nn.Identity()

            self.cnn.layer1[1] = nn.Identity()
            self.cnn.layer1[2] = nn.Identity()

            # self.cnn.layer2[0] = nn.Identity()
            # self.cnn.layer2[0].conv2 = nn.Identity()
            # self.cnn.layer2[0].bn2 = nn.Identity()

            fc_inputs = 256
            self.cnn.fc = nn.Linear(fc_inputs,self.feature_dim)

        else:
            print("=> Please check model name or configure architecture for feature extraction only, exiting...")
            exit()

        for param in self.cnn.parameters():
            param.requires_grad = self.re_trained

        #############
        # SNN module
        #############
        self.snn = SNN(device = self.device).to(self.device)
        self.snn_out_dim = self.snn.fc2.weight.size()[1]
        self.ann_out_dim = self.feature_dim
        self.cann_out_dim = 4 * self.cann_num
        self.reservior_inp_num = self.ann_out_dim + self.snn_out_dim + self.cann_out_dim
        self.LN = nn.LayerNorm(self.reservior_inp_num)
        if self.snn_pre_load:
            self.snn.load_state_dict(torch.load(self.snn_path)['snn'])

        #############
        # CANN module
        #############
        self.cann_num = self.cann_num
        self.cann = None
        self.num_class = self.num_class

        #############
        # MLSM module
        #############
        self.input_size = self.feature_dim
        self.reservoir_num = self.reservoir_num

        self.threshold = 0.5

        self.decay = nn.Parameter(torch.rand(self.reservoir_num))

        self.K = 128
        self.num_block = 5
        self.num_blockneuron = int(self.reservoir_num / self.num_block)

        self.decay_scale = 0.5
        self.beta_scale = 0.1

        self.thr_base1 = self.threshold

        self.thr_beta1 = nn.Parameter(self.beta_scale * torch.rand(self.reservoir_num))

        self.thr_decay1 = nn.Parameter(self.decay_scale * torch.rand(self.reservoir_num))

        self.ref_base1 = self.threshold
        self.ref_beta1 = nn.Parameter(self.beta_scale * torch.rand(self.reservoir_num))
        self.ref_decay1 = nn.Parameter(self.decay_scale * torch.rand(self.reservoir_num))

        self.cur_base1 = 0
        self.cur_beta1 = nn.Parameter(self.beta_scale * torch.rand(self.reservoir_num))
        self.cur_decay1 = nn.Parameter(self.decay_scale * torch.rand(self.reservoir_num))

        self.project = nn.Linear(self.reservior_inp_num, self.reservoir_num)

        self.project_mask_matrix = torch.zeros((self.reservior_inp_num, self.reservoir_num))


        input_node_list = [0, self.ann_out_dim, self.snn_out_dim, self.cann_num * 2, self.cann_num]

        input_cum_list = np.cumsum(input_node_list)

        for i in range(len(input_cum_list) - 1):
            self.project_mask_matrix[input_cum_list[i]:input_cum_list[i + 1],
            self.num_blockneuron * i:self.num_blockneuron * (i + 1)] = 1

        self.project.weight.data = self.project.weight.data * self.project_mask_matrix.t()

        self.lateral_conn = nn.Linear(self.reservoir_num, self.reservoir_num)

        self.lateral_conn_mask = torch.rand(self.reservoir_num, self.reservoir_num) > 0.8

        self.lateral_conn_mask = self.lateral_conn_mask * (1 - torch.eye(self.reservoir_num, self.reservoir_num))

        self.lateral_conn.weight.data = 0 * self.lateral_conn.weight.data * self.lateral_conn_mask.T

        #############
        # readout module
        #############

        # self.mlp1 = nn.Linear(self.reservoir_num, 256)
        # self.mlp2 = nn.Linear(256, self.num_class)
        self.mlp =  nn.Linear(self.reservoir_num, self.num_class)

    def cann_init(self, data):
        self.cann = CANN(data)

    def lr_initial_schedule(self, lrs=1e-3):
        hyper_param_list = ['decay',
                            'thr_beta1', 'thr_decay1',
                            'ref_beta1', 'ref_decay1',
                            'cur_beta1', 'cur_decay1']
        hyper_params = list(filter(lambda x: x[0] in hyper_param_list, self.named_parameters()))
        base_params = list(filter(lambda x: x[0] not in hyper_param_list, self.named_parameters()))
        hyper_params = [x[1] for x in hyper_params]
        base_params = [x[1] for x in base_params]
        optimizer = torch.optim.SGD(
            [
                {'params': base_params, 'lr': lrs},
                {'params': hyper_params, 'lr': lrs / 2},
            ], lr=lrs, momentum=0.9, weight_decay=1e-7
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(self.num_epoch))
        return optimizer, scheduler

    def wta_mem_update(self, fc, fv, k, inputx, spike, mem, thr, ref, last_cur):
        state = fc(inputx)
        mem = (mem - spike * ref) * self.decay + state + last_cur
        q0 = (100 -20)/100
        mem = mem.reshape(self.batch_size, self.num_blockneuron, -1)
        nps = torch.quantile(mem, q=q0, keepdim=True, axis=1)
        # nps =mem.max(axis=1,keepdim=True)[0] - thr

        mem = F.relu(mem - nps).reshape(self.batch_size, -1)

        spike = act_fun(mem - thr)
        return spike.float(), mem

    def trace_update(self, trace, spike, decay):
        return trace * decay + spike

    def forward(self, inp, epoch=100):
        aps_inp = inp[0].to(self.device)
        gps_inp = inp[1]
        dvs_inp = inp[2].to(self.device)
        head_inp = inp[3].to(self.device)

        batch_size, seq_len, channel, w, h = aps_inp.size()
        if self.w_fps > 0:
            aps_inp = aps_inp.view(batch_size * seq_len, channel, w, h)
            aps_out = self.cnn(aps_inp)
            out1 = aps_out.reshape(batch_size, self.seq_len_aps, -1).permute([1, 0, 2])
        else:
            out1 = torch.zeros(self.seq_len_aps, batch_size, -1, device=self.device).to(torch.float32)

        if self.w_dvs > 0:
            out2 = self.snn(dvs_inp, out_mode='time')
        else:
            out2 = torch.zeros(self.seq_len_dvs * 3, batch_size, self.snn_out_dim, device=self.device).to(torch.float32)

        ### CANN module
        if self.w_gps + self.w_head + self.w_time > 0:
            gps_record = []
            for idx in range(batch_size):
                buf = self.cann.update(torch.cat((gps_inp[idx],head_inp[idx].cpu()),axis=1), trajactory_mode=True)
                gps_record.append(buf[None, :, :, :])
            gps_out = torch.from_numpy(np.concatenate(gps_record)).cuda()
            gps_out = gps_out.permute([1, 0, 2, 3]).reshape(self.seq_len_gps, batch_size, -1)
        else:
            gps_out = torch.zeros((self.seq_len_gps, batch_size, self.cann_out_dim), device=self.device)

        # A generic CANN module was used for rapid testing; CANN1D/2D are provided in cann.py
        out3 = gps_out[:, :, self.cann_num:self.cann_num * 3].to(self.device).to(torch.float32)  # position
        out4 = gps_out[:, :, : self.cann_num].to(self.device).to(torch.float32)  # time
        out5 = gps_out[:, :, - self.cann_num:].to(self.device).to(torch.float32)  # direction

        out3 *= self.w_gps
        out4 *= self.w_time
        out5 *= self.w_head

        expand_len = int(self.seq_len_gps * self.seq_len_aps / np.gcd(self.seq_len_gps, self.seq_len_dvs) * self.dvs_expand)
        input_num = self.feature_dim + self.snn.fc2.weight.size()[1] + self.cann_num * self.dvs_expand

        r_spike = r_mem = r_sumspike = torch.zeros(batch_size, self.reservoir_num, device=self.device)

        thr_trace = torch.zeros(batch_size, self.reservoir_num, device=self.device)
        ref_trace = torch.zeros(batch_size, self.reservoir_num, device=self.device)
        cur_trace = torch.zeros(batch_size, self.reservoir_num, device=self.device)

        K_winner = self.K * (1 + np.clip(self.num_epoch - epoch, a_min=0, a_max=self.num_epoch) / self.num_epoch)

        out1_zeros = torch.zeros_like(out1[0], device=self.device)
        out3_zeros = torch.zeros_like(out3[0], device=self.device)
        out4_zeros = torch.zeros_like(out4[0], device=self.device)
        out5_zeros = torch.zeros_like(out5[0], device=self.device)

        self.project.weight.data = self.project.weight.data * self.project_mask_matrix.t().cuda()
        self.lateral_conn.weight.data = self.lateral_conn.weight.data * self.lateral_conn_mask.T.cuda()

        self.decay.data = torch.clamp(self.decay.data, min=0, max=1.)
        self.thr_decay1.data = torch.clamp(self.thr_decay1.data, min=0, max=1.)
        self.ref_decay1.data = torch.clamp(self.ref_decay1.data, min=0, max=1)
        self.cur_decay1.data = torch.clamp(self.cur_decay1.data, min=0, max=1)

        for step in range(expand_len):

            idx = step % 3
            if idx == 2:
                combined_input = torch.cat((out1[step // 3], out2[step], out3[step // 3], out4[step // 3], out5[step // 3]), axis=1)
            else:
                combined_input = torch.cat((out1_zeros, out2[step],out3_zeros, out4_zeros, out5_zeros), axis=1)

            thr = self.thr_base1 + thr_trace * self.thr_beta1
            # ref = self.ref_base1 + ref_trace * self.ref_beta1 # option: ref = self.ref_base1
            ref = self.ref_base1
            cur = self.cur_base1 + cur_trace * self.cur_beta1

            inputx = combined_input.float()
            r_spike, r_mem = self.wta_mem_update(self.project, self.lateral_conn, K_winner, inputx, r_spike, r_mem,
                                                 thr, ref, cur)
            thr_trace = self.trace_update(thr_trace, r_spike, self.thr_decay1)
            ref_trace = self.trace_update(ref_trace, r_spike, self.ref_decay1)
            cur_trace = self.trace_update(cur_trace, r_spike, self.cur_decay1)
            r_sumspike = r_sumspike + r_spike

        # cat_out = F.dropout(r_sumspike, p=0.5, training=self.training)
        # out1 = self.mlp1(r_sumspike).relu()
        # out2 = self.mlp2(out1)
        out2 = self.mlp(r_sumspike)

        neuron_pop = r_sumspike.reshape(batch_size, -1, self.num_blockneuron).permute([1, 0, 2])
        return out2, (neuron_pop[0], neuron_pop[1])













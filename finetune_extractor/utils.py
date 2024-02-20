import itertools
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
import torchvision

from torch.utils.data.sampler import BatchSampler

class AngProtoLoss4(torch.nn.Module):

    def __init__(self, config, device, to_normalize=True, refine_matrix=False, p_pct=85, g_blur=1, init_w=20.0, init_b=-15.0,
                 mse_fac=0, **kwargs):
        super(AngProtoLoss4, self).__init__()

        self.to_normalize = to_normalize
        
        self.w = torch.nn.Parameter(torch.tensor(init_w))
        self.b = torch.nn.Parameter(torch.tensor(init_b))
        self.margin = getattr(config, 'margin', 0)
        self.criterion  = torch.nn.CrossEntropyLoss()
        self.device = device
        self.dropout = torch.nn.Dropout(0.2)
        self.lsm = torch.nn.LogSoftmax(dim=1)
        self.mse_fac = mse_fac
        self.config = config
        embsize = 768 if not config["custom_embed_size"] else config["custom_embed_size"]

        self.refine_matrix = refine_matrix
        self.p_pct = p_pct
        self.g_blur = g_blur
        if self.g_blur > 0:
            self.kernel_size = int(4 * self.g_blur + 0.5) * 2 + 1
            print("kernel size", self.kernel_size)
            self.gk = torchvision.transforms.GaussianBlur(self.kernel_size, sigma=(self.g_blur, self.g_blur))
        
        print('Initialised AngleProto')
    
    def mod_cross_entropy(self, predictions, targets, epsilon=1e-12, threshold=None):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions. 
        Input: predictions (N, k) ndarray
            targets (N, k) ndarray        
        Returns: scalar
        """
        device = predictions.device
        targets = targets.to(device)
        predictions = predictions * self.w + self.b
        # only hard ones are trained
        if self.p_pct > 0 and self.refine_matrix:
            # use rel threshold value from g blur
            if threshold is not None:
                threshold = threshold * self.w + self.b
                #  ~ hard negative examples
                lower_mask = (predictions >= threshold.unsqueeze(1).repeat(1, predictions.size(0))) * (1 - targets)
                upper_mask = targets
            # use absolute value of p_pct as the threshold
            else:
                ppct = self.p_pct / 100 * self.w + self.b
                lower_mask = (predictions >= ppct) * (1 - targets)
                upper_mask = targets
            ppct_mask = 1 - (lower_mask + upper_mask)
            predictions = torch.log(torch.softmax(predictions.masked_fill(ppct_mask.bool(), -1e9), dim=-1) + 1e-9)
            targets = targets * (1 - ppct_mask)
        # no thresholding at all
        # which means all hard and easy are trained
        else:
            predictions = self.lsm(predictions)
        N = torch.sum(targets)
        ce = -torch.sum(targets*predictions)/N
        return ce
        
    def mse_loss(self, predictions, targets, threshold=None):
        """
        Computes mse entropy between targets (encoded as one-hot vectors)
        and predictions. 
        Input: predictions (N, k) ndarray
            targets (N, k) ndarray        
        Returns: scalar
        """
        device = predictions.device
        targets = targets.to(device)
        diff = predictions - targets
        # if we are thresholding
        # meaning we only force train positive pair
        if self.p_pct > 0:
            # if we did blurring previously
            if threshold is not None:
                # mower mask: ~ hard negative examples
                # upper mask: ~ hard positive examples
                lower_mask = (predictions >= (threshold*(1-self.margin)).unsqueeze(1).repeat(1, predictions.size(0))) * (1 - targets)
                upper_mask = (predictions <= (threshold*(1+self.margin)).unsqueeze(1).repeat(1, predictions.size(0))) * targets
            # no blurring previously
            else:
                # lower mask - negative pairs where sim score is high
                # upper mask - positive pairs where sim score is low
                upper_p_pct = 1.0 - self.p_pct / 100
                lower_mask = (predictions >= (self.p_pct / 100)) * (1 - targets)
                upper_mask = (predictions <= (self.p_pct / 100)) * targets
            # final mask = hard positive or hard negative examples
            mask = lower_mask + upper_mask
            mse = ((diff ** 2) * mask).sum() / mask.sum()
        # no thresholding at all
        else:
            mask = targets + (1 - targets)
            mse = ((mask*diff) ** 2).mean()
        return mse

    def forward(self, x, label=None):

        # x = x.to(self.device)
        label = label.to(self.device)
        # x = torch.from_numpy(x).float()
        x = x.view(x.size(0), -1)
        assert x.size(1) >= 2
        x = x.unsqueeze(1)
        

        out_anchor      = torch.mean(x,1)
        out_positive    = x[:,0,:]
 
        if self.to_normalize:
            out_anchor      = F.normalize(out_anchor)
            out_positive    = F.normalize(out_positive)
        stepsize        = out_anchor.size()[0]
    
        # cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
        # all diagonal elements is cos_sim of positive & anchor
        # non diag elements is cos_sim of positive & other positives
        cos_sim_diag = torch.diag(F.cosine_similarity(out_positive, out_anchor))
        cos_sim_matrix = F.cosine_similarity(out_positive.unsqueeze(-1), out_positive.unsqueeze(-1).transpose(0,2))
        cos_sim_matrix = cos_sim_matrix - torch.eye(out_positive.size(0)).to(out_positive.device) + cos_sim_diag
        # cos_sim_matrix = F.cosine_similarity(out_positive.unsqueeze(-1), out_positive.unsqueeze(-1).transpose(0,2))

        if label is not None:
            targets = torch.eye(len(label))
            distinct_labels = torch.unique(label)
            for l_i in distinct_labels:
                # get indices of embeddings from the same speaker
                ind_l = [ind for ind, el in enumerate(label) if el==l_i]
                # get the index of pairs of embeddings in the same cluster
                ind_2d = list(itertools.permutations(ind_l, 2))
                for ind in ind_2d:
                    targets[ind] = 1
            targets = targets.to(self.device)
        else:
            targets = torch.eye(cos_sim_matrix.size(0))
            targets = targets.to(self.device)

        cos_sim_matrix = (1 + cos_sim_matrix) / 2
        torch.clamp(self.w, 1e-6)

        threshold = None
        if self.g_blur > 0 and cos_sim_matrix.size(0) > self.kernel_size // 2:
            # make all diagonal elements 1
            # blur it
            # thres = diag * p_pct
            # get threshold after blurring
            cos_sim_matrix_blurred = cos_sim_matrix * (1 - targets) + targets
            cos_sim_matrix_blurred = cos_sim_matrix_blurred.view((1, 1, *cos_sim_matrix_blurred.size()))
            cos_sim_matrix_blurred = self.gk(cos_sim_matrix_blurred)
            cos_sim_matrix_blurred = cos_sim_matrix_blurred.squeeze()
            threshold = self.p_pct / 100 * torch.diagonal(cos_sim_matrix_blurred)

        if label is not None or targets is not None:
            # shift and scale affinity matrix
            nloss = self.mod_cross_entropy(cos_sim_matrix, targets, threshold=threshold)
            if self.mse_fac > 0:
                mseloss = self.mse_loss(cos_sim_matrix, targets, threshold=threshold)
                nloss = (1 - self.mse_fac) * nloss + self.mse_fac * mseloss

        else:
            label   = torch.from_numpy(np.asarray(range(0,stepsize))).to(self.device)
            nloss   = self.criterion(cos_sim_matrix, label)
        
       
        return nloss


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=3, min_delta=0, best_loss=None):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):

        if self.best_loss == None:
            self.best_loss = val_loss

        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0

        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, data_df, n_classes, n_samples, label_column_name='label'):

        self.labels_list = list(data_df[label_column_name])
        self.labels = torch.LongTensor(self.labels_list)
        # get distinct labels
        self.labels_set = list(set(self.labels.numpy()))
        # dict where key is the speaker, value is the index of utterance
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        # shuffle the order of utterances for each speaker
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        # keep track of what has been used in each class
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_spk_samples = n_samples
        self.batch_size = self.n_samples * self.n_classes
        self.data_df = data_df

    
    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.labels_list):
            # sample n classes
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            # for each class we get n samples
            for class_ in classes:
                curr_class_start_ind = self.used_label_indices_count[class_]
                curr_class_end_ind = self.used_label_indices_count[class_] + self.n_samples
                indices.extend(self.label_to_indices[class_][curr_class_start_ind:curr_class_end_ind])
                self.used_label_indices_count[class_] += self.n_samples
                # we update the dict that keeps track of how many items have been used from each class
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples
    
    '''       
    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.labels_list):
            # sample n classes
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            # for each class we get n samples
            for class_ in classes:
                curr_class_start_ind = self.used_label_indices_count[class_]
                curr_class_end_ind = self.used_label_indices_count[class_] + self.n_spk_samples
                indices.extend(self.label_to_indices[class_][curr_class_start_ind:curr_class_end_ind])
                self.used_label_indices_count[class_] += self.n_spk_samples
                # we update the dict that keeps track of how many items have been used from each class
                if self.used_label_indices_count[class_] + self.n_samples -3 > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples
    ''' 
    def __len__(self):
        return len(self.labels_list) // self.batch_size

'''This code is based on the CFlow-AD project (source: https://github.com/gudovskiy/cflow-ad/tree/master).
We modified and added the necessary modules or functions for our purposes.'''
import torch

_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))

# monitoring validation results
class Score_Observer:
    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = 0.0
        self.last = 0.0

    def update(self, score, epoch, txt_file, print_score=True):
        self.last = score
        save_weights = False
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
            save_weights = True
        if print_score:
            self.print_score(txt_file)
        
        return save_weights

    def print_score(self, txt_file):
        print('{:s}: \t last: {:.2f} \t max: {:.2f} \t epoch_max: {:d}'.format(
            self.name, self.last, self.max_score, self.max_epoch))
        txt_file.write('\n{:s}: \t last: {:.2f} \t max: {:.2f} \t epoch_max: {:d}'.format(
            self.name, self.last, self.max_score, self.max_epoch))


# calculate log-likelihood for input feature maps 
def get_logp_2d(C, z_2d, logdet_J_2d, m_2d, is_train=True):
    m_2d = torch.squeeze(m_2d, dim=1)
    logp_2d = C * _GCONST_ - 0.5*torch.sum(z_2d**2, 1) + logdet_J_2d # shape:(B,H,W)
    defect_logp_2d = logp_2d.clone() 
    defect_logp_2d[m_2d==1]=0 # deactivate good pixels
    good_logp_2d = logp_2d.clone() 
    good_logp_2d[m_2d==-1] =0 # deactivate defect pixels
    if is_train ==True:
        logp_2d = torch.multiply(logp_2d, m_2d)
    else:
        pass
    return logp_2d,  defect_logp_2d, good_logp_2d

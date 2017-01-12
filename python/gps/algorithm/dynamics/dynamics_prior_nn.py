import caffe
import numpy as np

from gps.utility import fix_logging as logging
LOGGER = logging.getLogger(__name__)


class DynamicsPriorNN(object):
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        if self._hyperparams['use_gpu']:
            caffe.set_device(self._hyperparams['gpu_id'])
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.solver = caffe.get_solver(self._hyperparams['solver'])
        self.X = None
        self.caffe_iter = 0

    def initial_state(self):
        mu0 = np.mean(self.X[:,0,:], axis=0)
        Phi = np.diag(np.var(self.X[:,0,:], axis=0))
        n0 = self.X.shape[2] * self._hyperparams['strength']
        m = self.X.shape[2] * self._hyperparams['strength']
        Phi = Phi * m
        return mu0, Phi, m, n0

    def update(self, X, U):
        if self.X is None:
            self.X = X
        else:
            self.X = np.concatenate([self.X, X], axis=0)
        N, T = X.shape[0], X.shape[1] - 2
        dO = X.shape[2] * 2 + U.shape[2] * 2
        XUXU = np.reshape(np.c_[X[:,:-2,:], U[:,:-2,:], X[:,1:-1,:], U[:,1:-1,:]], [N*T, dO])
        Xtp1 = np.reshape(X[:,2:,:], [N*T, X.shape[2]])
        batches_per_epoch = np.floor(N*T / self._hyperparams['batch_size'])
        idx = range(N * T)
        average_loss = 0
        np.random.shuffle(idx)
        for itr in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            start_idx = int(itr*self._hyperparams['batch_size'] %
                    (batches_per_epoch*self._hyperparams['batch_size']))
            idx_i = idx[start_idx:start_idx+self._hyperparams['batch_size']]
            self.solver.net.blobs[self.solver.net.blobs.keys()[0]].data[:] = XUXU[idx_i,:]
            self.solver.net.blobs[self.solver.net.blobs.keys()[1]].data[:] = Xtp1[idx_i,:]

            self.solver.step(1)

            # To get the training loss:
            train_loss = self.solver.net.blobs[self.solver.net.blobs.keys()[-1]].data
            average_loss += train_loss
            if itr % 500 == 0 and itr != 0:
                LOGGER.debug('Prior iteration %d, average loss %f', itr, average_loss / 500)
                average_loss = 0
        self.caffe_iter += self._hyperparams['iterations']

    def eval(self, dX, dU, pts):
        assert pts.shape[1] == 3*dX + 2*dU, pts.shape
        XUXU = pts[:,:2*dX+2*dU]
        Xtp1 = pts[:,2*dX+2*dU:]
        F, f = self.getF(XUXU, dX)
        Phi, mu0 = self.mix_nn_prior(F, f, XUXU, self.get_sigma_x(XUXU, Xtp1), dX, dU)
        return mu0, Phi, 1, 1

    def mix_nn_prior(self, F, f, XUXU, sigma_x, dX, dU):
        sigX = np.eye(XUXU.shape[1]) * self._hyperparams['strength']
        sigXK = sigX.dot(F.T)
        Phi = np.r_[np.c_[sigX, sigXK], np.c_[sigXK.T, F.dot(sigX).dot(F.T) + sigma_x]]
        mu0 = np.mean(np.c_[XUXU, XUXU.dot(F.T)+f], axis=0)
        return Phi, mu0

    def getF(self, XUXU, dX):
        N, dO = XUXU.shape
        F, f = np.zeros((dX, dO)), np.zeros(dX)
        for n in range(N):
            xuxu = XUXU[n,:]
            self.solver.test_nets[0].blobs[self.solver.net.blobs.keys()[0]].data[:] = xuxu
            x = self.solver.test_nets[0].forward().values()[0][0]
            F += self.jac(x, dO, dX)
            f += -F.dot(xuxu) + x
        return F / N, f / N

    def get_sigma_x(self, XUXU, Xtp1):
        outputs = np.zeros_like(Xtp1)
        for n in range(XUXU.shape[0]):
            xuxu = XUXU[n,:]
            self.solver.test_nets[0].blobs[self.solver.net.blobs.keys()[0]].data[:] = xuxu
            outputs[n,:] = self.solver.test_nets[0].forward().values()[0][0]
        diff = outputs - Xtp1
        return np.cov(diff.T)

    def jac(self, x, dO, dX):
        diff = np.diag(x)
        self.solver.test_nets[1].blobs[self.solver.test_nets[1].blobs.keys()[-1]].diff[...] = diff
        self.solver.test_nets[1].backward()
        F = self.solver.test_nets[1].blobs[self.solver.test_nets[1].blobs.keys()[0]].diff
        return F

    # For pickling.
    def __getstate__(self):
        self.solver.snapshot()
        return {
            'hyperparams': self._hyperparams,
            'X': self.X,
            'caffe_iter': self.caffe_iter,
        }

    def __setstate__(self, state):
        self.__init__(state['hyperparams'])
        self.X = state['X']
        self.caffe_iter = state['caffe_iter']
        self.solver.restore(self._hyperparams['weights_file_prefix'] + '_iter_' +
                str(self.caffe_iter) + '.solverstate')

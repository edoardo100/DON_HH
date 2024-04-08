from timeit import default_timer
import torch
import matplotlib.pyplot as plt
from .utility_dataset import *
from .architectures import L2relLoss, MSE, H1relLoss
from .utility_dataset import load_test

class Training():
    def __init__(self, model, epochs, optimizer, schedulerName, scheduler, loss, 
                 dataset_test, ntrain, ntest, indices, train_loader, test_loader, 
                 x_train, v_test, x_test, scale_fac, scaling, idx, writer, ep_step, 
                 device='cpu', show_every=100, plotting=False):
        self.model         = model
        self.epochs        = epochs
        self.optimizer     = optimizer
        self.schedulerName = schedulerName
        self.scheduler     = scheduler
        self.loss          = loss
        self.dataset_test  = dataset_test
        self.ntrain        = ntrain
        self.ntest         = ntest
        self.indices       = indices # shuffled indices     
        self.train_loader  = train_loader
        self.test_loader   = test_loader
        self.x_train       = x_train
        self.v_test        = v_test
        self.x_test        = x_test
        self.scale_fac     = scale_fac
        self.scaling       = scaling
        self.idx           = idx    # indices of tests to plot
        self.writer        = writer # the TensorboardX writer
        self.device        = device
        self.show_every    = show_every
        self.plotting      = plotting
        self.ep_step       = ep_step

    def load_unscaled_data(self,dataset_test,indices):
            # Unscaled dataset (for plotting)
            if "LR" in self.dataset_test:
                u_test_unscaled, x_test_unscaled, v_test_unscaled, _ = load_LR_test(dataset_test,full_v_data=True)
                v_test_unscaled = v_test_unscaled[:,0,:]
            else:
                u_test_unscaled, x_test_unscaled, v_test_unscaled = load_test(dataset_test,full_v_data=True)
            # Same order of scaled data
            u_test_unscaled = u_test_unscaled[indices]
            v_test_unscaled = v_test_unscaled[indices]
            return u_test_unscaled, x_test_unscaled, v_test_unscaled

    def single_train_step(self,ep,t1):
        #### One epoch of training
            self.model.train()
            train_loss = 0
            for v, u in self.train_loader:
                v, u = v.to(self.device), u.to(self.device)
                self.optimizer.zero_grad() # annealing the gradient
                out = self.model.forward((v,self.x_train)) # compute the output
                # compute the loss
                loss = self.loss(out, u)
                train_loss += loss.item() # update the loss function
                loss.backward() # automatic back propagation
                self.optimizer.step()
                if self.schedulerName.lower() == "cosineannealinglr":
                    self.scheduler.step()
            if self.schedulerName.lower() == "steplr":
                self.scheduler.step()
            #### Evaluate the model on the test set
            self.model.eval()
            test_l2  = 0.0
            test_mse = 0.0
            test_h1  = 0.0
            with torch.no_grad():
                for v, u in self.test_loader:
                    v, u = v.to(self.device), u.to(self.device)
                    out = self.model.forward((v,self.x_test))
                    if self.loss.get_name() == "L2_rel":
                        test_l2 += self.loss(out, u).item()
                        test_mse += MSE()(out, u).item()
                        test_h1 += H1relLoss()(out, u).item()
                    elif self.loss.get_name() == "mse":
                        test_l2 += L2relLoss()(out, u).item()
                        test_mse += self.loss(out, u).item()   
                        test_h1 += H1relLoss()(out, u).item()
                    elif self.loss.get_name() == "H1_rel":
                        test_l2 += L2relLoss()(out, u).item()
                        test_mse += MSE()(out, u).item()
                        test_h1 += self.loss(out, u).item()

            if self.schedulerName.lower() == "reduceonplateau":
                self.scheduler.step(test_l2)

            train_loss/= self.ntrain
            test_l2/= self.ntest
            test_mse/= self.ntest
            test_h1/= self.ntest

            print(self.loss.get_name())

            t2 = default_timer()
            if ep % self.show_every == 0:
                print(f'Epoch:{ep}  Time:{t2 - t1:.{2}f}  '
                      f'Train_loss_{self.loss.get_name()}:{train_loss:.{5}f}  '
                      f'Test_loss_l2:{test_l2:.{5}f}  '
                      f'Test_mse:{test_mse:.{5}f}  '
                      f'Test_loss_h1:{test_h1:.{5}f}')

                self.writer.add_scalars('NO_HH', {'Train_loss': train_loss,
                                                        'Test_loss_l2': test_l2,
                                                        'Test_mse':    test_mse,
                                                        'Test_loss_h1': test_h1
                                                         }, ep)

    def plot_results(self,ep,u_test_unscaled,x_test_unscaled,v_test_unscaled):
        idx = self.idx
        #### initial value of v
        esempio_test    = v_test_unscaled[idx, :].to('cpu')
        esempio_test_pp = self.v_test[idx, :].to('cpu')
        sol_test        = u_test_unscaled[idx]
        x_test_unscaled = x_test_unscaled.to('cpu')
        if ep == 0:
            fig, ax = plt.subplots(1, len(idx), figsize = (18, 4))
            fig.suptitle('Applied current (I_app)')
            ax[0].set(ylabel = 'I_app(t)')
            for i in range(len(idx)):
                ax[i].plot(x_test_unscaled, esempio_test[i])
                ax[i].set(xlabel = 't')
                ax[i].set_ylim([-0.2, 10])
                ax[i].grid()
            if self.plotting:
                plt.show()
            self.writer.add_figure('Applied current (I_app)', fig, 0)
            #### Approximate classical solution
            fig, ax = plt.subplots(1, len(idx), figsize = (18, 4))
            fig.suptitle('Numerical approximation (V_m)')
            ax[0].set(ylabel = 'V_m (mV)')
            for i in range(len(idx)):
                ax[i].plot(x_test_unscaled, sol_test[i].to('cpu'))
                ax[i].set(xlabel = 't')
                ax[i].grid()
            if self.plotting:
                plt.show()
            self.writer.add_figure('Numerical approximation (V_m)', fig, 0)

        #### approximate solution with NO of HH model
        if ep % self.ep_step == 0:
            with torch.no_grad():  # no grad for effeciency reason
                out_test = self.model((esempio_test_pp.to(self.device),self.x_test.to(self.device)))
                out_test = out_test.to('cpu')
            if self.scaling == "Default":
                out_test = unscale_data(out_test.to(self.device),self.scale_fac[0],self.scale_fac[1])
            elif self.scaling == "Gaussian":
                out_test = inverse_gaussian_scale(out_test.to(self.device),self.scale_fac[0],self.scale_fac[1])
            elif self.scaling == "Mixed":
                out_test = inverse_gaussian_scale(out_test.to(self.device),self.scale_fac[0],self.scale_fac[1])
            fig, ax = plt.subplots(1, len(idx), figsize = (18, 4))
            fig.suptitle('NO approximation (V_m)')
            ax[0].set(ylabel = 'V_m (mV)')
            for i in range(len(idx)):
                ax[i].plot(x_test_unscaled, out_test[i].to('cpu'))
                ax[i].set(xlabel = 't')
                ax[i].grid()
            if self.plotting:
                plt.show()
            self.writer.add_figure('NO approximation (V_m)', fig, ep)
            #### Module of the difference between classical and NO approximation
            diff = torch.abs(out_test.to('cpu') - sol_test.to('cpu'))
            fig, ax = plt.subplots(1, len(idx), figsize = (18, 4))
            fig.suptitle('Module of the difference')
            ax[0].set(ylabel = '|V_m(mV)|')
            for i in range(len(idx)):
                ax[i].plot(x_test_unscaled, diff[i])                    
                ax[i].set(xlabel = 'x')
                ax[i].grid()
            if self.plotting:
                plt.show()
            self.writer.add_figure('Module of the difference', fig, ep)

    def train(self):    
        t1 = default_timer()
        u_test_unscaled, x_test_unscaled, v_test_unscaled = self.load_unscaled_data(self.dataset_test,self.indices)
        # Training process
        for ep in range(self.epochs+1):
            self.single_train_step(ep,t1)
            self.plot_results(ep,u_test_unscaled,x_test_unscaled,v_test_unscaled)

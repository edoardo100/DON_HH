from timeit import default_timer
import torch
import torch.profiler
from .utility_dataset import *
from .architectures import L2relLoss, L2relLossMultidim, MSE, H1relLoss

class Training():
    def __init__(self, model, epochs, optimizer, schedulerName, scheduler, loss, 
                 dataset_test, ntrain, ntest, indices, train_loader, test_loader, 
                 x_train, x_test, device='cpu', show_every=100):
        self.model         = model
        self.epochs        = epochs
        self.optimizer     = optimizer
        self.schedulerName = schedulerName
        self.scheduler     = scheduler
        self.loss          = loss
        self.dataset_test  = dataset_test
        self.ntrain        = ntrain
        self.ntest         = ntest
        self.indices       = indices      # shuffled indices     
        self.train_loader  = train_loader
        self.test_loader   = test_loader
        self.x_train       = x_train
        self.x_test        = x_test
        self.device        = device
        self.show_every    = show_every

    def single_train_step(self,ep,t1):
        #### One epoch of training
            self.model.train()
            train_loss = 0
            for v, u in self.train_loader:
                v, u = v.to(self.device), u.to(self.device)
                if len(u.shape) > 2:
                    u    = u.permute(2,0,1) # e.g. back to (4,1600,500) for train
                self.optimizer.zero_grad() # annealing the gradient
                out = self.model.forward((v,self.x_train)) # compute the output
                # compute the loss
                u.requires_grad = True
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
                    if len(u.shape) > 2:
                        u = u.permute(2,0,1)
                        out = self.model.forward((v, self.x_test))
                        test_l2 += L2relLossMultidim()(out, u).item()
                    if len(out.shape) == 2:
                        out = self.model.forward((v, self.x_test))
                        test_l2 += L2relLoss()(out, u).item()
                        test_mse += MSE()(out, u).item()

            for v, u in self.test_loader:
                v, u = v.to(self.device), u.to(self.device)
                # Enable gradient calculation for H1 relative error
                #out.requires_grad = True
                u.requires_grad = True
                out = self.model.forward((v, self.x_test))
                test_h1 += H1relLoss()(out, u).item()
                
            train_loss/= self.ntrain
            test_l2/= self.ntest
            test_mse/= self.ntest
            test_h1/= self.ntest

            t2 = default_timer()
            if ep % self.show_every == 0:
                if self.loss.get_name() == "L2_rel_md":
                    print(f'Epoch:{ep}  Time:{t2 - t1:.{2}f}  '
                      f'Train_loss_{self.loss.get_name()}:{train_loss:.{5}f}  '
                      f'Test_loss_{self.loss.get_name()}:{test_l2:.{5}f}  '
                      )
                else:
                    print(f'Epoch:{ep}  Time:{t2 - t1:.{2}f}  '
                      f'Train_loss_{self.loss.get_name()}:{train_loss:.{5}f}  '
                      f'Test_loss_l2:{test_l2:.{5}f}  '
                      f'Test_mse:{test_mse:.{5}f}  '
                      f'Test_loss_h1:{test_h1:.{5}f}'
                      )
    def train(self):
        t1 = default_timer()
        # Profile the training function
        #my_schedule = schedule(
        #skip_first=3,
        #wait=2,
        #warmup=1,
        #active=1)
        #with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
        #    record_shapes=True, profile_memory=True, schedule=my_schedule) as prof:
        for ep in range(self.epochs+1):
            self.single_train_step(ep,t1)
               #prof.step()
        #print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
        #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        # Calculate and print total memory allocated for CUDA and CPU
        #max_cpu_mem = max([item.cpu_memory_usage for item in prof.key_averages()])
        #max_cuda_mem = max([item.cuda_memory_usage for item in prof.key_averages()])

        #print(f"Max CPU Memory consumption: {max_cpu_mem / (1024 * 1024)} MB")
        #print(f"Max CUDA Memory consumption: {max_cuda_mem / (1024 * 1024)} MB")
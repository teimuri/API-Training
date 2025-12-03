from clearml import Task
import matplotlib.pyplot as plt
from ml_trainer.utils.logger import LoggerTemplate
# cfg = {'task': None}
class CL_Logger(LoggerTemplate):
    def __init__(self,name="trainer", log_dir="logs"):
        super().__init__(name,log_dir)
        self.task = Task.init()
        self.logger = self.task.get_logger()

    def scaler(self, title, series, value, iteration=0):
        self.logger.report_scalar(
            title=title,
            series=series,
            value=value,
            iteration=iteration,
        )
    
    def plot(self,title,series,data,iteration=0):

        plt.plot(data.pop('X'),data.pop('Y'),**data)
        self.logger.report_matplotlib_figure(
            title=title,
            series=series,
            figure=plt.gcf(),
            iteration=iteration,
        )
        plt.close()
    
    # def hyperparameters(self,params,name="Hyperparameters"):
    #     self.task.connect(params,name=name)

    # def on_train_start(self):
    #     pass
    
    # def on_epoch_start(self):
    #     pass
    
    # def on_epoch_end(self,*args,**kwargs):
    #     pass

    # def on_save_checkpoint(self):
    #     pass

    # def on_batch_end(self,*args,**kwargs):
    #     pass

    # def on_train_end(self,*args,**kwargs):
    #     pass

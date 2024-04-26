import subprocess
import time
import json
from datetime import datetime
from multiprocessing import Process

from utils import check_gpu_memory



class SequentialTrain:

    models_2d = {} # TODO
    models_3d = {
        'attention_unet': 0, 
        'medformer': 0, 
        'resunet': 0, 
        'swin_unetr': 0, 
        'unet++': 0, 
        'unetr': 0, 
        'vnet': 0, 
        'segformer': 3, 
        }
    
    dimensions = ['3d'] # '2d',

    len_battery_test = 17

    dataset = 'btcv'
    

    def __call__(self):
        """Method to run the models in sequence and parallel.
        """        

        processes = []

        for dimension in self.dimensions:
            if dimension == '2d':
                models = self.models_2d
            else:
                models = self.models_3d

            for model in list(models.keys()):
                
                gpu_memory = check_gpu_memory() # Verify GPU memory
                while gpu_memory < 14124:  # Wait until GPU memory is available  9124
                    print("Need more GPU memory. Waiting...")
                    time.sleep(55)  
                    gpu_memory = check_gpu_memory()
               
                print(f"\nTest {model} ({dimension})...")  # Start model battery test
                try:
                    process = Process(
                        target=self.run_model, 
                        args=(model, dimension, str(models[model])))
                    process.start()
                    processes.append(process)

                except: pass
                time.sleep(30)  
                
                
        for process in processes:
            process.join()

        print("All models are tested")

    @staticmethod
    def run_model(model, dimension, run_version):
        """Function to run the model to predict the battery test.
        When the prediction is finished, the time with more parameters are saved in a json file.
        The place for the json file is in the respective folder results.

        Args:
            model (str): Name of the model
            dimension (str): Number of dimensions (2d or 3d)
            run_version (str): Version of the model
        """        
        args = [
            '--mode', 'Train',
            '--model', model, 
            '--dimension', dimension,
            '--run_version', run_version
            ]
        
        cmd = ['python3', 'train.py'] + args
        start_time = datetime.now()
        subprocess.run(cmd)
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        
        result = {
            'model': model,
            'dimension': dimension,
            'run_version': run_version,
            'dataset': SequentialTrain.dataset, 
            'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S"),
            'elapsed_time': str(elapsed_time),
        }

        with open('./logs/'+SequentialTrain.dataset+'/'+model+'/'+dimension+'lightning_logs/version_'+run_version+'/time_train.json', 'w') as json_file:
            json.dump(result, json_file, indent=4)
            json_file.write(',\n') 


    

if __name__ == "__main__":
    SequentialTrain()()
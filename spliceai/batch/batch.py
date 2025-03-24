# Original source code modified to add prediction batching support by Invitae in 2021.
# Modifications copyright (c) 2021 Invitae Corporation.

# Invitae source code modified to improve GPU utilization
# Modifications made by Geert Vandeweyer (Antwerp University Hospital, Belgium)

import logging
import time
import shelve
import numpy as np
import os
import tensorflow as tf
import pickle
import gc
import socket
import sys
import argparse

#from spliceai.batch.batch_utils import extract_delta_scores, get_preds
sys.path.append('../../../spliceai')
from spliceai.batch.batch_utils import   get_preds, initialize_devices, initialize_one_device
from spliceai.utils import Annotator, get_delta_scores

class TensorFlowError(Exception):
    """Custom exception for TensorFlow  errors."""
    pass

SequenceType_REF = 0
SequenceType_ALT = 1


# options : revised from __main__
def get_options():

    parser = argparse.ArgumentParser(description='Version: 1.3.1')
    parser.add_argument('-P', '--port', metavar='port', required=True, type=int)    
    parser.add_argument('-R', '--reference', metavar='reference', required=True,
                        help='path to the reference genome fasta file')
    parser.add_argument('-A', '--annotation',metavar='annotation', required=True,
                        help='"grch37" (GENCODE V24lift37 canonical annotation file in '
                             'package), "grch38" (GENCODE V24 canonical annotation file in '
                             'package), or path to a similar custom gene annotation file')
    parser.add_argument('-T', '--tensorflow_batch_size', metavar='tensorflow_batch_size', type=int,
                        help='tensorflow batch size for model predictions')
    parser.add_argument('-V', '--verbose', action='store_true', help='enables verbose logging')
    parser.add_argument('-t','--tmpdir', metavar='tmpdir',type=str,default='/tmp/',required=False,
                        help="Use Alternate location to store tmp files. (Note: B=4096 equals to roughly 15Gb of tmp files)")
    parser.add_argument('-d','--device',metavar='device',type=str,required=True,
                        help="CPU/GPU device to deploy worker on")
    parser.add_argument('-S', '--simulated_gpus',metavar='simulated_gpus',default='0',type=int, required=False,
                        help="For development: simulated logical gpus on a single physical device to simulate a multi-gpu environment")
    parser.add_argument('-M', '--mem_per_logical', metavar='mem_per_logical',default=0,type=int, required=False, 
                        help="For simulated GPUs assign this amount of memory (Mb)")
    parser.add_argument('-G','--gpus',metavar='gpus',type=str,default='all',required=False,
                        help="Number of GPUs to use for SpliceAI. Provide 'all', or comma-seperated list of GPUs to use. eg '0,2' (first and third). Defaults to 'all'")
    args = parser.parse_args()

    return args

# set a trap so that on any unexpected error, the worker will send an error message back to the server.
def handle_exception(socket, exc_type, exc_value, exc_traceback):
    logger = logging.getLogger(__name__)
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    socket.send(str.encode(f"Error : {exc_value}"))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    sys.exit(1)

def setup_exception_hook(socket):
    sys.excepthook = lambda *exc_info: handle_exception(socket, *exc_info)

def main():
    # get arguments
    args = get_options()
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(name)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=loglevel,
    )
    logger = logging.getLogger(__name__)
    
    # setup the socket
    try:
        s = socket.socket()
        host = socket.gethostname()  # locahost
        port = args.port
        s.connect((host, port))
    except Exception as e:
        raise(e)
    
    # setup the exception hook
    setup_exception_hook(s)

    try:
        # initialize && assign device
        if args.simulated_gpus > 0:
            devices = [x for x in initialize_devices(args)[0] if x.name == args.device]
        else:
            # no simulation : expose only the requested device to tensor.
            devices = initialize_one_device(args)
    except Exception as e:
        raise e


    if not devices:
        # raise with message
        raise ValueError(f"Specified device '{args.device}' not found!")
        
    try:
        device = devices[0].name
        with tf.device(device):
            logger.info(f"Working on device {args.device}")
            # initialize the VCFPredictionBatch, pass (non-masked) device name
            worker = VCFPredictionBatch(args=args,logger=logger,socket=s) 
            # start working !
            worker.process_batches()
    except tf.errors.ResourceExhaustedError as e:
        raise TensorFlowError("Caught TensorFlow OOM Error!")
    except Exception as e:
        raise e
    # done.
    s.close()
    logger.info("Worker done. Shutting down")
    sys.exit(0)



# Class to handle predictions
class VCFPredictionBatch:
    def __init__(self, args, logger, socket): 
        self.args = args
        self.ann = None
        self.tensorflow_batch_size = args.tensorflow_batch_size
        self.tmpdir = args.tmpdir
        self.device = args.device
        self.logger = logger
        self.socket = socket

        # store batches of predictions using 'tensor_size|batch_idx' as key. 
        self.shelf_preds_name = f"spliceai_preds.{self.device[1:].replace(':','_')}.shelf"
        self.shelf_preds = shelve.open(os.path.join(self.tmpdir, self.shelf_preds_name))
        
    # monitor the queue and submit incoming batches.
    def process_batches(self):
        
        #host = socket.gethostname()  # locahost
        #port = self.args.port
        #try:
        #    s.connect((host,port))
        #except Exception as e:
        #    raise(e)
        # first response : server is running
        res = self.socket.recv(2048)
        # then start polling queue
        msg = "Ready for work..."
        # first load annotation
        if not self.ann:
            # load annotation 
            self.ann = Annotator(self.args.reference, self.args.annotation,cpu=True)
        while True:
            # send request for work
            self.socket.send(str.encode(msg))
            res = self.socket.recv(2048).decode('utf-8')
            # response can be a job, 'hold on' for empty queue, or 'Done' for all finished.
            if res == 'Hold On':
                msg = 'Ready for work...'
                time.sleep(0.1)
            elif res == 'Finished':
                self.logger.info("Worker done. Shutting down")
                break
            else:
                # got a batch id:
                with open(os.path.join(self.tmpdir,res),'rb') as p:
                    data = pickle.load(p)
                # remove pickled batch
                os.unlink(os.path.join(self.tmpdir,res))
                # process : stats are send back as next 'ready for work' result.
                try:
                   msg = self._process_batch(data['tensor_size'],data['batch_ix'], data['data'],data['length'])
                except Exception as e:
                   raise Exception(f"Error processing batch {data['tensor_size']}|{data['batch_ix']}: {repr(e)}")
                   # send error message back to server
                   #msg = "Error : {}".format(repr(e))
        # send signal to server thread to exit.
        self.socket.send(str.encode('Done'))
        self.logger.info(f"Closing Worker on device {self.device}")


    def _process_batch(self, tensor_size, batch_ix, prediction_batch, nr_preds):
        start = time.time()
        
        # Sanity check dump of batch sizes
        self.logger.debug('Tensor size : {} : batch_ix {} : nr.entries : {}'.format(tensor_size, batch_ix, nr_preds))

        # Run predictions && add to shelf.
        self.shelf_preds["{}|{}".format(tensor_size,batch_ix)] = np.mean(
            get_preds(self.ann, prediction_batch, self.tensorflow_batch_size), axis=0
        )
        
        # status
        duration = time.time() - start
        preds_per_sec = nr_preds / duration
        preds_per_hour = preds_per_sec * 60 * 60

        msg = 'Device {} : Finished in {:0.2f}s, per sec: {:0.2f}, per hour: {:0.2f}'.format(self.device, duration, preds_per_sec, preds_per_hour)
        self.logger.debug(msg)
        return msg

    
    
if __name__ == '__main__':
    main()

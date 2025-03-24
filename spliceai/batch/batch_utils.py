# Original source code modified to add prediction batching support by Invitae in 2021.
# Modifications copyright (c) 2021 Invitae Corporation.

# Invitae source code modified to improve GPU utilization
# Modifications made by Geert Vandeweyer (Antwerp University Hospital, Belgium)


import logging
import shelve
import os
import gc
import numpy as np
import tensorflow as tf
import pickle
import socket
from multiprocessing import Process
import subprocess
import time
import sys

from spliceai.utils import get_alt_gene_delta_score, is_record_valid, get_seq, \
    is_location_predictable, get_cov, get_wid, is_valid_alt_record, encode_seqs, create_unhandled_delta_score

sys.path.append('../../spliceai')
from spliceai.batch.data_handlers import VCFReader, VCFWriter

logger = logging.getLogger(__name__)



###########
## INPUT ##
###########
## routine to create the batches for prediction.
def prepare_batches(ann, args, tmpdir, prediction_queue,nr_workers): 
    try:
        # create the parser object
        vcf_reader = VCFReader(ann=ann, 
                           input_data=args.input_data,
                           prediction_batch_size=args.prediction_batch_size,
                           prediction_queue=prediction_queue,
                           tmpdir=tmpdir,dist=args.distance,     
                          )
        # parse records
        vcf_reader.add_records()
        # finalize last batches
        vcf_reader.finish(nr_workers)
        # close the shelf.
        vcf_reader.shelf_records.close()
        # stats 
        logger.info("Read {} vcf records, queued {} predictions".format(vcf_reader.total_vcf_records, vcf_reader.total_predictions))
    except Exception as e:
        logger.error(f"Error in prepare_batches: {repr(e)}")
        raise(e)





##############
## ANALYSIS ##
##############
## routine to start the worker Threads
def start_workers(prediction_queue, tmpdir, args,devices,mem_per_logical):
    # start server socket 
    s = socket.socket()
    host = socket.gethostname()  # locahost
    port = args.port
    # select a free socket
    if args.port is None:
        try:
            #sock = socket.socket()
            s.bind(('', 0))
            args.port = s.getsockname()[1]
            logging.debug(f"PORT:{args.port}")
        except Exception as e:
            logging.error(f"Error: {repr(e)}")
            sys.exit(1)
    else:
        logger.info(f"Starting server: {host}:{port}")
        try:
           s.bind((host,port))
           
        except Exception as e:
            logger.error(f"Cannot bind to port {port} : {e}")
            sys.exit(1)
    logging.info("Server started as : {} : {}".format(host,port))
    s.listen(5)
    # start client sockets & server threads.
    clientThreads = list()
    serverThreads = list()

    for device in devices:
        # launch the worker.
        logger.info(f"Starting worker on device {device.name}, output is available under {tmpdir}")
        cmd = ["python",os.path.join(os.path.dirname(os.path.realpath(__file__)),"batch.py"),"-S",str(args.simulated_gpus),"-M",str(int(mem_per_logical)), "-t",tmpdir,"-d",device.name, '-R', args.reference, '-A', args.annotation, '-T', str(args.tensorflow_batch_size), '-P', str(args.port)]
        if args.verbose:
            cmd.append('-V')
        logger.debug(cmd)
        fh_stdout = open(tmpdir+'/'+device.name.replace('/','_').replace(':','.')+'.stdout','w')
        fh_stderr = open(tmpdir+'/'+device.name.replace('/','_').replace(':','.')+'.stderr','w')
        
        p = subprocess.Popen(cmd ,stdout=fh_stdout, stderr=fh_stderr)
        clientThreads.append(p)
        ## then a new thread in the server for this connection.
        client, address = s.accept()
        logger.debug("Connected to : " + address[0] + ' : ' + str(address[1]))
        p = Process(target=_process_server,args=(client,device.name,prediction_queue,))
        p.start()
        serverThreads.append(p)
        logger.debug(f"Thread {device.name} activated!")

    return clientThreads, serverThreads, devices

# routine that runs in the server threads, issuing work to the worker_clients.
def _process_server(clientsocket,device,queue):
    # initial response
    clientsocket.send(str.encode('Server is online'))
    while True:
        msg = clientsocket.recv(2048).decode('utf-8')
        if msg == 'Done':
            logger.debug(f"Stopping thread {device}")
            break
        elif msg.startswith('Error'):
            # send finish signal to worker to shut down cleanly
            clientsocket.sendall(str.encode('Finished'))
            # then raise the error to the main thread.
            raise Exception(msg)
        elif not msg == 'Ready for work...':
            logger.info(msg)
        # send/get new item
        try:
            item = queue.get(False) 
        except Exception as e:
            #print(str(e))
            item = 'Hold On'
        
        # set reply
        clientsocket.sendall(str.encode(str(item)))

    logger.debug(f"Closing {device} socket.")
    clientsocket.close()



## get tensorflow predictions using batch-based submissions (used in worker clients)
def get_preds(ann, x, batch_size=32):
    logger.debug('Running get_preds with matrix size: {}'.format(x.shape))
    try:
        predictions = [ann.models[m].predict(x, batch_size=batch_size, verbose=0) for m in range(5)]  
    except Exception as e:
        # try a smaller batch (less efficient, but lower on memory). if it crashes again : it raises.
        logger.warning("TF.predict failed ({}). Retrying with smaller batch size".format(e))
        predictions = [ann.models[m].predict(x, batch_size=4, verbose=0) for m in range(5)]
    # garbage collection to prevent memory overflow... 
    gc.collect()
    return predictions
    

## initialize a single device, hide others (only for non-simulated gpus)
def initialize_one_device(args):
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return tf.config.list_logical_devices('CPU')
    # get the index of specified device.
    idx = None
    for i in range(len(gpus)):
        if gpus[i].name.replace('physical_','') == args.device:
            idx = i
            break
    if idx is None:
        logger.error("Device not found")
        logger.debug(idx)
        logger.debug(args.device)
        logger.debug([x.name.replace('physical_','') for x in gpus])
        raise Exception(f"specified device '{args.device}' not found.")
    # set visible
    tf.config.set_visible_devices(gpus[idx], 'GPU')
    logical_devices = tf.config.list_logical_devices('GPU')
    return logical_devices


## management routine to initialize gpu/cpu devices and do simulated logical devices if needed
def initialize_devices(args):
    ## need to simulate gpus ?
    gpus = tf.config.list_physical_devices('GPU')
    mem_per_logical = 0
    if gpus: 
        if args.simulated_gpus > 1:
            logger.warning(f"Simulating {args.simulated_gpus} logical GPUs on the first physical GPU device")
            try:
                gpu_mem_mb = _get_gpu_memory()
            except Exception as e:
                logger.error(f"Could not get GPU memory (needs nvidia-smi) : {e}")
                sys.exit(1)

            # Create n virtual GPUs with [available] / n GB memory each
            if hasattr(args,'mem_per_logical'):
                mem_per_logical = args.mem_per_logical
            else:
                mem_per_logical = int((gpu_mem_mb[0]-2048) / args.simulated_gpus)

            logger.info(f"Assigning {mem_per_logical}mb of GPU memory per simulated GPU.")
            try:
                device_list = [tf.config.LogicalDeviceConfiguration(memory_limit=mem_per_logical)] * args.simulated_gpus
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    device_list)
                logical_gpus = tf.config.list_logical_devices('GPU')
           
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                raise(e)
        prediction_devices = tf.config.list_logical_devices('GPU')

        if not args.gpus.lower() == 'all':
            idxs = [int(x) for x in args.gpus.split(',')]
            prediction_devices = [prediction_devices[x] for x in idxs]
    else:
        # run on cpu
        prediction_devices = tf.config.list_logical_devices('CPU')[0]

    logger.info("Using the following devices for prediction:")
    for d in prediction_devices:
        logger.info(f"  - {d}")

    return prediction_devices, mem_per_logical

## helper function to get gpu memory. 
def _get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values





# Original source code modified to add prediction batching support by Invitae in 2021.
# Modifications copyright (c) 2021 Invitae Corporation.

import sys
import argparse
import logging
import pysam
import time
import tempfile
from multiprocessing import Process,Queue,Pool
from functools import partial
import shutil
import tensorflow as tf
import subprocess as sp
import os
import socket

from spliceai.batch.batch_utils import prepare_batches,  start_workers,initialize_devices
from spliceai.utils import Annotator, get_delta_scores
from spliceai.batch.data_handlers import VCFWriter,VCFReader

try:
    from sys.stdin import buffer as std_in
    from sys.stdout import buffer as std_out
except ImportError:
    from sys import stdin as std_in
    from sys import stdout as std_out


def get_options():

    parser = argparse.ArgumentParser(description='Version: 1.3.1')
    parser.add_argument('-P', '--port', metavar='port', type=int, default=None,
                        help='specify a port for socket/socketserver communication')
    parser.add_argument('-I', '--input_data', metavar='input', nargs='?', default=std_in,
                        help='path to the input VCF file, defaults to standard in')
    parser.add_argument('-O', '--output_data', metavar='output', nargs='?', default=std_out,
                        help='path to the output VCF file, defaults to standard out')
    parser.add_argument('-R', '--reference', metavar='reference', required=True,
                        help='path to the reference genome fasta file')
    parser.add_argument('-A', '--annotation',metavar='annotation', required=True,
                        help='"grch37" (GENCODE V24lift37 canonical annotation file in '
                             'package), "grch38" (GENCODE V24 canonical annotation file in '
                             'package), or path to a similar custom gene annotation file')
    parser.add_argument('-D', '--distance', metavar='distance', nargs='?', default=50,
                        type=int, choices=range(0, 5000),
                        help='maximum distance between the variant and gained/lost splice '
                             'site, defaults to 50')
    parser.add_argument('-M', '--mask', metavar='mask', nargs='?', default=0,
                        type=int, choices=[0, 1],
                        help='mask scores representing annotated acceptor/donor gain and '
                             'unannotated acceptor/donor loss, defaults to 0')
    parser.add_argument('-B', '--prediction_batch_size', metavar='prediction_batch_size', default=1, type=int,
                        help='number of predictions to process at a time, note a single vcf record '
                             'may have multiple predictions for overlapping genes and multiple alts')
    parser.add_argument('-T', '--tensorflow_batch_size', metavar='tensorflow_batch_size', type=int,
                        help='tensorflow batch size for model predictions')
    parser.add_argument('-V', '--verbose', action='store_true', help='enables verbose logging')
    parser.add_argument('-t','--tmpdir', metavar='tmpdir',type=str,default='/tmp/',required=False,
                        help="Use Alternate location to store tmp files. (Note: B=4096 equals to roughly 15Gb of tmp files)")
    parser.add_argument('-G','--gpus',metavar='gpus',type=str,default='all',required=False,
                        help="Number of GPUs to use for SpliceAI. Provide 'all', or comma-seperated list of GPUs to use. eg '0,2' (first and third). Defaults to 'all'")
    parser.add_argument('-S', '--simulated_gpus',metavar='simulated_gpus',default='0',type=int, required=False,
                        help="For development: simulated logical gpus on a single physical device to simulate a multi-gpu environment")
    args = parser.parse_args()

    return args


def main():
    args = get_options()
    # logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(name)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=loglevel,
    )
    # sanity check for mandatory arguments    
    if None in [args.input_data, args.output_data, args.distance, args.mask]:
        logging.error('Usage: spliceai [-h] [-I [input]] [-O [output]] -R reference -A annotation '
                      '[-D [distance]] [-M [mask]] [-B [prediction_batch_size]] [-T [tensorflow_batch_size]] [-t [tmp_location]]')
        exit()
    # select a free socket
    if args.port is None:
        try:
            sock = socket.socket()
            sock.bind(('', 0))
            args.port = sock.getsockname()[1]
            logging.debug(f"PORT:{args.port}")
        except Exception as e:
            logging.error(f"Error: {repr(e)}")
            sys.exit(1)

    ## revised code for batched analysis
    if args.prediction_batch_size > 1:
        # initialize the GPU and setup to estimate
        try:
            devices,mem_per_logical = initialize_devices(args)
        except Exception as e:
            logging.error(f"Error initializing devices: {repr(e)}")
            sys.exit(1)

        # Default the tensorflow batch size to the prediction_batch_size if it's not supplied in the args
        args.tensorflow_batch_size = args.tensorflow_batch_size if args.tensorflow_batch_size else args.prediction_batch_size
        
        # load annotation data:
        try:
            ann = Annotator(args.reference, args.annotation)
            logging.debug("Annotation loaded.")
        except Exception as e:
            logging.error(f"Error loading annotation: {repr(e)}")
            sys.exit(1)
        # run         
        try:
            run_spliceai_batched(args,ann,devices,mem_per_logical)
        except Exception as e:
            logging.error(f"Error running SpliceAI: {repr(e)}")
            sys.exit(1)

    else: # run original code:
        # load annotation
        ann = Annotator(args.reference, args.annotation)
        # run scoring
        run_spliceai(args, ann) 


## revised logic to allow batched tensorflow analysis on multiple GPUs
def run_spliceai_batched(args, ann,devices,mem_per_logical): 
    
    ## GOAL 
    ##  - launch a reader that preps & pickles input vcf
    ##  - launch per GPU/device, using sockets, a utility script that runs tasks from the queue on that device.
    ##  - communicate through sockets : server threads issue items from the queue to worker clients
    ##  - when all predictions are done, build the output vcf.


    ## track start time
    start_time = time.time()
    ## variables:
    input_data = args.input_data
    output_data = args.output_data
    distance = args.distance
    mask = args.mask
    prediction_batch_size = args.prediction_batch_size
    tensorflow_batch_size = args.tensorflow_batch_size

    ## mk a temp directory 
    tmpdir = tempfile.mkdtemp(dir=args.tmpdir) # TemporaryDirectory(dir=args.tmpdir)
    #tmpdir = tmpdir.name
    logging.info("Using tmpdir : {}".format(tmpdir))
        
    # creates a queue with max 10 ready-to-go batches in it.
    prediction_queue = Queue(maxsize=10)
    # starts processing & filling the queue.  
    reader_args={'ann':ann, 'args':args, 'tmpdir':tmpdir, 'prediction_queue': prediction_queue, 'nr_workers': len(devices)}
    reader = Process(target=prepare_batches, kwargs=reader_args) 
    reader.start()    
    logging.debug("Reader started") 
    worker_clients, worker_servers, devices = start_workers(prediction_queue,tmpdir,args,devices,mem_per_logical)
    logging.debug("workers started")
    ## wait for everything to finish.
    #   => If exit codes != 0 are detected, the main process will exit with the first non-zero exit code.
    while True:
        # any exit codes defined and != 0 ?
        exit_codes = [p.exitcode for p in worker_servers + [reader] if p.exitcode is not None] + [p.poll() for p in worker_clients if p.poll() is not None]
        logging.debug("exit codes: {}".format(exit_codes))
        if any(rc != 0 for rc in exit_codes):
            logging.error("Error encountered Exiting.")
            # kill all processes
            for p in worker_servers + [reader]:
                if p.is_alive():
                    p.kill()
            for p in worker_clients:
                if p.poll() is None:
                    p.kill()
            # and exit
            sys.exit(1) 
        if len(exit_codes) == len(worker_servers + [reader] + worker_clients):
            break
        time.sleep(30)

    # readers sends finish signal to workers
    logging.info("Cleanup VCF reader")
    reader.join()
    logging.debug("Reader joined!")
    # clients receive signal, send it to servers.
    logging.info("Cleaning up workers.")
    for p in worker_clients:
        # subprocesses : wait()
        p.wait()
    logging.debug("Workers are done!")
    logging.info("Waiting for servers to join.")
    for p in worker_servers:
        # mp processes : join()
        p.join()
    logging.debug("Servers are done")

    # stats without writing phase
    prediction_duration = time.time() - start_time
     
    # write results. in/out from args, devices to get shelf names
    logging.info("Writing output file")
    writer = VCFWriter(args=args,tmpdir=tmpdir,devices=devices,ann=ann)
    writer.process()

    # clear out tmp
    shutil.rmtree(tmpdir)
    ## stats
    overall_duration = time.time() - start_time
    preds_per_sec = writer.total_predictions / prediction_duration
    preds_per_hour = preds_per_sec * 60 * 60
    logging.info("Analysis Finished. Statistics:")
    logging.info("Total RunTime: {:0.2f}s".format(overall_duration))
    logging.info("Prediction RunTime: {:0.2f}s".format(prediction_duration))
    logging.info("Processed Records: {}".format(writer.total_vcf_records))
    logging.info("Processed Predictions: {}".format(writer.total_predictions))
    logging.info("Overall performance : {:0.2f} predictions/sec ; {:0.2f} predictions/hour".format(preds_per_sec, preds_per_hour))


# original flow : record by record reading/predict/write
def run_spliceai(args, ann):
    # assign variables
    input_data = args.input_data
    output_data = args_output_data
    distance = args.distance
    mask = args.mask
    
    # open infile
    try:
        vcf = pysam.VariantFile(input_data)
    except (IOError, ValueError) as e:
        logging.error('{}'.format(e))
        exit()

    header = vcf.header
    header.add_line('##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAIv1.3.1 variant '
                    'annotation. These include delta scores (DS) and delta positions (DP) for '
                    'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
                    'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">')

    try:
        output_data = pysam.VariantFile(output_data, mode='w', header=header)
    except (IOError, ValueError) as e:
        logging.error('{}'.format(e))
        exit()

    for record in vcf:
            scores = get_delta_scores(record, ann, distance, mask)
            if len(scores) > 0:
                record.info['SpliceAI'] = scores
            output_data.write(record)
   
    # close VCF
    vcf.close()
    output_data.close()

if __name__ == '__main__':
    main()

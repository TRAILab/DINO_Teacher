import datetime
import logging
import time
from collections import OrderedDict, abc
from typing import List, Union
from contextlib import ExitStack
import torch
from torch import nn
import pickle
import os

from detectron2.utils import comm
from detectron2.utils.logger import log_every_n_seconds
from detectron2.evaluation import (
    DatasetEvaluator,
    DatasetEvaluators,
    print_csv_format,
    inference_context,
)
from detectron2.structures import Instances

vit_dict = {384:'vits', 768:'vitb', 1024:'vitl', 1536:'vitg'}

def test_and_gen(
        cfg, 
        model, 
        parent_build_test_loader,
        parent_build_evaluator, 
        evaluators=None,
    ):
    """
    Duplicate of the test function in the Trainer class that saves the predictions.
    Taken from detectron2.engine.defaults.DefaultTrainer.test
    """
    output_dir=cfg.OUTPUT_DIR
    logger = logging.getLogger(__name__)
    if isinstance(evaluators, DatasetEvaluator):
        evaluators = [evaluators]
    if evaluators is not None:
        assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
            len(cfg.DATASETS.TEST), len(evaluators)
        )

    results = OrderedDict()
    for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
        data_loader = parent_build_test_loader(cfg, dataset_name)
        # When evaluators are passed in as arguments,
        # implicitly assume that evaluators can be created before data_loader.
        if evaluators is not None:
            evaluator = evaluators[idx]
        else:
            try:
                evaluator = parent_build_evaluator(cfg, dataset_name)
            except NotImplementedError:
                logger.warn(
                    "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                    "or implement its `build_evaluator` method."
                )
                results[dataset_name] = {}
                continue
        results_i, predictions = inference_with_outputs(model, data_loader, evaluator, keep_predictions=True)
        results[dataset_name] = results_i
        if comm.is_main_process():
            assert isinstance(
                results_i, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results_i
            )
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
        
            if cfg.MODEL.BACKBONE.NAME == 'build_dino_vit_backbone':
                dim = model.backbone.encoder.embed_dim
                model_size = vit_dict[dim]
                file_out = ('/').join([output_dir, 'predictions', dataset_name + '_dino_anno_{}.pkl'.format(model_size)])
            else:
                file_out = ('/').join([output_dir, 'predictions', dataset_name + '_anno.pkl'.format])

            os.makedirs(os.path.dirname(file_out), exist_ok=True)
            with open(file_out, 'wb') as f_out:
                pickle.dump(predictions, f_out)
            logger.info("Predictions saved to {}".format(file_out))

    if len(results) == 1:
        results = list(results.values())[0]
    return results


def inference_with_outputs(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], keep_predictions=True
):
    """
    Modified from detectron2.engine.defaults.inference_on_dataset
    """
    num_devices = comm.get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        if keep_predictions:
            output_list = []
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if keep_predictions:
                instances = Instances(outputs[0]['instances'].image_size)
                instances.pred_boxes = outputs[0]['instances'].pred_boxes.tensor.cpu()
                instances.scores = outputs[0]['instances'].scores.cpu()
                instances.pred_classes = outputs[0]['instances'].pred_classes.cpu()
                out_dict = {'file_name':inputs[0]['file_name'], 'image_id':inputs[0]['image_id'], 'height':inputs[0]['height'], 'width':inputs[0]['width'], 'instances_dino':instances}
                output_list.append(out_dict)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    if keep_predictions:
        return results, output_list
    else:
        return results, None

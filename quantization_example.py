import nncf
import torch
import torch_tensorrt
import timm
import time
import numpy as np
import torch.backends.cudnn as cudnn
from pathlib import Path
from torchvision import datasets, transforms
from fastdownload import FastDownload
from nncf.torch.dynamic_graph.context import no_nncf_trace

efficientnet_b0_model = timm.create_model('efficientnet_b0',pretrained=True)
model = efficientnet_b0_model.eval().to("cuda")

def benchmark(model, input_shape=(1024, 1, 224, 224), dtype='fp32', nwarmup=50, nruns=10000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))
                
def quantize_model(model):
    DATASET_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
    DATASET_PATH = '~/.cache/nncf/datasets'
    DATASET_CLASSES = 10
        
    def download_dataset() -> Path:
        downloader = FastDownload(base=DATASET_PATH,
                                archive='downloaded',
                                data='extracted')
        return downloader.get(DATASET_URL)
    
    dataset_path = download_dataset()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        root=f'{dataset_path}/val',
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=128, num_workers=4, shuffle=False)
    
    def transform_fn(data_item):
        images, _ = data_item
        return images
    
    calibration_dataset = nncf.Dataset(val_loader, transform_fn)
    quantized_model = nncf.quantize(model, calibration_dataset)
    return quantized_model


# print("Compiling model")
# trt_model_fp16 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.half)],
#     enabled_precisions = {torch.half}, # Run with FP16
#     workspace_size = 1 << 22
# )

# print("Benchmarking model")
# benchmark(trt_model_fp16, input_shape=(128, 3, 224, 224), dtype='fp16', nruns=100)

print("Quantizing model")
quantized_model = quantize_model(model)

stripped_q_model = quantized_model.controller.prepare_for_inference(make_model_copy=False) # NNCF quantize_impl for torch should be patched to get the controller

print("Compiling quantized model")
#with no_nncf_trace():
trt_model_int8 = torch_tensorrt.compile(stripped_q_model, inputs = [torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.float)],
    enabled_precisions = {torch.int8}, 
    workspace_size = 1 << 22
)

print("Benchmarking quantized model")
benchmark(trt_model_int8, input_shape=(128, 3, 224, 224), nruns=100)

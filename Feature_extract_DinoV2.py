import os

import openslide
import torch
import h5py
import staintools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
import multiprocessing as mp

class Feature_DinoV2:
    def __init__(self,model,wsi,coord,save_path,targetImage,patch_level=0,patch_size=224,colornorm=True):
        self.model = model
        self.WSIobject= wsi
        self.sampleName = os.path.basename(coord).split('.svs')[0]
        self.coords_file_path = coord
        self.save_path = save_path
        self.targetImage = targetImage
        self.patch_level= patch_level
        self.patch_size = patch_size
        self.colornorm = colornorm
        if self.colornorm:
            i2 = staintools.read_image(self.targetImage)
            self.normalizer = staintools.StainNormalizer(method='macenko')
            self.normalizer.fit(i2)

    @staticmethod
    def extract_feature(model, image):
        """extract backbone feature of dino v2 model on a single image"""
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        transform = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        net_input = transform(image).unsqueeze(0)

        with torch.no_grad():
            feature = model(net_input.to(device)).squeeze().cpu().numpy()
        return feature

    @staticmethod
    def PatchRead(coord,WSIobject,patch_level,patch_size,normalizer,model):
        patchimg = WSIobject.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB')
        normalizer=normalizer
        try:
            transformImg = Image.fromarray(normalizer.transform(np.array(patchimg)))
        except:
            print('This patch is a background!!!')
            transformImg = patchimg
        feature= Feature_DinoV2.extract_feature(model,transformImg)
        return feature

    def Run(self):
        save_path=self.save_path
        file = h5py.File(self.coords_file_path, 'r')
        dset = file['coords']
        coords = dset[:] ###Extract
        print(coords)
        indices=np.arange(len(coords))
        file.close()

        num_workers = mp.cpu_count()
        if num_workers > 4:
            num_workers = 4
        pool = mp.Pool(num_workers)
        results = []
        for coord in tqdm(coords):
            result=Feature_DinoV2.PatchRead(coord,self.WSIobject, self.patch_level, self.patch_size, self.normalizer,self.model)
            results.append(result)
        results=np.array(results)
        # iterable = [(coord, self.WSIobject, self.patch_level, self.patch_size, self.targetImage) for coord in coords]
        # print(iterable)
        # results = pool.starmap(Feature_DinoV2.PatchRead, iterable)
        # results = np.array([result for result in results if result is not None])
        data_shape=results.shape
        data_type=results.dtype
        fileOut = h5py.File(save_path+'/'+self.sampleName+".Feature.h5",'a')
        dset_feature = fileOut.create_dataset('Features',shape=data_shape, dtype=data_type)
        dset_feature[:]=results
        fileOut.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--model_size",
        type=str,
        default="largest",
        choices=["small", "base", "large", "largest"],
        help="DinoV2 model type",
    )
    parser.add_argument(
        "-p",
        "--model_path",
        type=str,
        default="/home/yangml/.cache/torch/hub/facebookresearch_dinov2_main",
        help="path to dinov2 model, useful when github is unavailable",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        help="root folder to save output results",
    )
    parser.add_argument(
        "-f",
        "--file_csv",
        type=str,
        required=True,
        help="path to a query image file or image folder",
    )
    parser.add_argument(
        "-t",
        "--target_image",
        type=str,
        default="/Data/yangml/Important_script/Target.jpg",
        help="reference image for color normalization",
    )

    args = parser.parse_args()
    model_size = args.model_size
    models_dict = {
        "small": "dinov2_vits14", ### feature Dim 384
        "base": "dinov2_vitb14", ### feature Dim 768
        "large": "dinov2_vitl14", ### feature Dim 1024
        "largest": "dinov2_vitg14", ### feature Dim 1536
    }
    model_used = models_dict[model_size]
    model_folder = (
        "facebookresearch/dinov2" if args.model_path is None else args.model_path
    )
    model_source = "github" if args.model_path is None else "local"


    model = torch.hub.load(model_folder, model_used,
                                   pretrained=True, source=model_source)

    df = pd.read_csv(args.file_csv)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    for idx,row in df.iterrows():
        print("[{}]: {}".format(idx,row[0]))
        WSIObject = openslide.open_slide(row[1])
        Feature = Feature_DinoV2(model=model,
                                 wsi=WSIObject,
                                 coord=row[2],
                                 save_path=args.output,
                                 targetImage="/Data/yangml/Important_script/Target.jpg")
        Feature.Run()
        WSIObject.close()
if __name__ == "__main__":
    main()
###Testing
# model = torch.hub.load('/home/yangml/.cache/torch/hub/facebookresearch_dinov2_main', "dinov2_vitg14",
#                                    pretrained=True, source="local")
# WSIObject = openslide.open_slide("/Data/yangml/Important_script/WSIimmune/SVS/TCGA-FS-A1ZN-01Z-00-DX1.7A4FB831-E03E-43B3-9CD9-8441682F146C.svs")
# Feature=Feature_DinoV2(model=model,
#                        wsi=WSIObject,
#                        coord="/Data/yangml/Important_script/WSIimmune/SVS/patches/TCGA-FS-A1ZN-01Z-00-DX1.7A4FB831-E03E-43B3-9CD9-8441682F146C.h5",
#                       save_path="/Data/yangml/Important_script/WSIimmune/SVS",
#                        targetImage="/Data/yangml/Important_script/Target.jpg")
# Feature.Run()
# WSIObject.close()
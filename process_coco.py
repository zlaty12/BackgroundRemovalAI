import os
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
from PIL import Image

def process_coco_data(dataDir, dataType, output_dir, limit=None):
    annFile = f'{dataDir}/annotations/instances_{dataType}.json'
    print(f"Looking for annotation file at: {annFile}")
    
    if not os.path.exists(annFile):
        print(f"Error: Annotation file not found at {annFile}")
        return

    coco = COCO(annFile)
    
    

    # Get all image ids
    imgIds = coco.getImgIds()

    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    for i, imgId in enumerate(imgIds[:limit]):
        # Load image
        img = coco.loadImgs(imgId)[0]
        I = io.imread(f"{dataDir}/images/{dataType}/{img['file_name']}")
        
        # Load annotations
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        
        # Create mask
        mask = np.zeros((img['height'],img['width']))
        for ann in anns:
            mask = np.maximum(coco.annToMask(ann), mask)
        
        # Save original image
        Image.fromarray(I).save(os.path.join(output_dir, 'images', img['file_name']))
        
        # Save mask
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            os.path.join(output_dir, 'masks', img['file_name'].replace('.jpg', '_mask.png'))
        )

        if i % 100 == 0:
            print(f"Processed {i+1} images")

        if i+1 == limit:
            break

    print("Processing complete!")

if __name__ == "__main__":
    dataDir = 'D:/AIEMB/coco_data'  
    dataType = 'train2017'  
    output_dir = 'D:/AIEMB/ProcessedData'
    
    # Check if the directories exist
    if not os.path.exists(dataDir):
        print(f"Error: Data directory not found at {dataDir}")
    elif not os.path.exists(f"{dataDir}/annotations"):
        print(f"Error: Annotations directory not found at {dataDir}/annotations")
    elif not os.path.exists(f"{dataDir}/images/{dataType}"):
        print(f"Error: Images directory not found at {dataDir}/images/{dataType}")
    else:
        process_coco_data(dataDir, dataType, output_dir)
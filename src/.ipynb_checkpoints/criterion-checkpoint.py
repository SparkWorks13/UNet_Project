# This is where you define your loss function and related helper functions.
from pycocotools.coco import COCO

def get_id_from_path(data, path):
    for i in data['images']:
        #print(str(train_dir) + "\\"+i['file_name'], "\n", path, sep='')
        if i['file_name'] in path:
            return i['id']

    return None

def create_mask(ID, coco):
	cat_ids = coco.getCatIds()
	anns_ids = coco.getAnnIds(imgIds=ID, catIds=cat_ids, iscrowd=None) # ID
	anns = coco.loadAnns(anns_ids)

	mask = coco.annToMask(anns[0])
	for j in range(len(anns)): # in case there are more than one annotation, yet to encounter any
		mask += coco.annToMask(anns[j])

	return mask
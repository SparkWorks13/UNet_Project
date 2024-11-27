# This is where you define your loss function and related helper functions.
def get_id_from_path(data, path):
    for i in data['images']:
        #print(str(train_dir) + "\\"+i['file_name'], "\n", path, sep='')
        if str(train_dir) + "\\"+i['file_name'] == path:
            return i['id']

    return None
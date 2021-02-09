  
base_path = '/disk/scratch1/s1960707/datasets/data'
experiment_type = 'maskrcnn_vf'


paths = {
    'rl': {
        'models.select': base_path + '',
        'models.assign': base_path + '',
        'masks': base_path + '',
        'rewards': base_path + ''
    },
    'supervised': {
        'models.select': base_path + '',
        'models.assign': base_path + '',
        'masks': base_path + ''
    },
    'oracle': {
        'masks': base_path + '/oracle/masks/'
    }
}

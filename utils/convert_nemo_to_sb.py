import os
import json

def convert(input_path: str, output_path: str) -> str:

    output_item = {}

    with open(input_path, mode='r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            input_item = json.loads(line.strip('\r\n'))
            output_item[os.path.basename(input_item['audio_filepath'])] = {
                'wav': os.path.join('{data_root}', input_item['audio_filepath']),
                'length': input_item['duration'],
                'language': input_item['class']
            }

    
    with open(output_path, mode='w', encoding='utf-8') as fw:
        json.dump(output_item, fw, indent=2)

    return output_path
        
if __name__ == '__main__':

    for split in ('train', 'dev', 'test'):

        INPUT_PATH = f'/home/daniel/projects/speechbrain_lid/tests/{split}_manifest.json'
        OUTPUT_PATH = f'/home/daniel/projects/speechbrain_lid/tests/{split}_manifest_sb.json'

        convert(INPUT_PATH, OUTPUT_PATH)

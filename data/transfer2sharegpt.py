import json
import re
def generate_data():

    with open('Data4Qwenvl/Processed_Train.json', 'r', encoding='utf-8') as file:
        data = json.load(file) # [{'id', 'conversations'} ...]

    for data_block in data:
        data_block['images'] = [data_block.pop('id') + '.png']

        conversation_rounds = len(data_block['conversations'])
        data_block['messages'] = data_block.pop('conversations')

        for item in data_block['messages']:
            item['role'] = item.pop('from')
            item['content'] = item.pop('value')


    with open('Train_sharegpt.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def replace_token(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for data_block in data:

        for item in data_block['messages']:
            # item['content'] = re.sub(r'<img>.*?</img>', '<image>', item['content'])
            item['content'] = re.sub(r'<box_start>(.*?)<box_end>', r'<|box_start|>\1<|box_end|>', item['content'])
            # item['content'] = re.sub(r'<ref>(.*?)</ref>', r'<|object_ref_start|>\1<|object_ref_end|>', item['content'])

    with open('Train_sharegpt_final.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    replace_token('')


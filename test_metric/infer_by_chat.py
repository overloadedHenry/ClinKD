import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader

import re

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def add_abs_img_path_to_question(text, image_path):

    pattern = r'<img>.*?</img>'
    result = re.sub(pattern, f'<img>{image_path}</img>', text)
    # print(result)

    return result



class VQADataset(Dataset):
    def __init__(self, test, img_root_path):
        self.test = json.load(open(test, "r"))
        self.img_root_path = img_root_path
 
    def __len__(self):
        return len(self.test)
 
    def __getitem__(self, idx):
        data = self.test[idx]

        image_name = data['id'] + ".png"
        image_path = os.path.join(self.img_root_path, image_name)
        if not os.path.exists(image_path):
            print(f"The path '{image_path}' does not exist.")
        question = data['conversations'][0][0]
        input_text = add_abs_img_path_to_question(question, image_path)
        annotation = data['conversations'][0][1]

        return {
            "idx": idx,
            "image_path": image_path,
            "question": question,
            "annotation": annotation,
            "input_text": input_text,
            'messages' : [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path,
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ]
        }
    
def collate_fn(batches, tokenizer):
    idxs = [_["idx"] for _ in batches]
    image_paths = [_["image_path"] for _ in batches]
    questions = [_["question"] for _ in batches]
    input_texts = [_["input_text"] for _ in batches]
    annotations = [_["annotation"] for _ in batches]
    messages = [_["messages"] for _ in batches]
    input_ids = tokenizer(input_texts, return_tensors="pt", padding=True)
    return (idxs, input_ids.input_ids, input_ids.attention_mask, questions, annotations, image_paths, messages)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/path/to/checkpoint_directory')
    parser.add_argument('--json_path', type=str, default='')
    parser.add_argument('--top_p', type=str, default='03')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=80)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--img_root_path', type=str, default='/media/adminroot/disk_2/ghy/GMAI___SA-Med2D-20M/raw/SAMed2Dv1/images')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    random.seed(args.seed)
    torch.manual_seed(1234)



    model_name = "--".join([args.checkpoint.split('/')[-2], args.checkpoint.split('/')[-1]])
    # model_name = args.checkpoint.split('/')[-1]
    save_json_name = os.path.basename(args.json_path[:-5]) + f"_pred_bfloat16_qwenvl2_infer.jsonl"
    save_root = f"/media/adminroot/disk_2/ghy/workspace/test_metric/{model_name}_nosample_nopad"
    os.makedirs(save_root, exist_ok=True)
    res_file = os.path.join(save_root, save_json_name)

    print(f"save_json_name: {res_file}")
    
    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    # tokenizer.padding_side = "left"
    # tokenizer.pad_token_id = tokenizer.eod_id


    dataset = VQADataset(test=args.json_path, img_root_path=args.img_root_path)
    dataloader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, collate_fn=partial(collate_fn, tokenizer=tokenizer),
        shuffle=False, num_workers=args.num_workers
    )
        
 
    model = Qwen2VLForConditionalGeneration.from_pretrained("/media/adminroot/disk_2/ghy/workspace/train_2025-01-08-11-04-20", torch_dtype="auto", device_map="auto")

    min_pixels = 512 * 512
    max_pixels = 1024 * 1024

    processor = AutoProcessor.from_pretrained("/media/adminroot/disk_2/ghy/workspace/train_2025-01-08-11-04-20", min_pixels=min_pixels, max_pixels=max_pixels)

    all_idx = 0

    for idxs, input_ids, attention_masks, questions, annotations, image_paths, messages in tqdm(dataloader):
        # input_ids = input_ids.to(model.device)
        # input_ids = tokenizer.batch_decode(input_ids)
        responses = []
        # for item in input_ids:
        #     ans, history = model.chat(tokenizer, query=item, history=[], generation_config=model.generation_config)
        #     responses.append(ans)
            
        # responses = [tokenizer.decode(_, skip_special_tokens=True) for _ in pred]
        
        for message in messages:
            text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=80)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            responses.append(output_text)

        for i in range(len(idxs)):
            # 组合成一个 JSON 对象
            data = {
                "id": idxs[i],
                "image": os.path.basename(image_paths[i]),
                "question": questions[i],
                "annotation": annotations[i],
                "response": responses[i]
            }
            # 将 JSON 对象写入 JSON Lines 文件
            with open(res_file, 'a') as f:
                json.dump(data, f)
                f.write('\n')  # 换行，以便下一个 JSON 对象写入新的一行
                all_idx += 1
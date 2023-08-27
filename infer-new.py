import torch
import numpy as np
import itertools
from lightning_modules.bros_spade_module import parse_initial_words,parse_subsequent_words
from lightning_modules.bros_spade_rel_module import parse_relations
import cv2
from model import get_model
from utils import get_class_names, get_config
from bros import BrosTokenizer
from evaluate import load_model_weight, get_eval_kwargs_spade
import json
from process import postprocess_keypair

print("OK")
tokenizer = BrosTokenizer.from_pretrained("naver-clova-ocr/bros-base-uncased")
pad_token_id = tokenizer.vocab["[PAD]"]
cls_token_id = tokenizer.vocab["[CLS]"]
sep_token_id = tokenizer.vocab["[SEP]"]
unk_token_id = tokenizer.vocab["[UNK]"]
max_seq_length = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_kwargs = get_eval_kwargs_spade('datasets/funsd_spade', max_seq_length)
cfg = get_config()
net = get_model(cfg)
load_model_weight(net, cfg.pretrained_model_file)
# net.to("cuda:1")
net.to(device)
net.eval()

def create_input_bros(img, boxes, texts):
    # img [H, W, C]
    # boxes [1, 8]
    width, height = img.shape[:2]
    input_ids = np.ones(max_seq_length, dtype=int) * pad_token_id
    bbox = np.zeros((max_seq_length, 8), dtype=np.float32)
    attention_mask = np.zeros(max_seq_length, dtype=int)
    are_box_first_tokens = np.zeros(max_seq_length, dtype=np.bool_)
    
    list_tokens = []
    list_bbs = []
    box2token_span_map = []
    box_to_token_indices = []
    cum_token_idx = 0
    cls_bbs = [0.0] * 8
    for word_idx, (bb, text) in enumerate(zip(boxes, texts)):
        this_box_token_indices = []
        bb = np.array(bb).reshape(-1, 2).tolist()
        tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        if len(tokens) == 0:
            tokens.append(unk_token_id)
        if len(list_tokens) + len(tokens) > max_seq_length - 2:
            break
        box2token_span_map.append([len(list_tokens) + 1, len(list_tokens) + len(tokens) + 1])
        list_tokens += tokens
    
        for coord_idx in range(4):
            bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
            bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))
        bb = list(itertools.chain(*bb))
        bbs = [bb for _ in range(len(tokens))]
        for _ in tokens:
            cum_token_idx += 1
            this_box_token_indices.append(cum_token_idx)
        list_bbs.extend(bbs)
        box_to_token_indices.append(this_box_token_indices)
    sep_bbs = [width, height] * 4
    
    # add [CLS] and [SEP]
    list_tokens = ([cls_token_id] + list_tokens[: max_seq_length - 2] + [sep_token_id])
    if len(list_bbs) == 0:
        # when len(words_info) == 0
        list_bbs = [cls_bbs] + [sep_bbs]
    else:
        list_bbs = [cls_bbs] + list_bbs[:max_seq_length - 2] + [sep_bbs]

    len_list_tokens = len(list_tokens)
    input_ids[:len_list_tokens] = list_tokens
    attention_mask[:len_list_tokens] = 1
    bbox[:len_list_tokens, :] = list_bbs
    # normalize box
    bbox[:, [0, 2, 4, 6]] = bbox[:, [0, 2, 4, 6]] / width
    bbox[:, [1, 3, 5, 7]] = bbox[:, [1, 3, 5, 7]] / height
    st_indices = [indices[0] for indices in box_to_token_indices if indices[0] < max_seq_length]
    are_box_first_tokens[st_indices] = True

    input_ids = torch.from_numpy(input_ids).unsqueeze(0).to(device)
    bbox = torch.from_numpy(bbox).unsqueeze(0).to(device)
    attention_mask = torch.from_numpy(attention_mask).unsqueeze(0).to(device)
    are_box_first_tokens = torch.from_numpy(are_box_first_tokens).unsqueeze(0).to(device)
    return_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "attention_mask": attention_mask,
            "are_box_first_tokens": are_box_first_tokens,
            "box_to_token_indices": box_to_token_indices
            }
    return return_dict


def key_extract():
    img = cv2.imread('/home/ubuntu/bros/images_023.jpg', cv2.IMREAD_COLOR)
    texts, boxes = get_ocr_result('/home/ubuntu/bros/result.json')
    input_data_item = create_input_bros(img, boxes, texts)
    # print(input_data_item['input_ids'].size())
    # print(input_data_item['bbox'].size())
    # print(input_data_item['attention_mask'].size())
    # print(input_data_item['are_box_first_tokens'].size())
    with torch.no_grad():
        head_outputs = net(input_data_item)
    
    class_names = eval_kwargs['class_names']
    dummy_idx = eval_kwargs['dummy_idx']
    itc_outputs = head_outputs['itc_outputs']
    stc_outputs = head_outputs['stc_outputs']
    # el_outputs = head_outputs['el_outputs']
    pr_itc_labels = torch.argmax(itc_outputs, -1)
    pr_stc_labels = torch.argmax(stc_outputs, -1)
    # pr_el_labels = torch.argmax(el_outputs, -1)
    # print('itc out', pr_itc_labels)
    # print('stc out', pr_stc_labels)
    # print('el out', pr_el_labels)

    are_box_first_tokens = input_data_item['are_box_first_tokens']
    attention_mask = input_data_item['attention_mask']
    box_to_token_indices = input_data_item['box_to_token_indices']
    # box_to_token_indices = [[v.cpu().numpy().tolist()[0] for v in u] for u in box_to_token_indices]
    # print('box token', are_box_first_tokens)
    # print('attn mask', attention_mask)
    bsz = pr_itc_labels.shape[0]
    
    list_results = [] 
    for example_idx in range(bsz):
        pr_init_words = parse_initial_words(pr_itc_labels[example_idx], are_box_first_tokens[example_idx], class_names)
        
        pr_class_words = parse_subsequent_words(
            pr_stc_labels[example_idx], attention_mask[example_idx], pr_init_words, dummy_idx)

        
        value_text, key_text = postprocess_keypair(pr_class_words, box_to_token_indices, texts, boxes)
        print("value \n",value_text,"\n key",key_text)
def get_ocr_result(json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    words = []
    coors = []
    for line in data['textAnnotations'][1::]:
        words.append(line['description'])
        coor = []
        for coords in line['boundingPoly']['vertices']:
            coor.append(coords['x'])
            coor.append(coords['y'])
        coors.append(coor)
    return words,coors
def get_ocr_orginal_result(json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    words = []
    coors = []
    data = data["form"]

    for line in data:
        for value in line['words']:
            words.append(value['text'])
            coor = []
            coor.append(value['box'][0])
            coor.append(value['box'][1])
            coor.append(value['box'][2])
            coor.append(value['box'][1])
            coor.append(value['box'][2])
            coor.append(value['box'][3])
            coor.append(value['box'][0])
            coor.append(value['box'][3])
            coors.append(coor)
    return words,coors
key_extract()
print('done')
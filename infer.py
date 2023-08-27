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


class BrosExtract:
    def init(self, max_seq_length=512):
        super().init()
        self.tokenizer = BrosTokenizer.from_pretrained("naver-clova-ocr/bros-base-uncased")
        self.pad_token_id = self.tokenizer.vocab["[PAD]"]
        self.cls_token_id = self.tokenizer.vocab["[CLS]"]
        self.sep_token_id = self.tokenizer.vocab["[SEP]"]
        self.unk_token_id = self.tokenizer.vocab["[UNK]"]
        self.max_seq_length = max_seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = get_config()
        # cfg.model.backbone = 'checkpoint/bros-vn-uncased'
        self.net = get_model(cfg)
        load_model_weight(self.net, cfg.pretrained_model_file)
        # net.to("cuda:1")
        self.net.to(self.device)
        self.net.eval()
        # backbone_type = "bros"
        self.eval_kwargs = get_eval_kwargs_spade('datasets/funsd_spade', self.max_seq_length)
    
    def create_input_bros(self, img, boxes, texts):
        # img [H, W, C]
        # boxes [1, 8]
        width, height = img.shape[:2]
        input_ids = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        bbox = np.zeros((self.max_seq_length, 8), dtype=np.float32)
        attention_mask = np.zeros(self.max_seq_length, dtype=int)
        are_box_first_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)
        
        list_tokens = []
        list_bbs = []
        box2token_span_map = []
        box_to_token_indices = []
        cum_token_idx = 0
        cls_bbs = [0.0] * 8
        for word_idx, (bb, text) in enumerate(zip(boxes, texts)):
            this_box_token_indices = []
            bb = np.array(bb).reshape(-1, 2).tolist()
            tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
            if len(tokens) == 0:
                tokens.append(self.unk_token_id)
            if len(list_tokens) + len(tokens) > self.max_seq_length - 2:
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
        list_tokens = ([self.cls_token_id] + list_tokens[: self.max_seq_length - 2] + [self.sep_token_id])
        if len(list_bbs) == 0:
            # when len(words_info) == 0
            list_bbs = [cls_bbs] + [sep_bbs]
        else:
            list_bbs = [cls_bbs] + list_bbs[:self.max_seq_length - 2] + [sep_bbs]

        len_list_tokens = len(list_tokens)
        input_ids[:len_list_tokens] = list_tokens
        attention_mask[:len_list_tokens] = 1
        bbox[:len_list_tokens, :] = list_bbs
        # normalize box
        bbox[:, [0, 2, 4, 6]] = bbox[:, [0, 2, 4, 6]] / width
        bbox[:, [1, 3, 5, 7]] = bbox[:, [1, 3, 5, 7]] / height
        st_indices = [indices[0] for indices in box_to_token_indices if indices[0] < self.max_seq_length]
        are_box_first_tokens[st_indices] = True


        input_ids = torch.from_numpy(input_ids).unsqueeze(0).to(self.device)
        bbox = torch.from_numpy(bbox).unsqueeze(0).to(self.device)
        attention_mask = torch.from_numpy(attention_mask).unsqueeze(0).to(self.device)
        are_box_first_tokens = torch.from_numpy(are_box_first_tokens).unsqueeze(0).to(self.device)
        return_dict = {
                "input_ids": input_ids,
                "bbox": bbox,
                "attention_mask": attention_mask,
                "are_box_first_tokens": are_box_first_tokens,
                "box_to_token_indices": box_to_token_indices
                # "itc_labels": itc_labels,
                # "stc_labels": stc_labels,
                }
        return return_dict

    def key_extract(self, input_img):
        img, boxes, texts = get_ocr_result(filepath=input_img, link_api='...', token=None)
        input_data_item = self.create_input_bros(img, boxes, texts)
        # print(input_data_item['input_ids'].size())
        # print(input_data_item['bbox'].size())
        # print(input_data_item['attention_mask'].size())
        # print(input_data_item['are_box_first_tokens'].size())
        with torch.no_grad():
            head_outputs = self.net(input_data_item)
        
        class_names = self.eval_kwargs['class_names']
        dummy_idx = self.eval_kwargs['dummy_idx']
        itc_outputs = head_outputs['itc_outputs']
        stc_outputs = head_outputs['stc_outputs']
        # el_outputs = head_outputs['el_outputs']
        pr_itc_labels = torch.argmax(itc_outputs, -1)
        pr_stc_labels = torch.argmax(stc_outputs, -1)
        # pr_el_labels = torch.argmax(el_outputs, -1)
        # print('itc out', pr_itc_labels)
        # print('stc out', pr_stc_labels)
        # print('el out', pr_el_labels)
        # gt_itc_labels = input_data_item['itc_labels']
        are_box_first_tokens = input_data_item['are_box_first_tokens']
        attention_mask = input_data_item['attention_mask']
        # gt_stc_labels = input_data_item['stc_labels']
        box_to_token_indices = input_data_item['box_to_token_indices']
        # box_to_token_indices = [[v.cpu().numpy().tolist()[0] for v in u] for u in box_to_token_indices]
        # print('box token', are_box_first_tokens)
        # print('attn mask', attention_mask)
        bsz = pr_itc_labels.shape[0]
        
        list_results = [] 
        for example_idx in range(bsz):
            results = []
            # gt_first_words = parse_initial_words(gt_itc_labels[example_idx], are_box_first_tokens[example_idx], class_names)
            # gt_class_words = parse_subsequent_words(gt_stc_labels[example_idx], attention_mask[example_idx], gt_first_words, dummy_idx)

            pr_init_words = parse_initial_words(pr_itc_labels[example_idx], are_box_first_tokens[example_idx], class_names)
            
            pr_class_words = parse_subsequent_words(
                pr_stc_labels[example_idx], attention_mask[example_idx], pr_init_words, dummy_idx)
            print('str_rs', pr_class_words)
            # pr_relations = parse_relations(pr_el_labels[example_idx], are_box_first_tokens[example_idx], dummy_idx)
            # pr_relations = sorted(pr_relations)
            
            results = postprocess_keypair(pr_class_words, pr_relations, box_to_token_indices, texts, boxes)
            for result in results:
                key_boxes = result['key box']
                value_boxes = result['value box']
                key_point = ((key_boxes[0] + key_boxes[2]) // 2, (key_boxes[1] + key_boxes[3]) // 2)
                value_point = ((value_boxes[0] + value_boxes[2]) // 2, (value_boxes[1] + value_boxes[3]) // 2)
                img = cv2.rectangle(img, (key_boxes[0], key_boxes[1]), (key_boxes[2], key_boxes[3]), (0, 255, 0), 2)
                img = cv2.rectangle(img, (value_boxes[0], value_boxes[1]), (value_boxes[2], value_boxes[3]), (255, 0, 0), 2)
                img = cv2.line(img, key_point, value_point, (0, 0, 255), 1)
            list_results.append(results)
        # cv2.imwrite('check.jpg', img)
        return list_results[0], img
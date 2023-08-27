import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
def postprocess_keypair(pr_class_words, box_to_token_indices, texts, boxes):
    box_to_token_indices_keys = {box[0]: box for box in box_to_token_indices}
    str_rs_keys = pr_class_words[1]
    str_rs_values = pr_class_words[2]
    

    str_rs_keys_dict = {key[0]: key for key in str_rs_keys}
    str_rs_values_dict = {value[0]: value for value in str_rs_values}

    result_key = []
    result_value = []

    result_key_text = []
    result_value_text = []
    for value in str_rs_keys_dict:
        list_idx = []
        for span in str_rs_keys_dict[value]:
            if span in box_to_token_indices_keys.keys():
                box_token = box_to_token_indices_keys[span]
                key_idx = box_to_token_indices.index(box_token)
                list_idx.append(key_idx)
        key_text = [texts[idx] for idx in list_idx]
        key_text = ' '.join(key_text)
        result_key.append(list_idx)
        result_key_text.append(key_text)


    for value in str_rs_values_dict:
        list_idx = []
        for span in str_rs_values_dict[value]:
            if span in box_to_token_indices_keys.keys():
                box_token = box_to_token_indices_keys[span]
                key_idx = box_to_token_indices.index(box_token)
                list_idx.append(key_idx)
        value_text = [texts[idx] for idx in list_idx]
        value_text = ' '.join(value_text)

        result_value_text.append(value_text)
        result_value.append(list_idx)
    return result_key_text, result_value_text




    # for key_spans in str_rs_keys:
    #     for key_span in key_spans:
    #         if key_span in box_to_token_indices_keys.keys():
    #             box_token = box_to_token_indices_keys[key_span]
    #             key_idx = box_to_token_indices.index(box_token)
    #             list_key_idx.append(key_idx)
    #     if key_spans in str_rs_keys:
    #         str_rs_keys.remove(key_spans)
    #     key_pair_text = [texts[idx] for idx in list_key_idx]
    #     key_pair_boxes = [boxes[idx] for idx in list_key_idx]
    #     key_pair_boxes = merge_cells(key_pair_boxes)


    # for value_spans in str_rs_values:
    #     for value_span in value_spans:
    #         if value_span in box_to_token_indices_keys.keys():
    #             box_token = box_to_token_indices_keys[value_span]
    #             value_idx = box_to_token_indices.index(box_token)
    #             list_value_idx.append(value_idx)
    #     if value_spans in str_rs_values:
    #         str_rs_values.remove(value_spans)
    #     value_pair_text = [texts[idx] for idx in list_value_idx]
    #     value_pair_boxes = [boxes[idx] for idx in list_value_idx]
    #     value_pair_boxes = merge_cells(value_pair_boxes)

    #     # key_pair_text = ' '.join(key_pair_text)
    #     # value_pair_text = ' '.join(value_pair_text)


    #     result = dict()
    #     result['key'] = key_pair_text
    #     result['value'] = value_pair_text
    #     result['key box'] = key_pair_boxes
    #     result['value box'] = value_pair_boxes

    #     results.append(result)
    # print(results)
    # print('relations', pr_relations)
    # for pairs in pr_relations:
    #     if pairs[0] in str_rs_values_dict.keys() and pairs[1] in str_rs_keys_dict.keys():
    #         key_pair = pairs[1]
    #         value_pair = pairs[0]
    #     else:
    #         key_pair = pairs[0]
    #         value_pair = pairs[1]
    #     if key_pair not in str_rs_keys_dict.keys() or value_pair not in str_rs_values_dict.keys():
    #         continue
        
    #     list_key_idx = []
    #     list_value_idx = []
        
    #     key_spans = str_rs_keys_dict[key_pair]
    #     for key_span in key_spans:
    #         if key_span in box_to_token_indices_keys.keys():
    #             box_token = box_to_token_indices_keys[key_span]
    #             key_idx = box_to_token_indices.index(box_token)
    #             list_key_idx.append(key_idx)
    #     if key_spans in str_rs_keys:
    #         str_rs_keys.remove(key_spans)
    #     if value_pair not in str_rs_values_dict.keys():
    #         continue
    #     value_spans = str_rs_values_dict[value_pair]
    #     for value_span in value_spans:
    #         if value_span in box_to_token_indices_keys.keys():
    #             box_token = box_to_token_indices_keys[value_span]
    #             value_idx = box_to_token_indices.index(box_token)
    #             list_value_idx.append(value_idx)
    #     key_pair_text = [texts[idx] for idx in list_key_idx]
    #     value_pair_text = [texts[idx] for idx in list_value_idx]
    #     key_pair_boxes = [boxes[idx] for idx in list_key_idx]
    #     value_pair_boxes = [boxes[idx] for idx in list_value_idx]
    #     key_pair_boxes = merge_cells(key_pair_boxes)
    #     value_pair_boxes = merge_cells(value_pair_boxes)

    #     if value_spans in str_rs_values:
    #         str_rs_values.remove(value_spans)

    #     key_pair_text = ' '.join(key_pair_text)
    #     value_pair_text = ' '.join(value_pair_text)

    #     result = dict()
    #     result['key'] = key_pair_text
    #     result['value'] = value_pair_text
    #     result['key box'] = key_pair_boxes
    #     result['value box'] = value_pair_boxes
    #     results.append(result)
def merge_cells(cells):
    cells = [np.array(cell).reshape(-1, 2).tolist() for cell in cells]
    polygons = []
    for cell in cells:
        polygons.append(Polygon(cell))
    boundary = gpd.GeoSeries(unary_union(polygons))
    boxes = boundary.bounds
    boxes = boxes[['minx', 'miny', 'maxx', 'maxy']].values
    boxes = boxes.tolist()
    boxes = [list(map(int, box)) for box in boxes]
    return boxes[0]
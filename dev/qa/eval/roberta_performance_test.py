import torch
import numpy as np 
import transformers 
from sklearn.metrics import classification_report
import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.trainer_utils import get_last_checkpoint
from transformers.pipelines import pipeline
import pdb
import sys 
sys.path.append("../utils")
import qa_metric
import spacy
import os
from tqdm import tqdm
from keybert import KeyBERT
os.environ["CUDA_LAUNCH_BLOCKING"]="7"
nlp = spacy.load("en_core_web_sm")
# tok_name = "../../tok"
# model_name = "./output_dir/checkpoint-14800"
# tok_name = model_name = "deepset/roberta-base-squad2"
tok_name = get_last_checkpoint("./deepsettosed_kf")
model_name = get_last_checkpoint("./deepsettosed_kf")
file_path = "../../data/sed/test.json"
tokenizer = AutoTokenizer.from_pretrained(tok_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model.cuda()
model.eval()
kw_extractor = KeyBERT('distilbert-base-nli-mean-tokens')
def load_json_file(file_path):
    dataset_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    for p in dataset["data"]:
        dataset_list.append(p)
    return dataset_list

def find_sep_position(id_lists, sep_id):
    sep_pos = -1
    for i in range(len(id_lists)):
        if id_lists[i] == sep_id:
            sep_pos = i
            break 
    return sep_pos

def is_valid_start_end(start_index_cand, end_index_cand, id_lists, tokenizer):
    sep_pos = find_sep_position(id_lists, tokenizer.sep_token_id)
    is_valid = False 
    ans_text = ""
    # if start_index_cand<=end_index_cand and start_index_cand>sep_pos:
    if start_index_cand<=end_index_cand and end_index_cand>sep_pos+1:
        if start_index_cand<=sep_pos:
            start_index_cand = sep_pos + 2
        is_valid = True 
        ans_text = tokenizer.decode(id_lists[start_index_cand:end_index_cand+1])
    return is_valid, ans_text

def demo(question="Who was Jim Henson ?", text="Jim Henson was a nice puppet"):
    # question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    question = repharse_question(question)
    input_ids = tokenizer.encode(question, text)
    start_scores, end_scores = model(torch.tensor([input_ids],device="cuda"))
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    is_valid, ans_text = is_valid_start_end(start_index, end_index, input_ids, tokenizer)
    print(is_valid)
    print(ans_text)

def predict_doc(question, doc_text):
    input_ids = tokenizer.encode(question, doc_text, truncation=True, max_length=512)
#     print(input_ids)
    start_scores, end_scores = model(torch.tensor([input_ids], device="cuda"), return_dict=False)
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    has_ans, ans_text = is_valid_start_end(start_index, end_index, input_ids, tokenizer)
    return has_ans, ans_text

def load_test_doc_data(test_file_path):
    json_items = load_json_file(test_file_path)
    return json_items

def test_doc_evaluate(test_file_path):
    y_label = []
    y_hat_list = []
    dataset = load_test_doc_data(test_file_path)
    for para in dataset:
        for p in para["paragraphs"]:
            doc_text = p["context"]
            for q in p["qas"]:
                ques = q["question"]
                # y_hat, y_prob, y_sent_hat = predict_ques_and_doc_v1(doc, ques, model, model_v1)
                has_ans, ans_text = predict_doc(ques, doc_text)
                y_hat = 1 if has_ans else 0
                y_label.append(int(q["is_impossible"]))
                y_hat_list.append(y_hat)
    t = classification_report(y_label, y_hat_list, labels=[0, 1])

# def repharse_question(ques):
#     doc = nlp(ques)
#     ques = " ".join([t.text for t in doc])
#     map_dict = {"you":"I", "your":"my"}
#     ques_token_list = ques.split()
#     for i in range(len(ques_token_list)):
#         _token = ques_token_list[i]
#         _lower_token = _token.lower()
#         if _lower_token in map_dict:
#             sub_token = map_dict[_lower_token]
#             ques_token_list[i] = sub_token
#     return " ".join(ques_token_list)

# def repharse_question(ques):
#     doc = nlp(ques)
#     return " ".join([t.text for t in doc.ents])

def repharse_question(ques):
#     kw_extractor = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = [x[0] for x in kw_extractor.extract_keywords(ques,  keyphrase_ngram_range=(1, 1), top_n=2, stop_words=None)]
    return " ".join(keywords)


def test_doc_evaluate_with_overlap_f1(test_file_path, verbose=False):
    detail_output_path = test_file_path+"_roberta_qa_prediction_detail_only_writing_01_12.txt"
    y_label = []
    y_hat_list = []
    dataset = load_test_doc_data(test_file_path)
    overlap_f1_list = []
    gold_ans_list = []
    predict_ans_list = []
    doc_text_list = []
    doc_text_id_list = []
    ques_list = []
    for para in tqdm(dataset):
        for p in para["paragraphs"]:
            doc_text = p["context"]
            for q in p["qas"]:
                ques = q["question"]
                gold_ans = q["answers"][0]["text"] if q["answers"] else ""
                sample_id = q["id"]
                # y_hat, y_prob, y_sent_hat = predict_ques_and_doc_v1(doc, ques, model, model_v1)
                trans_ques = repharse_question(ques)
#                 trans_ques = ques
                has_ans, ans_text = predict_doc(trans_ques, doc_text)
                overlap_f1 = qa_metric.compute_f1(gold_ans, ans_text)
                gold_ans_list.append(gold_ans)
                predict_ans_list.append(ans_text)
                overlap_f1_list.append(overlap_f1)
                doc_text_list.append(doc_text)
                doc_text_id_list.append(sample_id)
                ques_list.append(ques)
                y_hat = 1 if has_ans else 0
                y_label.append(int(not q["is_impossible"]))
                y_hat_list.append(y_hat)
    t = classification_report(y_label, y_hat_list, labels=[0, 1])
    overlap_f1 = "Overlap f1 {:.4f}".format(np.mean(overlap_f1_list))
    print(t)
    print(overlap_f1)
    # output detail
    if verbose:
        with open(detail_output_path, "w", encoding="utf-8") as f:
            for i in range(len(doc_text_list)):
                split_line = "-"*30+"\n\n"
                _id = doc_text_id_list[i]
                _doc_text = doc_text_list[i]
                _gold_ans = gold_ans_list[i]
                _pred_ans = predict_ans_list[i]
                _gold_have_ans_label = y_label[i]
                _pred_have_ans_label = y_hat_list[i]
                _overlap_f1 = overlap_f1_list[i]
                _ques = ques_list[i]
                _id_line = "ID:{}\n".format(_id)
                _ques_line = "-Question:\n  {}\n".format(_ques)
                _doc_text_line = "-Text:\n  {}\n".format(_doc_text)
                _label_line = "-Label: T:{}   P:{}\n".format(_gold_have_ans_label, _pred_have_ans_label)
                _gold_ans_line = "{}\n  {}\n".format("-gold ans:",_gold_ans)
                _pred_ans_line = "{}\n  {}\n".format("-pred ans:",_pred_ans)
                _overlap_f1_line = "{}: {}\n".format("-overlap f1",_overlap_f1)
                f.write("".join([_id_line,_ques_line, _doc_text_line, _label_line, _gold_ans_line, _pred_ans_line, _overlap_f1_line, split_line]))
            f.write(t+"\n")
            f.write(overlap_f1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path")
    test_doc_evaluate_with_overlap_f1(file_path, True)

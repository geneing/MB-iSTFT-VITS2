import argparse
import text
from utils import load_filepaths_and_text
from tqdm import trange, tqdm
import torch
from transformers import DebertaV2Model, DebertaV2Tokenizer


MODEL_NAME = 'microsoft/deberta-v3-small'
model = DebertaV2Model.from_pretrained(MODEL_NAME)
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

def tokenize_bart_enc(text, wav_path):
    encoded_input = tokenizer(text.strip(), return_tensors='pt')
    output = model(**encoded_input, output_hidden_states=True)
    bart_out = torch.cat(output["hidden_states"][-3:-2], -1)[0]

    converted=tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])

    remove_whitespace = lambda c: (" " + c[1:]) if ((len(c)>0) and c[0] == "▁") else c

    tokens_nospecials = [ (c if not(c in tokenizer.all_special_tokens) else "") for c in converted ]

    token_list = list(map(remove_whitespace, tokens_nospecials))
    if len(token_list[0])==0:
        token_list[1] = token_list[1].strip()
    repl_list = [len(token) for token in token_list]
    
    repeats = torch.tensor(repl_list, device=bart_out.device)
    # print(f"{bart_out.shape=}")
    # print(f"{repeats.shape=}")
    repeated_tensor = torch.repeat_interleave(bart_out, repeats, dim=0)
    assert repeated_tensor.shape[0] == sum(repl_list), f"{text}: {bart_out.shape=}, {repeated_tensor.shape=}, {sum(repl_list)=}"
    # print(f"{repeated_tensor.shape=}, {sum(repl_list)=}")

    bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")
    torch.save(repeated_tensor, bert_path)

    return # (repeated_tensor, repl_list)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=1, type=int)
  parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
  parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners4"])
  parser.add_argument("--bert_save", action="store_true", 
                    help="save bert output")
 
  args = parser.parse_args()
    

  for filelist in args.filelists:
    print("START:", filelist)
    filepaths_and_text = load_filepaths_and_text(filelist)
    for i in trange(len(filepaths_and_text)):
      original_text = filepaths_and_text[i][args.text_index]
      cleaned_text = text._clean_text(original_text, args.text_cleaners)
      filepaths_and_text[i][args.text_index] = cleaned_text
      if(args.bert_save):
        tokenize_bart_enc(cleaned_text, filepaths_and_text[i][0])
      

    new_filelist = filelist + "." + args.out_extension
    with open(new_filelist, "w", encoding="utf-8") as f:
      f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])

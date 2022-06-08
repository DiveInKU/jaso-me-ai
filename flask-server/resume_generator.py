import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast


def generate_ai_text(user_text):
    model = torch.load("./kogpt/saved-model/pretrained.pt")
    model.eval()
    tokenizer = torch.load("./kogpt/saved-model/pretrained_tokenizer.pt")
    text = user_text
    input_ids = tokenizer.encode(text)
    gen_ids = model.generate(torch.tensor([input_ids]),
                             max_length=80,
                             repetition_penalty=2.0,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             bos_token_id=tokenizer.bos_token_id,
                             use_cache=True)
    generated = tokenizer.decode(gen_ids[0, :].tolist())
    print(generated)
    generated_list = generated.split("\n")
    print(generated_list)
    three_sentences = "\n".join(generated_list[0:len(generated_list) - 1])
    print(three_sentences)
    return three_sentences


def save_model():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>',
                                                        unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    # tokenizer.tokenize("안녕하세요. 한국어 GPT-2 입니다.😤:)l^o")
    torch.save(tokenizer, "./kogpt/saved-model/pretrained_tokenizer.pt")
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    torch.save(model, "./kogpt/saved-model/pretrained.pt")


if __name__ == '__main__':
    save_model()
    generate_ai_text('컴퓨터')

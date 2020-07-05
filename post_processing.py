import torch
import torch.nn.functional as F


# Cf) : https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def get_response(model, tokenizer, context):
    num_generate = 1
    max_length = 128
    temperature = 0.7
    top_k = 40
    top_p = 0.9
    context_ids = tokenizer.encode(context)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_tensor = torch.tensor(context_ids, dtype=torch.long, device=device)
    context_tensor = context_tensor.unsqueeze(0).repeat(num_generate, 1)
    samples = context_tensor
    with torch.no_grad():
        while True:
            inputs = {'input_ids': samples}
            outputs = model(**inputs)
            prob_next = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            prob_filtered = top_k_top_p_filtering(prob_next, top_k=top_k, top_p=top_p)
            if temperature == 0.0:
                next_word = torch.argmax(prob_filtered, dim=-1).unsqueeze(-1)
            else:
                next_word = torch.multinomial(F.softmax(prob_filtered, dim=-1), num_samples=1)
            samples = torch.cat((samples, next_word), dim=1)
            if (samples[:, len(context_ids):] == tokenizer.eos_token_id).any(dim=1).all():
                break
            if samples.shape[1] - len(context_ids) >= max_length:
                break
    samples = samples[:, len(context_ids):].tolist()
    candidates = []
    for sample in samples:
        response = tokenizer.decode(sample, clean_up_tokenization_spaces=True)
        response = response[: response.find(tokenizer.eos_token)]
        candidates.append(response)
    return candidates


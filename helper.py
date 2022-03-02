import matplotlib.pyplot as plt 
import torch
from einops import rearrange
import numpy as np


class VizHelper:
    def __init__(self, model, tokenizer, raw_data, proc_data):
        self.model = model
        self.raw_data = raw_data
        self.proc_data = proc_data
        self.tokenizer = tokenizer
        
    def _forward(self, idx, no_grad=True):
        self.model.eval()
        item = self.proc_data[[idx]]
        
        if no_grad:
            with torch.no_grad():
                outputs = self.model(**item, output_attentions=True, output_hidden_states=True)
        else:
            outputs = self.model(**item, output_attentions=True, output_hidden_states=True)
        return outputs
    
    def get_hta(self, idx, use_inputs=True):
        """Hidden Token Attribution on input embeddings"""    
    
        # get input embeddings
        item = self.proc_data[[idx]]
        input_len = item["attention_mask"].sum()
        
        if use_inputs:
        
            embedding_matrix = self.model.bert.embeddings.word_embeddings.weight
            vocab_size = embedding_matrix.shape[0]
            onehot = torch.nn.functional.one_hot(item["input_ids"][0], vocab_size).float()
            embeddings = torch.matmul(onehot, embedding_matrix)
            embeddings = rearrange(embeddings, "s h -> () s h")
            
            outputs = self.model(
                inputs_embeds=embeddings,
                attention_mask=item["attention_mask"],
                token_type_ids=item["token_type_ids"],
            )
            
        else:
            outputs = self._forward(idx)
            embeddings = outputs.hidden_states[0]

        grad = torch.autograd.grad(
            outputs.logits,
            embeddings,
            grad_outputs=torch.ones_like(outputs.logits)
        )[0]
        
        grad = torch.norm(grad, dim=2)[0]
        grad = grad / torch.sum(grad)
        return grad[: input_len]
    
    def get_kernel_shap(self, idx):
        raise NotImplementedError()
        
    def get_deep_shap(self, idx):
        raise NotImplementedError()
        
    def get_soc(self, idx):
        raise NotImplementedError()
        
    def _get_attentions(self, idx, head, layer):
        outputs = self._forward(idx)
        attentions = torch.cat(outputs.attentions)
        attentions = rearrange(attentions, "l h s1 s2 -> h l s1 s2")
        
        item = self.proc_data[idx]
        input_len = item["attention_mask"].sum()
        attentions = attentions[head, layer, :input_len, :input_len]
        return attentions
    
    def show_attention(self, idx, head, **kwargs):
        layer = kwargs.get("layer", 10)
        fontsize = kwargs.get("fontsize", 14)
        figsize = kwargs.get("figsize", (8,8))

        attentions = self._get_attentions(idx, head, layer)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(attentions)
        
        item = self.proc_data[idx]
        input_len = item["attention_mask"].sum()
        ticks = self.tokenizer.batch_decode(item["input_ids"][:input_len])

        ax.set_xticks(np.arange(input_len))
        ax.set_yticks(np.arange(input_len))
        ax.set_xticklabels(ticks, rotation=90, fontsize=fontsize)
        ax.set_yticklabels(ticks, fontsize=14)
        
        fig.tight_layout()      
    
    
    def _get_effective_attention(self, idx, head, layer):
        item = self.proc_data[idx]
        input_len = item["attention_mask"].sum()
        
        outputs = self._forward(idx)
        values = [v.detach() for v in outputs.value]
        attentions = [a.detach()[0] for a in outputs.attentions]
        
        #effective_attention_map = []
        #for current_layer in range(12):
        U, S, V = torch.Tensor.svd(values[layer], some=False, compute_uv=True)
        bound = torch.finfo(S.dtype).eps * max(U.shape[1], V.shape[1])

        greater_than_bound = S > bound

        basis_start_index = torch.max(torch.sum(greater_than_bound, dtype=int, axis=2))
        null_space = U[:, :, :, basis_start_index:]

        B = torch.matmul(attentions[layer], null_space)
        transpose_B = torch.transpose(B, -1, -2)
        projection_attention = torch.matmul(null_space, transpose_B)
        projection_attention = torch.transpose(projection_attention, -1, -2)
        effective_attention = torch.sub(attentions[layer], projection_attention)
        
        # select head in effective attention
        effective_attention = effective_attention[0][head, :input_len, :input_len]
        return effective_attention
        
    
    def show_effective_attention(self, idx, head, **kwargs):
        layer = kwargs.get("layer", 10)
        fontsize = kwargs.get("fontsize", 14)
        
        item = self.proc_data[idx]
        input_len = item["attention_mask"].sum()
        
        effective_attention = self._get_effective_attention(idx, head, layer) 
            
        fig, ax = plt.subplots(figsize=(11,11))
        ax.imshow(effective_attention)
        ticks = self.tokenizer.batch_decode(item["input_ids"][:input_len])

        ax.set_xticks(np.arange(input_len))
        ax.set_yticks(np.arange(input_len))
        ax.set_xticklabels(ticks, rotation=90, fontsize=fontsize)
        ax.set_yticklabels(ticks, fontsize=14)
    
    
    def compare_attentions(self, idx, head, layer, **kwargs):
        fontsize = kwargs.get("fontsize", 14)
        
        effective_attentions = self._get_effective_attention(idx, head, layer) 
        attentions = self._get_attentions(idx, head, layer)
        
        #attentions = -(1 / attentions).log()
        #effective_attentions = -(1 / effective_attentions).log()
        
        fig, ax = plt.subplots(ncols=2, figsize=(16,8), sharey=True)
        ax1, ax2 = ax
        ax1.imshow(attentions)
        ax2.imshow(effective_attentions)
        
        item = self.proc_data[idx]
        input_len = item["attention_mask"].sum()

        ticks = self.tokenizer.batch_decode(item["input_ids"][:input_len])
        
        list(map(lambda x: x.set_xticks(np.arange(input_len)), ax))
        list(map(lambda x: x.set_xticklabels(ticks, rotation=90, fontsize=fontsize), ax))
        
        ax1.set_yticks(np.arange(input_len))
        ax1.set_yticklabels(ticks, fontsize=fontsize)
        
        #ax1.set_xticks()
        #ax1.set_xticklabels(ticks, rotation=90, fontsize=fontsize)
        
        fig.tight_layout()
        
        return fig
    
    def classify(self, idx):
        
        print("IDX:", idx)
        print("Text:", self.raw_data[idx]["text"])
        
        outputs = self._forward(idx) 
        logits = outputs.logits
        
        print("True label:", self.raw_data[idx]["label"])
        print("Prediction:", logits.argmax(-1).item())
        
    def compute_table(self, idx):
        return self.get_hta(idx)
        
    
        
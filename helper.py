from multiprocessing.sharedctypes import Value
from unicodedata import normalize
import matplotlib.pyplot as plt 
import torch
from einops import rearrange
import numpy as np
import pandas as pd
from tqdm import tqdm
from captum.attr import KernelShap

class VizHelper:
    def __init__(self, model, tokenizer, raw_data, proc_data):
        self.model = model
        self.raw_data = raw_data
        self.proc_data = proc_data
        self.tokenizer = tokenizer

    
    def _get_input_embeds(self, idx):
        item = self._get_item(idx)

        embedding_matrix = self.model.bert.embeddings.word_embeddings.weight
        vocab_size = embedding_matrix.shape[0]
        onehot = torch.nn.functional.one_hot(item["input_ids"][0], vocab_size).float()
        embeddings = torch.matmul(onehot, embedding_matrix)
        embeddings = rearrange(embeddings, "s h -> () s h")
        return embeddings
    
    def _forward(self, idx, no_grad=True, model=None, use_inputs=False):
        model = model if model else self.model
        model.eval()
        item = self._get_item(idx)

        def _foward_pass(use_inputs=False):
            
            if use_inputs:
                embeddings = self._get_input_embeds(idx)
                outputs = self.model(
                    inputs_embeds=embeddings,
                    attention_mask=item["attention_mask"],
                    token_type_ids=item["token_type_ids"],
                    output_hidden_states=True
                )

                return outputs, embeddings

            else:
                outputs = model(**item, output_attentions=True, output_hidden_states=True)
                return outputs


        if no_grad:
            with torch.no_grad():
                outputs = _foward_pass(use_inputs)
        else:
            outputs = _foward_pass(use_inputs)
        
        return outputs

    
    def _get_item(self, idx):
        if isinstance(idx, int):
            return self.proc_data[[idx]]
        elif isinstance(idx, str):
            return self.tokenizer(idx)
        else:
            raise ValueError(f"{idx} is of unknown type")

    def get_hta(self, idx, **kwargs):
        layer = kwargs.get("layer", 10)

        item = self._get_item(idx)
        input_len = item["attention_mask"].sum()

        embedding_matrix = self.model.bert.embeddings.word_embeddings.weight
        vocab_size = embedding_matrix.shape[0]
        onehot = torch.nn.functional.one_hot(item["input_ids"][0], vocab_size).float()
        embeddings = torch.matmul(onehot, embedding_matrix)
        embeddings = rearrange(embeddings, "s h -> () s h")
        
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=item["attention_mask"],
            token_type_ids=item["token_type_ids"],
            output_hidden_states=True
        )

        # get hidden states of a specific layer
        hidden_states = outputs.hidden_states[layer+1][0]

        grads = list()
        pbar = tqdm(total=input_len.item())
        for hs in hidden_states[:input_len]:

            grad = torch.autograd.grad(
                hs.unsqueeze(0),
                embeddings,
                grad_outputs=torch.ones_like(hs.unsqueeze(0)),
                retain_graph=True
            )[0]
            
            grads.append(grad)
            pbar.update()

        pbar.close()

        grads = torch.cat(grads) # (input_len, max_len, hidden_size)
        grads = grads[:, :input_len, :]
        
        # compute per-token HTAs
        htas = list()
        for g in grads:
            g = g.norm(dim=-1)
            g /= g.sum()
            htas.append(g)
            
        htas = torch.stack(htas)
        return htas


    # def get_hta_old(self, idx, use_inputs=True):
    #     """Hidden Token Attribution on input embeddings"""    
    
    #     # get input embeddings
    #     item = self.proc_data[[idx]]
    #     input_len = item["attention_mask"].sum()
        
    #     if use_inputs:
        
    #         embedding_matrix = self.model.bert.embeddings.word_embeddings.weight
    #         vocab_size = embedding_matrix.shape[0]
    #         onehot = torch.nn.functional.one_hot(item["input_ids"][0], vocab_size).float()
    #         embeddings = torch.matmul(onehot, embedding_matrix)
    #         embeddings = rearrange(embeddings, "s h -> () s h")
            
    #         outputs = self.model(
    #             inputs_embeds=embeddings,
    #             attention_mask=item["attention_mask"],
    #             token_type_ids=item["token_type_ids"],
    #         )
            
    #     else:
    #         outputs = self._forward(idx, no_grad=False)
    #         embeddings = outputs.hidden_states[0]

    #     grad = torch.autograd.grad(
    #         outputs.logits,
    #         embeddings,
    #         #grad_outputs=torch.ones_like(outputs.logits)
    #     )[0]
        
    #     grad = torch.norm(grad, dim=2)[0]
    #     grad = grad / torch.sum(grad)
    #     return grad[: input_len]
    
    def get_kernel_shap(self, idx, target=1):
        item = self._get_item(idx)
        input_len = item["attention_mask"].sum().item()

        def func(input_embeds):
            outputs = self.model(
                inputs_embeds=input_embeds,
                attention_mask=item["attention_mask"],
                token_type_ids=item["token_type_ids"],
            )
            scores = outputs.logits.softmax(-1)[0]
            return scores[target]

        ks = KernelShap(func)
        inputs = self._get_input_embeds(idx)
        fmask = list()
        for i in range(inputs.shape[1]):
            fmask.append(torch.full((inputs.shape[-1],), i))
        fmask = torch.stack(fmask).unsqueeze(0)

        attr = ks.attribute(inputs, n_samples=200, feature_mask=fmask, show_progress=True)
        attr = attr[0, :input_len, 0]

        return attr


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
    
    def _get_effective_attention(self, idx, head, layer, effective_model):
        item = self.proc_data[idx]
        input_len = item["attention_mask"].sum()
        
        outputs = self._forward(idx, model=effective_model)
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
        effective_model = kwargs.get("effective_model", self.model)
        
        effective_attentions = self._get_effective_attention(idx, head, layer, effective_model)
        attentions = self._get_attentions(idx, head, layer)
        hta = self.get_hta(idx, layer=layer)
        
        fig, ax = plt.subplots(ncols=3, figsize=(18,8), sharey=True)
        ax1, ax2, ax3 = ax
        ax1.imshow(attentions)
        ax2.imshow(effective_attentions)
        ax3.imshow(hta)

        item = self._get_item(idx)
        input_len = item["attention_mask"].sum().item()

        ticks = self.tokenizer.batch_decode(item["input_ids"][0][:input_len])
        
        list(map(lambda x: x.set_xticks(np.arange(input_len)), ax))
        list(map(lambda x: x.set_xticklabels(ticks, rotation=90, fontsize=fontsize), ax))
        
        ax1.set_yticks(np.arange(input_len))
        ax1.set_yticklabels(ticks, fontsize=fontsize)
        
        #ax1.set_xticks()
        #ax1.set_xticklabels(ticks, rotation=90, fontsize=fontsize)
        
        fig.tight_layout()
        
        return fig
    
    def get_gradient(self, idx, target=1):
        outputs, embeddings = self._forward(idx, use_inputs=True, no_grad=False)
        out = outputs.logits[0][target]

        item = self._get_item(idx)
        input_len = item["attention_mask"].sum().item()

        # compute loss
        scores = outputs.logits.softmax(-1)[0]
        loss = - torch.log(scores[target] / scores.exp().sum()) # cross entropy
        
        # compute gradients of loss wrt input embeddings
        grad = torch.autograd.grad(loss, embeddings)[0]
        grad = grad[:, :input_len, :]

        embeddings = embeddings[:, :input_len, :]

        prods = list()
        for g, e in zip(grad[0], embeddings[0]):
            r = torch.dot(-g, e)
            prods.append(r)

        grad_input = torch.tensor(prods)
        grad_input /= grad_input.norm(dim=-1, p=1) # l1 normalization

        normalized_grad = -grad[0].mean(-1) # avg over hidden size
        normalized_grad /= normalized_grad.norm(dim=-1, p=1) # normalize over tokens

        return grad_input, normalized_grad

    def classify(self, idx):
        
        print("IDX:", idx)
        print("Text:", self.raw_data[idx]["text"])
        
        outputs = self._forward(idx) 
        logits = outputs.logits
        
        print("True label:", self.raw_data[idx]["label"])
        print("Prediction:", logits.argmax(-1).item())
        
    def compute_table(self, idx, target=1):
        item = self.proc_data[idx]
        input_len = item["attention_mask"].sum()
        tokens = self.tokenizer.batch_decode(self.proc_data[idx]["input_ids"])[:input_len]
        
        # saliency methods
        grad_inputs, normalized_grad = self.get_gradient(idx, target=target)

        # shap
        k_shap = self.get_kernel_shap(idx, target=target)

        d = {
            "tokens": tokens,
            "G": normalized_grad,
            "GxI": grad_inputs,
            "KernelSHAP": k_shap,

        }
        table = pd.DataFrame(d).set_index("tokens").T
        
        return table
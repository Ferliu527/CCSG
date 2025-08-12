import torch
import numpy as np
import json
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class IntegratedGradients:
    def __init__(self, model, tokenizer, embedding_layer_name):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
        # Embedding Layer
        self.embedding_layer = self._get_embedding_layer(embedding_layer_name)
        
        self.embedding_gradients = None
        
        # hook
        self.hook_handles = []
        self._register_hooks()
    
    def _get_embedding_layer(self, layer_name):
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"embedding layer {layer_name} notfound")
    
    def _register_hooks(self):
        # get gradients
        def hook_fn(module, grad_in, grad_out):
            self.embedding_gradients = grad_in[0]
        
        handle = self.embedding_layer.register_backward_hook(hook_fn)
        self.hook_handles.append(handle)
    
    def _get_embeddings(self, input_ids):
        return self.embedding_layer(input_ids)
    
    def _generate_baseline(self, input_ids):
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        return self._get_embeddings(baseline_ids)
    
    def _interpolate(self, embeddings, baseline_embeddings, alpha):
        return baseline_embeddings + alpha * (embeddings - baseline_embeddings)
    
    def compute(self, text, target_class=None, steps=50):
        # tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # get embedding
        embeddings = self._get_embeddings(input_ids)
        baseline_embeddings = self._generate_baseline(input_ids)
        
        # init gradients
        integrated_grads = torch.zeros_like(embeddings)
        
        for i in range(steps):
            alpha = i / (steps - 1) if steps > 1 else 1.0
            interpolated_embeddings = self._interpolate(embeddings, baseline_embeddings, alpha)
            
            self.model.zero_grad()
            outputs = self.model(inputs_embeds=interpolated_embeddings, attention_mask=attention_mask)
            logits = outputs.logits
            
            if target_class is None:
                target_class = torch.argmax(logits, dim=1).item()

            loss = logits[0, target_class]
            loss.backward(retain_graph=True)
            
            if self.embedding_gradients is not None:
                integrated_grads += self.embedding_gradients / steps
        
        integrated_grads *= (embeddings - baseline_embeddings)
        
        # get word contribution
        token_contributions = torch.sum(torch.abs(integrated_grads), dim=2).squeeze()
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        
        # fliter filling token
        results = []
        for token, contribution in zip(tokens, token_contributions):
            if token != self.tokenizer.pad_token:
                results.append({
                    "token": token,
                    "contribution": contribution.item()
                })
        
        return results
    
    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()

def save_contributions(contributions, output_dir="contribution_output", filename="contributions.json"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, filename)
    
    # save as json
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(contributions, f, ensure_ascii=False, indent=4)
    
    print(f"word contributions saved: {file_path}")

# use example
if __name__ == "__main__":

    model_name = "embedding model"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    ig = IntegratedGradients(model, tokenizer)
    
    # example input
    text = "This movie is fantastic! The acting was superb and the plot was engaging."
    
    contributions = ig.compute(text)
    save_contributions(contributions)
    
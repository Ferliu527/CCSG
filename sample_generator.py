import os
import json
import random
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import wordnet as wn
import spacy

class SampleGenerator:
    def __init__(self, glove_path, spacy_model="en_core_web_sm", contribution_dir="contribution_output"):
        self.contribution_dir = contribution_dir
        
        # load spaCy
        print("Loading spaCy...")
        self.nlp = spacy.load(spacy_model)
        print("spaCy is ready")
        
        # load Glove
        print("Loading Glove...")
        self.glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
        print("Glove is ready")
        
        self.positive_dir = "positive_samples"
        self.negative_dir = "negative_samples"
        os.makedirs(self.positive_dir, exist_ok=True)
        os.makedirs(self.negative_dir, exist_ok=True)
        
        # keep nouns and verbs
        self.target_pos = {'N', 'V'}
    
    def is_target_pos(self, word):
        if not word:
            return False
            
        doc = self.nlp(word)
        if not doc:
            return False
            
        pos_tag = doc[0].tag_[0] 
        return pos_tag in self.target_pos
    
    def filter_by_pos(self, words):
        return [word for word in words if self.is_target_pos(word)]
    
    def load_contributions(self, filename="contributions.json"):
        # load word contribution
        file_path = os.path.join(self.contribution_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def find_most_contributing_word(self, contributions):
        if not contributions:
            return None
        return max(contributions, key=lambda x: x['contribution'])
    
    def get_similar_words(self, word, top_k=50):
        try:
            if word not in self.glove_model:
                print(f"'{word}' not found")
                return []
            
            similar_words = self.glove_model.most_similar(word, topn=top_k)
            similar_words = [word for word, _ in similar_words]
            
            filtered_words = self.filter_by_pos(similar_words)
            
            return filtered_words
        except Exception as e:
            print(e)
            return []
    
    def get_related_words_from_wordnet(self, word, candidate_words, m=5):
        # get synonyms and hyponymy
        if not candidate_words:
            return []

        doc = self.nlp(word)
        pos_tag = doc[0].tag_[0]
        
        wn_pos = wn.NOUN if pos_tag == 'N' else wn.VERB if pos_tag == 'V' else None
        if not wn_pos:
            return candidate_words[:m]
            
        original_synsets = wn.synsets(word, pos=wn_pos)
        if not original_synsets:
            return candidate_words[:m]  
        
        # compute similar
        word_similarities = []
        for candidate in candidate_words:
            try:
                cand_doc = self.nlp(candidate)
                cand_pos_tag = cand_doc[0].tag_[0]
                if (pos_tag == 'N' and cand_pos_tag != 'N') or (pos_tag == 'V' and cand_pos_tag != 'V'):
                    continue
                    
                cand_wn_pos = wn.NOUN if cand_pos_tag == 'N' else wn.VERB
                candidate_synsets = wn.synsets(candidate, pos=cand_wn_pos)
                if not candidate_synsets:
                    continue
                
                max_sim = 0
                for orig_syn in original_synsets:
                    for cand_syn in candidate_synsets:
                        sim = orig_syn.path_similarity(cand_syn)
                        if sim and sim > max_sim:
                            max_sim = sim
                
                if max_sim > 0:
                    word_similarities.append((candidate, max_sim))
            except Exception as e:
                print(e)
                continue
        
        word_similarities.sort(key=lambda x: x[1], reverse=True)
        result = [word for word, _ in word_similarities[:m]]
        
        if len(result) < m:
            remaining = [w for w in candidate_words if w not in result]
            result.extend(remaining[:m-len(result)])
            
        return result
    
    def generate_positive_samples(self, original_sentence, contributions, top_k=50, n=10, m=5):
        target_word_info = self.find_most_contributing_word(contributions)
        if not target_word_info:
            return []
        
        target_word = target_word_info['token']
        
        similar_words = self.get_similar_words(target_word, top_k)
        if not similar_words:
            return []
        
        if len(similar_words) > n:
            candidate_words = random.sample(similar_words, n)
        else:
            candidate_words = similar_words
        
        final_candidates = self.get_related_words_from_wordnet(target_word, candidate_words, m)
        
        # generate positive samples
        positive_samples = []
        for word in final_candidates:
            # word replace
            sample = original_sentence.replace(target_word, word)
            positive_samples.append({
                "original_word": target_word,
                "replaced_word": word,
                "original_pos": self.nlp(target_word)[0].tag_,
                "replaced_pos": self.nlp(word)[0].tag_,
                "sentence": sample
            })
        
        return positive_samples
    
    def generate_negative_samples(self, original_sentence, contributions, total_candidates=100, n=10, m=5):
        target_word_info = self.find_most_contributing_word(contributions)
        if not target_word_info:
            return []
        
        target_word = target_word_info['token']
        
        similar_words = self.get_similar_words(target_word, total_candidates)
        
        # words that are too dissimilar are not good, choose your rate
        start_idx = int(len(similar_words) * 0.6)
        end_idx = int(len(similar_words) * 0.7)
        
        if end_idx - start_idx < n:
            expand = (n - (end_idx - start_idx)) // 2 + 1
            start_idx = max(0, start_idx - expand)
            end_idx = min(len(similar_words), end_idx + expand)
        
        candidates_range = similar_words[start_idx:end_idx]
        if len(candidates_range) > n:
            candidate_words = random.sample(candidates_range, n)
        else:
            candidate_words = candidates_range
        
        final_candidates = self.get_related_words_from_wordnet(target_word, candidate_words, m)
        
        # generate negative samples 
        negative_samples = []
        for word in final_candidates:
            sample = original_sentence.replace(target_word, word)
            negative_samples.append({
                "original_word": target_word,
                "replaced_word": word,
                "original_pos": self.nlp(target_word)[0].tag_,
                "replaced_pos": self.nlp(word)[0].tag_,
                "sentence": sample
            })
        
        return negative_samples
    
    def save_samples(self, samples, is_positive=True, filename="samples.json"):
        target_dir = self.positive_dir if is_positive else self.negative_dir
        file_path = os.path.join(target_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=4)
        print('save completed')



if __name__ == "__main__":
    GLOVE_PATH = "glove.6B.100d.txt" 

    generator = SampleGenerator(GLOVE_PATH)
    contributions = generator.load_contributions()

    # example input
    original_sentence = "This movie is fantastic! The acting was superb and the plot was engaging."
    
    positive_samples = generator.generate_positive_samples(
        original_sentence, 
        contributions, 
        top_k=50, 
        n=10, 
        m=5
    )
    
    generator.save_samples(positive_samples, is_positive=True)
    
    negative_samples = generator.generate_negative_samples(
        original_sentence, 
        contributions, 
        total_candidates=100, 
        n=10, 
        m=5
    )
    
    generator.save_samples(negative_samples, is_positive=False)

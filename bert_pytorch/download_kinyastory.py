import requests
import os
import json
from pathlib import Path
import pandas as pd
import json
import nltk
from nltk.corpus import words
nltk.download('words')
class DownloadKinyastory:
    def __init__(self):
        self.url = "https://storytelling-m5a9.onrender.com/stories"
        self.save_dir = 'kinyastory_data'
        self.save_path = os.path.join(os.getcwd(), self.save_dir)
        self.file_path = os.path.join(self.save_path, "stories.json")

        # if os.path.exists(self.file_path):
        #     with open(self.file_path, 'r') as f:
        #         self.data = json.load(f)
        #     print(f"Loaded {len(self.data)} stories from {self.file_path}")
        # else:
        response = requests.get(self.url)
        if response.status_code == 200:
            self.data = response.json()
        else:
            raise Exception(f"Failed to fetch data from {self.url}. Status code: {response.status_code}")
        
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    def download(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump(self.data, f)
            print(f"Downloaded {len(self.data)} stories to {self.file_path}")



class MakeUsableDataset:
    def __init__(self, stories,directory):
        self.stories = stories
        self.save_path = os.path.join(os.getcwd(),directory)

    def create_csv(self, filename):
        # with open('../nltk_data/corpora/words/en', 'r') as f:
        #     words = f.read().splitlines()
        self.common_english_words = set(words.words())
        other_english_words = ['foreign', 'fiction', 'children', 'all','young', 'adult', 'non', 'fiction', 'mystery', 'thriller', 'romance', 'science', 'fiction', 'fantasy', 'horror', 'humor', 'comics', 'graphic', 'novels', 'historical', 'fiction', 'history', 'biography', 'memoir', 'poetry', 'essays', 'short', 'stories', 'self', 'help', 'true', 'crime', 'paranormal', 'religious', 'spiritual', 'inspirational', 'health', 'fitness', 'cookbooks', 'food', 'wine', 'crafts', 'hobbies', 'lifestyle', 'personal', 'growth', 'business', 'investing', 'economics', 'management', 'leadership', 'professional', 'technical', 'reference', 'science', 'nature', 'travel', 'sports', 'outdoors', 'romance', 'science', 'fiction', 'fantasy', 'horror', 'humor', 'comics', 'graphic', 'novels', 'historical', 'fiction', 'history', 'biography', 'memoir', 'poetry', 'essays', 'short', 'stories', 'self', 'help', 'true', 'crime', 'paranormal', 'religious', 'spiritual', 'inspirational', 'health', 'fitness', 'cookbooks', 'food', 'wine', 'crafts', 'hobbies', 'lifestyle', 'personal', 'growth', 'business', 'investing', 'economics', 'management', 'leadership', 'professional', 'technical', 'reference', 'science', 'nature', 'travel', 'sports', 'outdoors', 'romance', 'science', 'fiction', 'fantasy', 'horror', 'humor', 'comics', 'graphic', 'novels', 'historical', 'fiction', 'history', 'biography', 'memoir', 'poetry', 'essays', 'short', 'stories', 'self', 'help', 'true', 'crime', 'paranormal', 'religious', 'spiritual', 'inspirational', 'health', 'fitness', 'cookbooks', 'food', 'wine', 'crafts', 'hobbies', 'lifestyle', 'personal', 'growth', 'business', 'investing', 'economics','man','men','male','female']
        # test if the words are loaded
        print("Loaded", len(self.common_english_words), "words")
        print("First 10 words:", list(self.common_english_words)[:10])
        data = []
        for story in self.stories:
            demographic_values = ''
            if story['demographic']:
                try:
                    demographic_dict = json.loads(story['demographic'])
                    if isinstance(demographic_dict, dict):
                        demographic_values = ' '.join(map(str, demographic_dict.values()))
                except json.JSONDecodeError:
                    #demographic_values = ' '.join([ word for word in story['demographic'].split(',') if word not in self.common_english_words])
                    pass

            story['genre'] = ' '.join([word for word in story['genre'].split(' ') if word.lower() not in self.common_english_words and word.lower() not in other_english_words]) if story['genre'] is not None else ''
            story['themes'] = ' '.join([word for word in story['themes'].split(' ') if word.lower() not in self.common_english_words and word.lower() not in other_english_words]) if story['themes'] is not None else ''

            story_input = f"{story['story_title']} {story['genre']} {story['themes']}"
            
            data.append({
                'storyId': story['story_id'],
                'story_input': story_input,
                'story_output': story['story_text']
            })
        df = pd.DataFrame(data)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        df.to_csv(os.path.join(self.save_path, filename), index=False)
    def _len_(self):
        return len(self.stories)
    
if __name__ == "__main__":
    kinyastory = DownloadKinyastory()
    kinyastory.download()
    stories = kinyastory.data
    dataset = MakeUsableDataset(stories, 'kinyastory_data')
    dataset.create_csv('kinyastory.csv')
    print(f"Created kinyastory.csv with {len(stories)} stories")
    print(f"Saved in {os.getcwd()}")
   


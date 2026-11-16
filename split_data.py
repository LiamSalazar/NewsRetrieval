import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/news_raw_corpus.tsv', sep='\t')

df = df.dropna(subset=['category']).copy()

body_col = 'description/body' if 'description/body' in df.columns else 'description'
df['text'] = df['title'].fillna('') + ' ' + df[body_col].fillna('')
df['text'] = df['text'].str.strip()

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    shuffle=True,
    random_state=0,
    stratify=df['category']
)
dev_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    shuffle=True,
    random_state=0,
    stratify=temp_df['category']
)
train_df['split'] = 'train'
dev_df['split'] = 'dev'
test_df['split'] = 'test'

labeled_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
labeled_df = labeled_df[['id', 'text', 'category', 'split']].rename(columns={'id': 'document_id'})

labeled_df.to_csv('labeled_documents.tsv', sep='\t', index=False)

print("Distribution by split:")
print(labeled_df['split'].value_counts())
print("\nCategory distribution by split:")
print(pd.crosstab(labeled_df['category'], labeled_df['split']))
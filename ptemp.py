import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, query, value, key, pad_mask=None, att_mask=None):
        query = query.permute(1, 0, 2)  # (Q,B,E)
        value = value.permute(1, 0, 2)  # (V,B,E))
        key = key.permute(1, 0, 2)      # (K,B,E)
        embed, att = self.attention(query, value, key, key_padding_mask=pad_mask, attn_mask=att_mask)  # (Q,B,E), (B,Q,V)

        embed = self.normalize(embed + query)  # (Q,B,E)
        embed = embed.permute(1, 0, 2)  # (B,Q,E)
        return embed, att  # (B,Q,E), (B,Q,V)

class Classifier(nn.Module):
    def __init__(self, num_topics, num_states, cnn=None, tnn=None,
                 fc_features=2048, embed_dim=128, num_heads=1, dropout=0.1):
        super().__init__()

        # For img & txt embedding and feature extraction
        self.cnn = cnn
        self.tnn = tnn
        self.img_features = nn.Linear(fc_features, num_topics * embed_dim) if cnn != None else None
        self.txt_features = MultiheadAttention(embed_dim, num_heads, dropout) if tnn != None else None

        # For classification
        self.topic_embedding = nn.Embedding(num_topics, embed_dim)
        self.state_embedding = nn.Embedding(num_states, embed_dim)
        self.attention = MultiheadAttention(embed_dim, num_heads)
        # self.label_embedding = nn.Embedding(num_topics, embed_dim)

        # Some constants
        self.num_topics = num_topics
        self.num_states = num_states
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, img=None, txt=None, lbl=None, txt_embed=None, pad_mask=None, pad_id=3, threshold=0.5,
                get_embed=False, get_txt_att=False):
        # --- Get img and txt features ---
        if img != None:  # (B,C,W,H) or ((B,V,C,W,H), (B,V))
            img_features, wxh_features = self.cnn(img)  # (B,F), (B,F,W,H) ([8, 1024]),([8, 1024, 8, 8])
            img_features = self.dropout(img_features)  # (B,F)
        if txt != None:
            if pad_id >= 0 and pad_mask == None:
                pad_mask = (txt == pad_id)
            txt_features = self.tnn(token_index=txt, pad_mask=pad_mask)  # (B,L,E) ([8, 1000, 256])
        elif txt_embed != None:
            txt_features = self.tnn(token_embed=txt_embed, pad_mask=pad_mask)  # (B,L,E)

        # --- Fuse img and txt features ---
        if img != None and (txt != None or txt_embed != None):
            topic_index = torch.arange(self.num_topics).unsqueeze(0).repeat(img_features.shape[0], 1).to(img_features.device)  # (B,T)
            state_index = torch.arange(self.num_states).unsqueeze(0).repeat(img_features.shape[0], 1).to(img_features.device)  # (B,C)
            topic_embed = self.topic_embedding(topic_index)  # (B,T,E),(8,114,256)
            state_embed = self.state_embedding(state_index)  # (B,C,E),(8,2,256)
            img_features = self.img_features(img_features).view(img_features.shape[0], self.num_topics, -1)  # (B,F) --> (B,T*E) --> (B,T,E):([8, 1024])-->([8, 29184])-->([8, 114, 256])
            # img_features_torch.Size([8, 1024])-->([8, 114, 256])
            txt_features, txt_attention = self.txt_features(img_features, txt_features, txt_features, pad_mask)  # (B,T,E), (B,T,L) topic_embed=H  txt_features=Q
            img_features, txt_attention = self.txt_features(img_features, img_features, txt_features, pad_mask)
            # ([8, 114, 256]),([8, 114, 1000])= self.txt_features((8, 1000, 256), (8,114,256), (8, 1000))


            final_embed = self.normalize(img_features + txt_features)  # (B,T,E) final_embed=D(fused)


        elif img != None:
            topic_index = torch.arange(self.num_topics).unsqueeze(0).repeat(img_features.shape[0], 1).to(img_features.device)  # (B,T)
            state_index = torch.arange(self.num_states).unsqueeze(0).repeat(img_features.shape[0], 1).to(img_features.device)  # (B,C)
            topic_embed = self.topic_embedding(topic_index)  # (B,T,E)
            state_embed = self.state_embedding(state_index)  # (B,C,E)
            img_features = self.img_features(img_features).view(img_features.shape[0], self.num_topics,-1)  # (B,F) --> (B,T*E) --> (B,T,E)
            final_embed = img_features  # (B,T,E)

        elif txt != None or txt_embed != None:
            topic_index = torch.arange(self.num_topics).unsqueeze(0).repeat(txt_features.shape[0], 1).to(txt_features.device)  # (B,T)
            state_index = torch.arange(self.num_states).unsqueeze(0).repeat(txt_features.shape[0], 1).to(txt_features.device)  # (B,C)
            topic_embed = self.topic_embedding(topic_index)  # (B,T,E)
            state_embed = self.state_embedding(state_index)  # (B,C,E)

            txt_features, txt_attention = self.txt_features(txt_features, topic_embed, pad_mask)  # (B,T,E), (B,T,L)
            final_embed = txt_features  # (B,T,E) final_embed=D(fused)

        else:
            raise ValueError('img and (txt or txt_embed) must not be all none')

        # Classifier output
        emb, att = self.attention(state_embed, final_embed)  # (B,T,E), (B,T,C)

        if lbl != None:  # Teacher forcing
            emb = self.state_embedding(lbl)  # (B,T,E)
        else:
             emb = self.state_embedding((att[:, :, 1] > threshold).long())  # (B,T,E)

        if get_embed:
            return att, final_embed, emb  # (B,T,C), (B,T,E)
        elif get_txt_att and (txt != None or txt_embed != None):
            return att, txt_attention  # (B,T,C), (B,T,L)
        else:
            return att  # (B,T,C)
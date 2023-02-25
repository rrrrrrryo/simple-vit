import torch


from multi_head_attention import MultiHeadSelfAttention


class VitEncoderBlock(torch.nn.Module):
    def __init__(self, emb_dim: int=384, head: int=8, hidden_dim: int=384*4, dropout: float=0.) -> None:
        """
        args:
            emb_dim: 埋め込み後のベクトル長
            head: ヘッド数
            hidden_dim: Encoder BlockのMLPにおける中間層のベクトル長
            drop_out: ドロップアウト率
        """
        super().__init__()

        self.layer_norm_1 = torch.nn.LayerNorm(emb_dim)
        self.mhsa = MultiHeadSelfAttention(
            emb_dim=emb_dim,
            head=head,
            dropout=dropout
        )

        self.layer_norm_2 = torch.nn.LayerNorm(emb_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, emb_dim),
            torch.nn.Dropout(dropout)
        )


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        args:
            z: Encoder Blockへの入力。 shape(B, N, D)
                B: バッチサイズ, N: トークン数, D: ベクトルの長さ
        
        return:
            out: Encoder Blockの出力。 shape(B, N, D)
        """
        out = self.mhsa(self.layer_norm_1(z)) + z

        out = self.mlp(self.layer_norm_2(out)) + out

        return out
    

if __name__ == '__main__':

    z_0 = torch.rand(2, 5, 384)
    
    enc = VitEncoderBlock()
    z_1 = enc(z_0)

    print(z_1.shape)

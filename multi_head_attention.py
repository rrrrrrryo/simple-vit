import torch
import torch.nn.functional as F

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, emb_dim: int=384, head: int=3, dropout: float=0.) -> None:
        """
        args:
            emb_dim: 埋め込みベクトルの長さ
            head: ヘッドの数
            dropout: ドロップアウト率
        """
        super().__init__()

        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim**0.5

        self.w_q = torch.nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.w_k = torch.nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.w_v = torch.nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        self.attention_dropout = torch.nn.Dropout(dropout)

        self.w_o = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.Dropout(dropout)
        )


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        args:
            z: Multi Head Self Attentionへの入力。 shape(B, N, D)
                B: バッチサイズ, N: トークンの数, D: ベクトルの長さ
        
        return:
            out: Multi Head Self Attentionの出力。 shape(B, N, D)
        """
        batch_size, num_patch, _ = z.size()

        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # q, k, vをMulti Headの数に分ける
        # (B, N, D) -> (B, N, h, D/h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        # (バッチサイズ, ヘッド, トークン数, バッチのベクトル)
        # (B, N, h, D/h) -> (B, h, N, D/h)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # keyを転置
        # (B, h, N, D/h) -> (B, h, D/h, N)
        k_T = k.transpose(2, 3)
        
        # dot product attention
        # (B, h, N, D/h) @ (B, h, D/h, N)
        attention = F.softmax(((q@k_T)/self.sqrt_dh), dim=-1)
        
        attention = self.attention_dropout(attention)

        # 加重和、qkとvalueをかける
        # (B, h, N, N) @ (B, h, N, D/h) -> (B, h, N, D/h)
        out = attention @ v

        # (B, h, N, D/h) -> (B, N, h, D/h)
        # (B, N, h, D/h) -> (B, N, D)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        # 出力層
        out = self.w_o(out)

        return out


if __name__ == '__main__':
    
    z_0 = torch.rand(2, 5, 384).cuda()
    mhsa = MultiHeadSelfAttention()
    mhsa = mhsa.cuda()
    out = mhsa(z_0)
    print(out.shape)

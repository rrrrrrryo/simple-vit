import torch

from input_layer import VitInputLayer
from encoder import VitEncoderBlock


class Vit(torch.nn.Module):
    def __init__(self,
                 in_channels: int=3, 
                 num_classes: int=10, 
                 emb_dim: int=384, 
                 num_patch_row: int=2, 
                 image_size: int=32,
                 num_blocks: int=7,
                 head: int=8,
                 hidden_dim: int=384+4,
                 dropout: float=0.
                 ) -> None:
        """
        args:
            in_channels: 入力画像のチャンネル数
            num_classes: 画像分類のクラス数
            emb_dim: 埋め込み後のベクトル長
            num_patch_row: 1辺のパッチ数
            image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定
            num_blocks: Encoder Blockの数
            head: ヘッドの数
            hidden_dim: Encoder BlockのMLPにおける中間層のベクトル長
            dropout: ドロップアウト率
        """
        super().__init__()

        self.input_layer = VitInputLayer(
            in_cannels=in_channels,
            emb_dim=emb_dim,
            num_patch_row=num_patch_row,
            image_size=image_size
        )

        self.encoder = torch.nn.Sequential(*[
            VitEncoderBlock(
                emb_dim=emb_dim,
                head=head,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            for i in range(num_blocks)]
        )

        self.mlp_head = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_dim),
            torch.nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: ViTへの入力画像。shape(B, C, H, W)
                B: バッチサイズ, C: チャンネル数, H: Height, W: Width

        return:
            out: ViTの出力。shape(B, M)
                B: バッチサイズ, M: クラス数
        """
        out = self.input_layer(x)
        out = self.encoder(out)

        cls_token = out[:, 0]

        pred = self.mlp_head(cls_token)

        return pred
    

if __name__ == '__main__':
    num_class = 10
    batch_size, channel, height, width = 2, 3, 32, 32

    x = torch.rand(batch_size, channel, height, width)

    vit = Vit(in_channels=channel, num_classes=num_class)
    pred = vit(x)

    print(pred.shape)

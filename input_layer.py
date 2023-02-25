import torch

class VitInputLayer(torch.nn.Module):
    def __init__(self, in_cannels: int=3, emb_dim: int=384, num_patch_row: int=2, image_size: int=32) -> None:
        """
        args:
            in_cannels: 入力画像のチャンネル数
            emb_dim: 埋め込み後のベクトルの長さ
            num_patch_row: 高さ方向のパッチの数
            image_size: 入力画像の大きさ
        """
        super().__init__()
        
        self.in_cannels = in_cannels
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.image_size = image_size

        self.num_patch = self.num_patch_row **2

        self.patch_size = int(self.image_size // self.num_patch_row)

        self.patch_emb_layer = torch.nn.Conv2d(
            in_channels=self.in_cannels,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        self.cls_token = torch.nn.Parameter(
            torch.rand(1, 1, self.emb_dim)
        )

        self.pos_emb = torch.nn.Parameter(
            torch.rand(1, self.num_patch+1, self.emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: 入力画像, shape(B, C, H, W)
                B: Batch, C: channels, H: Height, W: Width

        return:
            z_0: Vitへの入力, shape(B, N, D)
                B: Batch, N: token, D: Embeded vec len
        """
        z_0 = self.patch_emb_layer(x)
        
        z_0 = z_0.flatten(2)

        z_0 = z_0.transpose(1, 2)

        z_0 = torch.cat(
            [self.cls_token.repeat(repeats=(x.size(0), 1, 1)), z_0],
            dim=1
        )
        
        z_0 = z_0 + self.pos_emb

        return z_0

if __name__ == '__main__':
    batch_size, channel, height, width = 2, 3, 32, 32

    x = torch.rand(batch_size, channel, height, width)
    input_layer = VitInputLayer(num_patch_row=2)

    if torch.cuda.is_available():
        x = x.cuda()
        input_layer = input_layer.cuda()
        
    z_0 = input_layer(x)
    print(z_0.shape)


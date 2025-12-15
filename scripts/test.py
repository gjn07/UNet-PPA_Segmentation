import cv2
from models.unet import *
from data.data import *

num_classes = 1 + 1

if __name__ == '__main__':
    net = UNet(num_classes).cuda()

    weights = '../checkpoints/unet.pth'
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successfully')
    else:
        print('no loading')

    _input = r'../data/processed/val/images/r1_Im069.png'

    img = keep_image_size_open_rgb(_input)
    img_data = transform(img).cuda()
    img_data = torch.unsqueeze(img_data, dim=0)
    net.eval()
    out = net(img_data)
    out = torch.argmax(out, dim=1)
    out = torch.squeeze(out, dim=0)
    out = out.unsqueeze(dim=0)
    print(set((out).reshape(-1).tolist()))
    out = (out).permute((1, 2, 0)).cpu().detach().numpy()
    cv2.imwrite('../runs/outputs/result.png', out)
    cv2.imshow('out', out * 255.0)
    cv2.waitKey(0)
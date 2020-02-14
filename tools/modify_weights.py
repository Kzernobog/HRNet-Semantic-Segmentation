import torch
import torch.nn as nn

if __name__ == "__main__":

    # change aux_head.3.weight from ([19, 720, 1, 1]) to ([3, 720, 1, 1])
    # change aux_head.3.bias from ([19]) to ([3])
    pretrained = '/home/aditya/small_obstacle_ws/HRNet/pretrained/hrnet_cs_8090_torch11.pth'
    pretrained_dict = torch.load(pretrained, map_location={'cuda:0': 'cpu'})
    pretrained_dict = {k.replace('last_layer', 'aux_head').replace('model.', ''): v for k, v in pretrained_dict.items()}
    print(pretrained_dict['aux_head.3.weight'].shape)
    print(pretrained_dict['aux_head.3.bias'].shape)
    final_layer_shape = [3, 720, 1, 1]
    pretrained_dict['aux_head.3.weight'] = nn.init.kaiming_normal_(torch.empty(final_layer_shape))
    pretrained_dict['aux_head.3.bias'] = nn.init.constant_(torch.empty(final_layer_shape[0]),0)
    new_weight = '/home/aditya/small_obstacle_ws/HRNet/pretrained/small_obs_pretrained.pth'
    with open(new_weight, "wb") as f:
        torch.save(pretrained_dict, f)
    # for k, v in pretrained_dict.items():
    #     print(k)
